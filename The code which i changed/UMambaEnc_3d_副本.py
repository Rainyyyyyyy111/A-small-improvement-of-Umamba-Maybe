import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba # Make sure mamba_ssm is installed: pip install mamba_ssm
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

# Helper Layer for Upsampling
class UpsampleLayer(nn.Module):
    """
    Upsamples the input tensor and applies a 1x1 convolution.
    """
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

# Mamba Layer Wrapper
class MambaLayer(nn.Module):
    """
    Wrapper for the Mamba State Space Model layer.
    Includes Layer Normalization.
    Handles potential float16 input by casting to float32 for Mamba computation.
    Allows switching between patch tokens and channel tokens.
    """
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        print(f"Initializing MambaLayer: dim={dim}, d_state={d_state}, channel_token={channel_token}") # Log initialization params
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.channel_token = channel_token # whether to use channel as tokens

    def forward_patch_token(self, x):
        """ Process features where channels are the model dimension (C) and spatial dims are flattened into tokens (N). """
        B, d_model = x.shape[:2]
        assert d_model == self.dim, f"Input channel dimension {d_model} does not match MambaLayer dimension {self.dim}"
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # Reshape input for Mamba: (B, C, D, H, W) -> (B, D*H*W, C)
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # Reshape back to original image dimensions: (B, D*H*W, C) -> (B, C, D, H, W)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        """ Process features where channels are the tokens (N) and spatial dims are flattened into the model dimension (C). """
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"Flattened spatial dimension {d_model} does not match MambaLayer dimension {self.dim}"
        img_dims = x.shape[2:]
        # Reshape input for Mamba: (B, C, D, H, W) -> (B, C, D*H*W)
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"Unexpected flattened dimension: {x_flat.shape[2]}, expected {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # Reshape back to original image dimensions: (B, C, D*H*W) -> (B, C, D, H, W)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast(enabled=False) # Mamba typically requires float32
    def forward(self, x):
        # Ensure input is float32 for Mamba
        original_dtype = x.dtype
        if original_dtype == torch.float16 or original_dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        # Cast back to original dtype if necessary
        if out.dtype != original_dtype:
             out = out.type(original_dtype)

        return out


# Residual Block used in Encoder/Decoder
class BasicResBlock(nn.Module):
    """
    A basic residual block with two convolutions, normalization, and activation.
    Includes an optional 1x1 convolution for the identity connection if dimensions change.
    """
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        # 1x1 convolution for the identity connection if channels or spatial dimensions change
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            # If no 1x1 conv needed, identity mapping is direct only if channels and stride match
            self.conv3 = nn.Identity() if all(s == 1 for s in maybe_convert_scalar_to_list(conv_op, stride)) and input_channels == output_channels else None
            if self.conv3 is None:
                 # This path should ideally be handled by setting use_1x1conv=True externally
                 print(f"Warning: BasicResBlock identity path needs handling for stride={stride} or channel change "
                       f"({input_channels} -> {output_channels}) without use_1x1conv=True. Creating 1x1 conv.")
                 self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)


    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))

        if self.conv3:
            identity = self.conv3(identity) # Apply transformation if needed

        # Ensure identity and y have compatible shapes for addition
        if identity.shape != y.shape:
             print(f"Warning: Shape mismatch in BasicResBlock residual connection. "
                   f"Identity shape: {identity.shape}, Main path shape: {y.shape}. Check strides/padding/convs.")
             # Attempt to handle simple cases, but this might indicate an issue
             if identity.shape[2:] == y.shape[2:] and identity.shape[1] != y.shape[1]:
                 print("  -> Channel mismatch without 1x1 conv specified.")
             elif identity.shape[1] == y.shape[1] and identity.shape[2:] != y.shape[2:]:
                  print("  -> Spatial dimension mismatch without 1x1 conv specified.")
             # Add a fallback or raise error if necessary
             # For now, we'll proceed, but this might cause errors later
        try:
             y += identity # Add the identity (potentially transformed)
        except RuntimeError as e:
            print(f"Error during residual addition in BasicResBlock: {e}")
            print(f"Identity shape: {identity.shape}, Main path shape: {y.shape}")
            raise e

        return self.act2(y)

# Encoder combining Residual Blocks and Mamba Layers
class ResidualMambaEncoder(nn.Module):
    """
    Encoder part of the U-Mamba architecture. Uses Residual Blocks for feature extraction
    and incorporates Mamba layers sequentially after convolutional blocks in each stage.
    *Optimization*: Increased d_state for Mamba layers.
    """
    def __init__(self,
                 input_size: Tuple[int, ...], # Needed to determine channel_token mode
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv', # Not directly used here as strides handle downsampling
                 mamba_d_state: int = 32 # Optimized Mamba state dimension
                 ):
        super().__init__()
        # Parameter handling and validation (same as previous versions)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [1] + [strides] * (n_stages - 1) # Assume first stride is 1
        elif isinstance(strides[0], int):
             strides = [s for s in strides]
        # ... (rest of stride handling) ...

        assert len(kernel_sizes) == n_stages, "kernel_sizes must match n_stages"
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must match n_stages"
        assert len(features_per_stage) == n_stages, "features_per_stage must match n_stages"
        assert len(strides) == n_stages, "strides must match n_stages"

        # Determine if channel_token mode should be used based on feature map size vs channels
        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = list(input_size)
        for s in range(n_stages):
            current_stride = strides[s]
            if isinstance(current_stride, int):
                 current_stride = [current_stride] * len(feature_map_size)
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, current_stride)])
            feature_map_size = feature_map_sizes[-1]
            # Use channel token if the number of spatial features is less than or equal to the number of channels
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        print(f"Encoder feature_map_sizes: {feature_map_sizes}")
        print(f"Encoder do_channel_token: {do_channel_token}")

        # Convolution padding calculation (same as previous versions)
        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
             if isinstance(krnl, int):
                 self.conv_pad_sizes.append([krnl // 2] * conv_op.ndim)
             else:
                 self.conv_pad_sizes.append([i // 2 for i in krnl])


        # Build the stem (initial block)
        stem_channels = features_per_stage[0] if stem_channels is None else stem_channels
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op, input_channels=input_channels, output_channels=stem_channels,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0], stride=1, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=(input_channels != stem_channels)
            ),
            *[BasicBlockD(
                conv_op=conv_op, input_channels=stem_channels, output_channels=stem_channels,
                kernel_size=kernel_sizes[0], stride=1, conv_bias=conv_bias, norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
              ) for _ in range(n_blocks_per_stage[0] - 1)]
        )
        current_input_channels = stem_channels

        # Build the main encoder stages with Mamba layers
        stages = []
        mamba_layers = []
        for s in range(n_stages):
             current_stride = strides[s]
             if isinstance(current_stride, int):
                 current_stride = [current_stride] * conv_op.ndim

             # Convolutional part of the stage
             stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    input_channels=current_input_channels, output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s], padding=self.conv_pad_sizes[s], stride=current_stride,
                    use_1x1conv=(current_input_channels != features_per_stage[s] or any(st > 1 for st in current_stride)),
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                ),
                *[BasicBlockD(
                    conv_op=conv_op, input_channels=features_per_stage[s], output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s], stride=1, conv_bias=conv_bias, norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                  ) for _ in range(n_blocks_per_stage[s] - 1)]
            )
             stages.append(stage)

             # Mamba layer for this stage (applied after convolutions)
             mamba_layers.append(
                MambaLayer(
                    # Dimension depends on whether we use channel tokens
                    dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                    d_state=mamba_d_state, # Use optimized state dimension
                    channel_token=do_channel_token[s]
                )
            )
             current_input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # Store necessary attributes for potential use by decoder
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.conv_pad_sizes = self.conv_pad_sizes

    def forward(self, x):
        skips = []
        if self.stem is not None:
            x = self.stem(x)

        # Process through each stage: Conv -> Mamba
        for s in range(len(self.stages)):
            x = self.stages[s](x) # Apply convolutional blocks
            x = self.mamba_layers[s](x) # Apply Mamba layer
            if self.return_skips:
                skips.append(x)

        if self.return_skips:
            return skips
        else:
            return x # Return only the final output

    def compute_conv_feature_map_size(self, input_size):
         # This function is used by the framework to estimate memory consumption
         # Requires careful implementation based on the blocks used.
         pass # Placeholder


# Standard UNet Decoder with Residual Blocks
class UNetResDecoder(nn.Module):
    """
    Standard nnU-Net V2 Residual Decoder.
    """
    def __init__(self,
                 encoder, # Takes the encoder instance as input
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder # Store encoder reference
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        # Ensure n_conv_per_stage has the correct length
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have resolution stages - 1"

        stages = []
        upsample_layers = []
        seg_layers = [] # Segmentation layers for deep supervision

        # Loop through encoder stages in reverse order
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            # Upsampling layer
            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest' # Or 'trilinear'
            ))

            # Determine kernel size and padding for this decoder stage
            current_kernel_size = encoder.kernel_sizes[-(s + 1)]
            current_padding = encoder.conv_pad_sizes[-(s + 1)]
            if isinstance(current_kernel_size, int):
                 current_kernel_size = [current_kernel_size] * encoder.conv_op.ndim
            if isinstance(current_padding, int):
                 current_padding = [current_padding] * encoder.conv_op.ndim


            # Decoder stage blocks
            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op, norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip, # Concatenated features
                    output_channels=input_features_skip,
                    kernel_size=current_kernel_size, padding=current_padding, stride=1,
                    use_1x1conv=True # Channels change due to concatenation
                ),
                *[BasicBlockD(
                    conv_op=encoder.conv_op, input_channels=input_features_skip, output_channels=input_features_skip,
                    kernel_size=current_kernel_size, stride=1, conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs
                  ) for _ in range(n_conv_per_stage[s-1] - 1)]
            ))

            # Segmentation layer for deep supervision
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
         # This function is used by the framework to estimate memory consumption
         pass # Placeholder


# Main UMambaEnc Model
class UMambaEnc(nn.Module):
    """
    UMambaEnc architecture combining a ResidualMambaEncoder and a UNetResDecoder.
    Mamba layers are applied within the encoder stages.
    """
    def __init__(self,
                 input_size: Tuple[int, ...], # Needed for encoder init
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], # Used for encoder
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], # Used for decoder
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None, # Not used
                 dropout_op_kwargs: dict = None, # Not used
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None,
                 mamba_d_state: int = 32 # Allow passing optimized d_state
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage # Alias for clarity
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # --- Architecture Modification (Optional Reduction) ---
        # for s in range(math.ceil(n_stages / 2), n_stages):
        #     n_blocks_per_stage[s] = 1
        # for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
        #     n_conv_per_stage_decoder[s] = 1
        # --- End Modification ---

        # Architecture checks
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must match n_stages"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must be n_stages - 1"

        # Initialize Encoder with Mamba layers and optimized d_state
        self.encoder = ResidualMambaEncoder(
            input_size=input_size, # Pass input size
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True, # Need skips for decoder
            stem_channels=stem_channels,
            mamba_d_state=mamba_d_state # Pass optimized d_state
        )

        # Initialize Decoder
        self.decoder = UNetResDecoder(
            self.encoder, # Pass encoder instance
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        # This function is used by the framework to estimate memory consumption
        pass # Placeholder


# Function to get the UMambaEnc model based on plans
def get_umamba_enc_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    """
    Instantiates the UMambaEnc architecture based on parameters from the plans for 3D data.
    Includes optimized Mamba d_state.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    assert dim == 3, "This function is for 3D UMambaEnc only."
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    # Define kwargs for the UMambaEnc based on nnU-Net defaults
    kwargs = {
        'input_size': configuration_manager.patch_size, # Pass patch size to encoder
        'conv_bias': True,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'mamba_d_state': 32 # Optimized d_state for Mamba layers in encoder
    }

    # Get number of blocks/convs per stage from the configuration manager
    conv_or_blocks_per_stage_encoder = configuration_manager.n_conv_per_stage_encoder
    conv_or_blocks_per_stage_decoder = configuration_manager.n_conv_per_stage_decoder

    # Instantiate the UMambaEnc model
    model = UMambaEnc(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        n_conv_per_stage=conv_or_blocks_per_stage_encoder, # Pass encoder config
        n_conv_per_stage_decoder=conv_or_blocks_per_stage_decoder, # Pass decoder config
        deep_supervision=deep_supervision,
        **kwargs # Pass additional shared kwargs
    )

    # Apply weight initialization
    model.apply(InitWeights_He(1e-2))
    # model.apply(init_last_bn_before_add_to_0) # Apply if using residual blocks with BN before add

    print("UMambaEnc 3D (Optimized Encoder d_state=32): {}".format(model.__class__.__name__))

    return model
