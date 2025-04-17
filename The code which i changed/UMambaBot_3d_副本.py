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
    """
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )

    @autocast(enabled=False) # Mamba typically requires float32
    def forward(self, x):
        # Ensure input is float32 for Mamba
        original_dtype = x.dtype
        if original_dtype == torch.float16 or original_dtype == torch.bfloat16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        assert C == self.dim, f"Input channel dimension {C} does not match MambaLayer dimension {self.dim}"
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # Reshape input for Mamba: (B, C, D, H, W) -> (B, D*H*W, C)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # Reshape back to original image dimensions: (B, D*H*W, C) -> (B, C, D, H, W)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

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
            # If no 1x1 conv needed, identity mapping is direct
            self.conv3 = nn.Identity() if stride == 1 and input_channels == output_channels else None
            if self.conv3 is None:
                 # Fallback if stride or channels change without explicit 1x1 conv (should be handled by use_1x1conv=True)
                 print(f"Warning: BasicResBlock identity path needs handling for stride={stride} or channel change "
                       f"({input_channels} -> {output_channels}) without use_1x1conv=True.")


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
                 # This case shouldn't happen if use_1x1conv is True when channels change
                 print("  -> Channel mismatch without 1x1 conv specified.")
             elif identity.shape[1] == y.shape[1] and identity.shape[2:] != y.shape[2:]:
                  # This case shouldn't happen if use_1x1conv is True when stride > 1
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

# Residual Mamba Block for the Bottleneck (Optimized d_state)
class ResidualMambaBlock(nn.Module):
    """
    A residual block incorporating the Mamba layer for the bottleneck.
    Applies Mamba and adds the result to the original input (identity).
    Includes optional normalization and activation.
    *Optimization*: Increased d_state for potentially better modeling capacity.
    """
    def __init__(self,
                 input_channels: int,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 d_state: int = 32, # Increased state dimension (default was 16)
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.input_channels = input_channels
        print(f"Initializing ResidualMambaBlock with d_state={d_state}") # Log the state dimension
        self.mamba_layer = MambaLayer(dim=input_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        # Optional: Add normalization and activation after Mamba + residual
        self.norm = norm_op(input_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        self.act = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.mamba_layer(x)
        x = x + identity # Residual connection
        x = self.norm(x)
        x = self.act(x)
        return x


# Standard UNet Encoder with Residual Blocks
class UNetResEncoder(nn.Module):
    """
    Standard nnU-Net V2 Residual Encoder.
    """
    def __init__(self,
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
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
             # If strides is an int, apply it to all stages except the first (which is usually 1)
            strides = [1] + [strides] * (n_stages - 1)
        elif isinstance(strides[0], int):
             # If strides is a list/tuple of ints, apply them directly
             strides = [s for s in strides] # Ensure it's a list
        elif isinstance(strides[0], (list, tuple)):
             # If strides is a list/tuple of lists/tuples (per-axis strides)
             pass # Keep as is
        else:
             raise TypeError("strides must be int, list/tuple of ints, or list/tuple of list/tuples of ints")


        # network configuration checks
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        # compute convolution padding
        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            if isinstance(krnl, int): # Handle int kernel size
                 self.conv_pad_sizes.append([krnl // 2] * conv_op.ndim)
            else:
                 self.conv_pad_sizes.append([i // 2 for i in krnl])


        # Build the stem (initial block)
        stem_channels = features_per_stage[0] if stem_channels is None else stem_channels
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1, # Stem stride is always 1
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=(input_channels != stem_channels) # Use 1x1 conv only if channels change
            ),
            *[
                BasicBlockD( # Use BasicBlockD for subsequent blocks in the stem
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1) # -1 because the first block is BasicResBlock
            ]
        )
        current_input_channels = stem_channels

        # Build the main encoder stages
        stages = []
        for s in range(n_stages):
             # Ensure strides[s] is a tuple/list of the correct dimension
             current_stride = strides[s]
             if isinstance(current_stride, int):
                 current_stride = [current_stride] * conv_op.ndim

             stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=current_input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=current_stride, # Apply stride here for downsampling
                    use_1x1conv=(current_input_channels != features_per_stage[s] or any(st > 1 for st in current_stride)), # Use 1x1 conv if channels change or stride > 1
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[
                    BasicBlockD( # Use BasicBlockD for subsequent blocks in the stage
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1) # -1 because the first block is BasicResBlock
                ]
            )
             stages.append(stage)
             current_input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages) # Use ModuleList to store stages
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # store attributes for the decoder
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.conv_pad_sizes = self.conv_pad_sizes # Store padding sizes

    def forward(self, x):
        # Store skip connections
        skips = []
        if self.stem is not None:
            x = self.stem(x)

        # Process through each stage
        for s in self.stages:
            x = s(x)
            if self.return_skips:
                 skips.append(x)

        if self.return_skips:
            return skips
        else:
            return x # Return only the final output if skips are not needed

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

        # Loop through encoder stages in reverse order (except the last one, which is the input to the first decoder stage)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s] # Features from the lower resolution stage
            input_features_skip = encoder.output_channels[-(s + 1)] # Features from the skip connection
            stride_for_upsampling = encoder.strides[-s] # Strides used in the corresponding encoder stage

            # Upsampling layer
            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip, # Upsample to match skip connection channels
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest' # Or 'trilinear'/'bilinear' depending on conv_op dimension
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
                BasicResBlock( # First block combines upsampled features and skip connection
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip, # Concatenated features
                    output_channels=input_features_skip,
                    kernel_size=current_kernel_size, # Use corresponding encoder kernel size
                    padding=current_padding, # Use corresponding encoder padding
                    stride=1, # Stride is always 1 in decoder blocks
                    use_1x1conv=True # Use 1x1 conv as channels change (concatenation doubles channels)
                ),
                *[ # Subsequent blocks in the stage
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=current_kernel_size,
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1) # -1 because the first block is BasicResBlock
                ]
            ))

            # Segmentation layer for deep supervision at this stage
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1] # Input from the bottleneck (last element of skips)
        seg_outputs = [] # Store deep supervision outputs

        # Iterate through decoder stages
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input) # Upsample features from below
            x = torch.cat((x, skips[-(s+2)]), 1) # Concatenate with skip connection
            x = self.stages[s](x) # Process through decoder blocks

            # Generate segmentation output if deep supervision is enabled or if it's the final stage
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1): # Final stage
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x # Update input for the next stage

        # Reverse the order of seg_outputs because they are collected from deep to shallow
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0] # Return only the final segmentation
        else:
            return seg_outputs # Return all deep supervision segmentations

    def compute_conv_feature_map_size(self, input_size):
         # This function is used by the framework to estimate memory consumption
         # Requires careful implementation based on the blocks used.
         pass # Placeholder


# Main UMambaBot Model
class UMambaBot(nn.Module):
    """
    UMambaBot architecture combining a UNetResEncoder, a ResidualMambaBlock bottleneck,
    and a UNetResDecoder.
    """
    def __init__(self,
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
                 dropout_op: Union[None, Type[_DropoutNd]] = None, # Not used in current blocks
                 dropout_op_kwargs: dict = None, # Not used
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage # Alias for clarity in encoder init
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # --- Architecture Modification ---
        # Original UMambaBot reduces block count in deeper encoder/decoder stages.
        # Keeping the original block counts for potentially better feature extraction.
        # --- End Modification ---

        # Architecture checks
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must match n_stages"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must be n_stages - 1"

        # Initialize Encoder
        self.encoder = UNetResEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage, # Use potentially modified block count
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True, # Crucial for U-Net structure
            stem_channels=stem_channels
        )

        # Initialize Bottleneck with ResidualMambaBlock (Optimized d_state)
        self.bottleneck = ResidualMambaBlock(
            input_channels=features_per_stage[-1], # Channels from the last encoder stage
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            d_state=32 # Increased state dimension
            # Add d_conv, expand if you want non-default Mamba params
        )

        # Initialize Decoder
        self.decoder = UNetResDecoder(
            self.encoder, # Pass encoder instance for skip connections info
            num_classes,
            n_conv_per_stage_decoder, # Use potentially modified block count
            deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        # Apply the ResidualMambaBlock bottleneck to the deepest features
        bottleneck_output = self.bottleneck(skips[-1])
        # Replace the last skip connection with the bottleneck output
        skips = skips[:-1] + [bottleneck_output]
        # Pass skips (with modified bottleneck) to the decoder
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
         # This function is used by the framework to estimate memory consumption
         # Requires careful implementation based on the blocks used.
         pass # Placeholder


# Function to get the UMambaBot model based on plans
def get_umamba_bot_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    """
    Instantiates the UMambaBot architecture based on parameters from the plans.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    assert dim == 3, "This function is for 3D UMambaBot only."
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    # Define kwargs for the UMambaBot based on nnU-Net defaults
    kwargs = {
        'conv_bias': True,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
    }

    # Get number of blocks/convs per stage from the configuration manager
    conv_or_blocks_per_stage_encoder = configuration_manager.n_conv_per_stage_encoder
    conv_or_blocks_per_stage_decoder = configuration_manager.n_conv_per_stage_decoder

    # Instantiate the UMambaBot model
    model = UMambaBot(
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
    # Special initialization for residual blocks if needed
    # Check if encoder/decoder actually use BasicResBlock or BasicBlockD if applying this
    # model.apply(init_last_bn_before_add_to_0) # Apply if using residual blocks with BN before add

    print("UMambaBot 3D (Optimized Bottleneck d_state=32): {}".format(model.__class__.__name__)) # Print model name

    return model
