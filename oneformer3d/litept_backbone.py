
"""
LitePT Backbone for ForestFormer

This module integrates LitePT into the ForestFormer architecture by wrapping 
the LitePT model and adapting its interface to match ForestFormer's expectations.

Design Choices:
1. Input: Adapts spconv.SparseConvTensor to LitePT's Point format
2. Output: Returns list of features per batch item, matching PTV3Backbone behavior
3. Voxelization: Uses grid_size parameter, handled by LitePT's serialization
4. Model: Wraps the standalone LitePT implementation

Author: Adapted for ForestFormer integration
"""

import sys
import os
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from addict import Dict

from mmdet3d.registry import MODELS

# Add LitePT to path dynamically
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # ForestFormer/oneformer3d -> ForestFormer -> forestpoint -> LitePT
# # Assuming LitePT is at ../../LitePT relative to this file
# litept_path = os.path.abspath(os.path.join(current_dir, "../../LitePT"))
# if litept_path not in sys.path:
#     sys.path.insert(0, litept_path)

IMPORT_ERROR = None
try:
    # from models.litept.litept import LitePT
    # # Point structure is needed for some operations if we were doing them manually, 
    # # but LitePT class handles most. Importing just in case.
    # from models.utils.structure import Point
    from .litept_v3m1 import PointTransformerV3 as LitePT
except ImportError as e:
    IMPORT_ERROR = e
    print(f"Warning: Could not import LitePT (PointTransformerV3). Error: {e}")
    # Define dummy classes to allow registration without crashing if deps are missing
    class LitePT(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(f"LitePT not found. Original error: {IMPORT_ERROR}")


@MODELS.register_module("LitePT")
class LitePTBackbone(nn.Module):
    """
    LitePT Backbone wrapper for ForestFormer.
    
    This wrapper adapts LitePT to work with ForestFormer's data flow:
    1. Takes input from ForestFormer's collate function (spconv.SparseConvTensor)
    2. Converts to LitePT's dictionary format
    3. Runs LitePT encoder-decoder
    4. Returns features in ForestFormer's expected format (list of tensors)
    
    Args:
        in_channels (int): Input feature channels (before padding/adapter). Default: 6
        out_channels (int): Output feature channels. Default: 72
        grid_size (float): Voxel size for grid sampling. Default: 0.02
        pretrained (str): Path to pretrained weights. Default: None
        input_mode (str): How to handle input channel mismatch. Options:
            - "zero_pad": Zero-pad input to match LitePT expectations (default)
            - "adapter": Use learnable adapter layer to project input -> LitePT input
            - "reinit_embedding": Re-initialize embedding layer to accept input channels
        # LitePT Args
        order (tuple): Serialization orders.
        stride (tuple): Downsampling strides.
        enc_depths (tuple): Encoder block depths.
        enc_channels (tuple): Encoder channel dims.
        enc_num_head (tuple): Encoder attention heads.
        enc_patch_size (tuple): Encoder patch sizes.
        enc_conv (tuple): Enable conv in encoder.
        enc_attn (tuple): Enable attn in encoder.
        enc_rope_freq (tuple): RoPE frequency for encoder.
        dec_depths (tuple): Decoder block depths.
        dec_channels (tuple): Decoder channel dims.
        dec_num_head (tuple): Decoder attention heads.
        dec_patch_size (tuple): Decoder patch sizes.
        dec_conv (tuple): Enable conv in decoder.
        dec_attn (tuple): Enable attn in decoder.
        dec_rope_freq (tuple): RoPE frequency for decoder.
        mlp_ratio (float): MLP expansion ratio.
        qkv_bias (bool): Enable bias in QKV.
        qk_scale (float): Scale for QK.
        attn_drop (float): Attention dropout.
        proj_drop (float): Projection dropout.
        drop_path (float): Stochastic depth rate.
        pre_norm (bool): Enable pre-norm.
        shuffle_orders (bool): Shuffle serialization orders.
        enc_mode (bool): Encoder-only mode.
    """
    
    def __init__(
        self,
        in_channels=6,
        out_channels=72,
        grid_size=0.02,
        pretrained=None,
        input_mode="zero_pad",  # "zero_pad", "adapter", or "reinit_embedding"
        # LitePT defaults (based on small-v1m1 config)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.out_channels = out_channels
        self.input_mode = input_mode
        
        # Determine LitePT input channels based on mode
        if input_mode == "reinit_embedding":
            self.litept_in_channels = in_channels
        else:
            # For zero_pad and adapter modes, assume default 6 channels or whatever enc_channels[0] expects?
            # Actually LitePT embedding layer takes `in_channels` arg.
            # Default scannet config uses in_channels=6 (rgb+norm probably).
            # ForestFormer usually inputs 3 (xyz) or 4 (xyz+i) etc.
            self.litept_in_channels = 6 # Default for LitePT configs
            
        # Build input adapter if using adapter mode
        if input_mode == "adapter":
            self.input_adapter = nn.Sequential(
                nn.Linear(in_channels, self.litept_in_channels),
                nn.LayerNorm(self.litept_in_channels),
            )
            print(f"[LitePT] Created input adapter: {in_channels}D -> {self.litept_in_channels}D")
        
        # Initialize LitePT
        self.litept = LitePT(
            in_channels=self.litept_in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            enc_cpe=enc_conv,
            enc_attn=enc_attn,
            enc_rope_freq=enc_rope_freq,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            dec_cpe=dec_conv,
            dec_attn=dec_attn,
            dec_rope_freq=dec_rope_freq,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enc_mode=enc_mode,
            grid_size=grid_size,
        )
        
        # Output projection/norm layer to ensure clean output
        # LitePT returns features from decoder directly.
        # ForestFormer expects a specific output dimension.
        # Typically the decoder output channel matches dec_channels[0].
        if dec_channels[0] != out_channels:
             self.output_proj = nn.Linear(dec_channels[0], out_channels)
        else:
            self.output_proj = nn.Identity()

        # Load pretrained weights if provided
        if pretrained is not None:
            self.load_pretrained(pretrained)
            
    def load_pretrained(self, pretrained_path):
        """Load pretrained weights."""
        print(f"[LitePT] Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Filter keys
        model_state_dict = self.litept.state_dict()
        loaded_state_dict = {}
        
        for k, v in state_dict.items():
            # Handle potential prefix differences (module., backbone., etc)
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("backbone."):
                clean_k = clean_k[9:]
                
            if clean_k in model_state_dict:
                if model_state_dict[clean_k].shape == v.shape:
                    loaded_state_dict[clean_k] = v
                else:
                    print(f"[LitePT] Skipping {clean_k} due to shape mismatch: {model_state_dict[clean_k].shape} vs {v.shape}")
                    
        # Load
        msg = self.litept.load_state_dict(loaded_state_dict, strict=False)
        print(f"[LitePT] Weights loaded: {msg}")

    def forward(self, x: spconv.SparseConvTensor):
        """
        Forward pass.
        
        Args:
            x (spconv.SparseConvTensor): Input sparse tensor.
            
        Returns:
            List[Tensor]: List of feature tensors, one per batch item.
        """
        # Extract data from sparse tensor
        features = x.features  # (N, in_channels)
        indices = x.indices    # (N, 4) - [batch_idx, x, y, z]
        batch_size = x.batch_size
        
        # Get batch indices and coordinates
        batch_indices = indices[:, 0].long()
        coords = indices[:, 1:].float()  # Grid coordinates
        
        # Handle input channels based on mode
        if self.input_mode == "adapter":
            features = self.input_adapter(features)
        elif self.input_mode == "zero_pad":
            if features.shape[1] < self.litept_in_channels:
                padding = torch.zeros(
                    features.shape[0], 
                    self.litept_in_channels - features.shape[1],
                    device=features.device,
                    dtype=features.dtype
                )
                features = torch.cat([features, padding], dim=1)
        # "reinit_embedding" assumes features passed as-is
        
        # Compute offset
        batch_counts = torch.bincount(batch_indices, minlength=batch_size)
        offset = torch.cumsum(batch_counts, dim=0).long()
        
        # Prepare dictionary for LitePT
        # LitePT expects: feat, grid_coord, offset/batch
        data_dict = {
            "feat": features,
            "grid_coord": coords.int(),
            "coord": coords * self.grid_size, # Approx original coords if not available
            "batch": batch_indices,
            "offset": offset,
            "grid_size": self.grid_size
        }
        
        # Run LitePT
        point = self.litept(data_dict)
        
        # Output features
        output_feat = point.feat
        output_feat = self.output_proj(output_feat)
        
        # Split features by batch to match ForestFormer expectation
        out = []
        for i in range(batch_size):
            mask = batch_indices == i
            out.append(output_feat[mask])
            
        return out
