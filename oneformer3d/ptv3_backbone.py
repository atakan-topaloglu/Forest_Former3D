"""
Point Transformer V3 Backbone for ForestFormer

This module integrates PTV3 into the ForestFormer architecture by wrapping 
the PTV3 model and adapting its interface to match ForestFormer's expectations.

Design Choices:
1. Input: 6 channels (centered_xyz + zero padding) for pretrained weight compatibility
2. Output: 64 channels (dec_channels[0]) - decoder will be configured accordingly
3. Voxelization: Uses grid_size parameter, handled by PTV3's serialization
4. PDNorm: Configurable, with "ForAINet" as default condition
5. Serialization: Uses default z-order and hilbert curves

Author: Adapted for ForestFormer integration
Based on: Point Transformer V3 by Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
"""

import sys
from functools import partial
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from addict import Dict

from mmdet3d.registry import MODELS

try:
    import flash_attn
except ImportError:
    flash_attn = None


# =============================================================================
# Serialization Functions (from PTV3)
# =============================================================================

class KeyLUT:
    """Look-up table for z-order encoding/decoding."""
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self._xyz2key(r256, zero, zero, 8),
                self._xyz2key(zero, r256, zero, 8),
                self._xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self._key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def _xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def _key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def z_order_encode(x, y, z, b=None, depth=16):
    """Encode xyz coordinates to z-order (Morton) code."""
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key


def _right_shift(binary, k=1, axis=-1):
    """Right shift for hilbert encoding."""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )
    return shifted


def _gray2binary(gray, axis=-1):
    """Convert Gray code to binary."""
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def hilbert_encode(locs, num_dims=3, num_bits=16):
    """Encode locations to Hilbert curve index."""
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = torch.logical_xor(
                gray[:, 0, bit + 1:], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
            )
            gray[:, dim, bit + 1:] = torch.logical_xor(
                gray[:, dim, bit + 1:], to_flip
            )
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()
    return hh_uint64


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """Encode grid coordinates to serialization code."""
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
        code = z_order_encode(x, y, z, b=None, depth=depth)
    elif order == "z-trans":
        x, y, z = grid_coord[:, 1].long(), grid_coord[:, 0].long(), grid_coord[:, 2].long()
        code = z_order_encode(x, y, z, b=None, depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, num_dims=3, num_bits=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], num_dims=3, num_bits=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


# =============================================================================
# Utility Functions
# =============================================================================

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


# =============================================================================
# Point Data Structure
# =============================================================================

class Point(Dict):
    """
    Point Structure for PTV3.
    
    A Point (point cloud) is a dictionary containing various properties:
    - "coord": original coordinates
    - "grid_coord": discrete coordinates after grid sampling
    - "feat": point features
    - "offset" or "batch": batch information
    - "serialized_code/order/inverse": serialization info
    - "sparse_conv_feat": SparseConvTensor for SpConv
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """Point Cloud Serialization using space-filling curves."""
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = [encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order]
        code = torch.stack(code)
        order_indices = torch.argsort(code)
        inverse = torch.zeros_like(order_indices).scatter_(
            dim=1,
            index=order_indices,
            src=torch.arange(0, code.shape[1], device=order_indices.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order_indices = order_indices[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_indices
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """Create SparseConvTensor for SpConv operations."""
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


# =============================================================================
# PTV3 Module Components
# =============================================================================

class PointModule(nn.Module):
    """Base module for Point-based operations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    """Sequential container for Point modules."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    """Per-Dataset Normalization for multi-dataset training."""
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ForAINet",),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer(num_features)
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(nn.Module):
    """Relative Position Encoding."""
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)
        return out


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    Implements: y = W x + b + (alpha/r) * B(A(dropout(x)))
    where A: in->r, B: r->out.

    Notes:
    - By default, base weights are frozen (train_base=False) and only LoRA params train.
    - LoRA params are initialized with B=0 so the initial behavior matches the base layer.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        train_base: bool = False,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        assert r >= 0

        self.base = base
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 0.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if not train_base:
            for p in self.base.parameters():
                p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Linear(self.base.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return out


class SerializedAttention(PointModule):
    """Serialized Attention with optional Flash Attention."""
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        lora_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_train_base: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        
        if enable_flash:
            assert enable_rpe is False
            assert upcast_attention is False
            assert upcast_softmax is False
            assert flash_attn is not None, "flash_attn not installed"
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)

        qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        proj = nn.Linear(channels, channels)
        if lora_enabled and lora_r > 0:
            self.qkv = LoRALinear(
                qkv,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                train_base=lora_train_base,
            )
            self.proj = LoRALinear(
                proj,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                train_base=lora_train_base,
            )
        else:
            self.qkv = qkv
            self.proj = proj
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
        lora_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_train_base: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

        if lora_enabled and lora_r > 0:
            self.fc1 = LoRALinear(
                fc1,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                train_base=lora_train_base,
            )
            self.fc2 = LoRALinear(
                fc2,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                train_base=lora_train_base,
            )
        else:
            self.fc1 = fc1
            self.fc2 = fc2

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    """Transformer Block with serialized attention."""
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        lora_attention: bool = False,
        lora_mlp: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_train_base: bool = False,
        checkpoint_enabled: bool = False,
        checkpoint_use_reentrant: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.checkpoint_enabled = bool(checkpoint_enabled)
        self.checkpoint_use_reentrant = bool(checkpoint_use_reentrant)

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            lora_enabled=lora_attention,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_train_base=lora_train_base,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
                lora_enabled=lora_mlp,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_train_base=lora_train_base,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        if self.checkpoint_enabled and self.training:
            import torch.utils.checkpoint as checkpoint

            def attn_forward(feat, point_ref):
                point_ref.feat = feat
                point_ref = self.attn(point_ref)
                return point_ref.feat

            new_feat = checkpoint.checkpoint(
                attn_forward,
                point.feat,
                point,
                use_reentrant=self.checkpoint_use_reentrant,
            )
            point.feat = new_feat
            point = self.drop_path(point)
        else:
            point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        if self.checkpoint_enabled and self.training:
            import torch.utils.checkpoint as checkpoint

            def mlp_forward(feat, point_ref):
                point_ref.feat = feat
                point_ref = self.mlp(point_ref)
                return point_ref.feat

            new_feat = checkpoint.checkpoint(
                mlp_forward,
                point.feat,
                point,
                use_reentrant=self.checkpoint_use_reentrant,
            )
            point.feat = new_feat
            point = self.drop_path(point)
        else:
            point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    """Pooling layer using serialization."""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        else:
            self.norm = None
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        else:
            self.act = None

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    """Unpooling layer for decoder."""
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    """Input embedding layer."""
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


# =============================================================================
# Main PTV3 Backbone
# =============================================================================

class PointTransformerV3(PointModule):
    """Point Transformer V3 backbone."""
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ForAINet",),
        # LoRA (optional, parameter-efficient fine-tuning)
        lora_enabled: bool = False,
        lora_attention: bool = True,
        lora_mlp: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_train_base: bool = False,
        checkpoint_enabled: bool = False,
        checkpoint_use_reentrant: bool = True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        self.lora_enabled = bool(lora_enabled and (lora_r > 0))
        self.lora_attention = bool(lora_attention)
        self.lora_mlp = bool(lora_mlp)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_train_base = bool(lora_train_base)
        self.checkpoint_enabled = bool(checkpoint_enabled)
        self.checkpoint_use_reentrant = bool(checkpoint_use_reentrant)

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # Norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # Encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        lora_attention=(self.lora_enabled and self.lora_attention),
                        lora_mlp=(self.lora_enabled and self.lora_mlp),
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        lora_train_base=self.lora_train_base,
                        checkpoint_enabled=self.checkpoint_enabled,
                        checkpoint_use_reentrant=self.checkpoint_use_reentrant,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # Decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            lora_attention=(self.lora_enabled and self.lora_attention),
                            lora_mlp=(self.lora_enabled and self.lora_mlp),
                            lora_r=self.lora_r,
                            lora_alpha=self.lora_alpha,
                            lora_dropout=self.lora_dropout,
                            lora_train_base=self.lora_train_base,
                            checkpoint_enabled=self.checkpoint_enabled,
                            checkpoint_use_reentrant=self.checkpoint_use_reentrant,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def set_checkpointing(self, enabled: bool, use_reentrant: bool = True):
        """Enable/disable gradient checkpointing for attention/MLP blocks."""
        self.checkpoint_enabled = bool(enabled)
        self.checkpoint_use_reentrant = bool(use_reentrant)
        for module in self.modules():
            if isinstance(module, Block):
                module.checkpoint_enabled = self.checkpoint_enabled
                module.checkpoint_use_reentrant = self.checkpoint_use_reentrant

    def forward(self, data_dict):
        """Forward pass.
        
        Args:
            data_dict: Dictionary containing:
                - feat: point features
                - grid_coord or (coord + grid_size)
                - offset or batch
        
        Returns:
            Point: Output point cloud with features
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        return point


# =============================================================================
# ForestFormer Backbone Wrapper
# =============================================================================

@MODELS.register_module()
class PTV3Backbone(nn.Module):
    """
    PTV3 Backbone wrapper for ForestFormer.
    
    This wrapper adapts PTV3 to work with ForestFormer's data flow:
    1. Takes input from ForestFormer's collate function
    2. Converts to PTV3's Point format with proper feature handling
    3. Runs PTV3 encoder-decoder
    4. Returns features in ForestFormer's expected format
    
    Args:
        in_channels (int): Input feature channels (before padding). Default: 3
        grid_size (float): Voxel size for grid sampling. Default: 0.02
        out_channels (int): Output feature channels. Default: 64
        pretrained (str): Path to pretrained weights. Default: None
        input_mode (str): How to handle input channel mismatch. Options:
            - "zero_pad": Zero-pad input to 6 channels (default, backward compatible)
            - "adapter": Use learnable adapter layer to project 3D -> 6D
            - "reinit_embedding": Re-initialize embedding layer to accept 3 channels
        freeze_backbone (bool): Freeze encoder/decoder weights (not embedding). Default: False
        freeze_embedding (bool): Freeze embedding layer weights. Default: False
        order (tuple): Serialization orders. Default: ("z", "z-trans", "hilbert", "hilbert-trans")
        stride (tuple): Downsampling strides. Default: (2, 2, 2, 2)
        enc_depths (tuple): Encoder block depths. Default: (2, 2, 2, 6, 2)
        enc_channels (tuple): Encoder channel dims. Default: (32, 64, 128, 256, 512)
        enc_num_head (tuple): Encoder attention heads. Default: (2, 4, 8, 16, 32)
        enc_patch_size (tuple): Encoder patch sizes. Default: (1024, 1024, 1024, 1024, 1024)
        dec_depths (tuple): Decoder block depths. Default: (2, 2, 2, 2)
        dec_channels (tuple): Decoder channel dims. Default: (64, 64, 128, 256)
        dec_num_head (tuple): Decoder attention heads. Default: (4, 4, 8, 16)
        dec_patch_size (tuple): Decoder patch sizes. Default: (1024, 1024, 1024, 1024)
        pdnorm_bn (bool): Use PDNorm for BatchNorm. Default: False
        pdnorm_ln (bool): Use PDNorm for LayerNorm. Default: False
        pdnorm_conditions (tuple): Conditions for PDNorm. Default: ("ForAINet",)
        enable_flash (bool): Enable Flash Attention. Default: True
        **kwargs: Additional arguments for PTV3
    """
    
    def __init__(
        self,
        in_channels=3,
        grid_size=0.02,
        out_channels=64,
        pretrained=None,
        input_mode="zero_pad",  # "zero_pad", "adapter", or "reinit_embedding"
        freeze_backbone=False,
        freeze_embedding=False,
        use_gradient_checkpointing=False,  # Enable gradient checkpointing for memory savings
        # LoRA (optional, parameter-efficient fine-tuning)
        lora_enabled: bool = False,
        lora_attention: bool = True,
        lora_mlp: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_train_base: bool = False,
        keep_lora_trainable: bool = True,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ForAINet",),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.out_channels = out_channels
        self.input_mode = input_mode
        self.freeze_backbone = freeze_backbone
        self.freeze_embedding = freeze_embedding
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_flash = enable_flash  # Store for checkpointing compatibility check
        self.lora_enabled = bool(lora_enabled and (lora_r > 0))
        self.keep_lora_trainable = bool(keep_lora_trainable)
        
        if self.use_gradient_checkpointing:
            print(f"[PTV3] Gradient checkpointing ENABLED for memory savings")
        
        assert input_mode in ["zero_pad", "adapter", "reinit_embedding"], \
            f"input_mode must be 'zero_pad', 'adapter', or 'reinit_embedding', got {input_mode}"
        
        # Determine PTV3 input channels based on mode
        if input_mode == "reinit_embedding":
            # Build PTV3 with actual input channels (embedding will be re-initialized)
            self.ptv3_in_channels = in_channels
        else:
            # For zero_pad and adapter modes, PTV3 expects 6 channels
            self.ptv3_in_channels = 6
        
        # Store PDNorm settings for condition handling
        self.use_pdnorm = pdnorm_bn or pdnorm_ln
        self.pdnorm_condition = pdnorm_conditions[0] if pdnorm_conditions else "ForAINet"
        
        # Build input adapter if using adapter mode
        if input_mode == "adapter":
            self.input_adapter = nn.Sequential(
                nn.Linear(in_channels, 6),
                nn.LayerNorm(6),
            )
            print(f"[PTV3] Created input adapter: {in_channels}D -> 6D")
        
        # Build PTV3 backbone
        self.ptv3 = PointTransformerV3(
            in_channels=self.ptv3_in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=False,
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions,
            lora_enabled=self.lora_enabled,
            lora_attention=lora_attention,
            lora_mlp=lora_mlp,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_train_base=lora_train_base,
        )
        
        # Output layer to ensure proper output format
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(dec_channels[0], eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        # Verify output channels match
        assert dec_channels[0] == out_channels, \
            f"dec_channels[0]={dec_channels[0]} must match out_channels={out_channels}"
        
        # Load pretrained weights if provided
        if pretrained is not None:
            self.load_pretrained(pretrained)
        
        # Apply freezing after loading pretrained weights
        self._apply_freezing()
    
    def _apply_freezing(self):
        """Apply freezing to backbone and/or embedding based on config."""
        if self.freeze_backbone:
            frozen_count = 0
            # Freeze encoder and decoder, but not embedding
            for name, param in self.ptv3.named_parameters():
                if name.startswith('enc.') or name.startswith('dec.'):
                    if self.keep_lora_trainable and ('lora_A' in name or 'lora_B' in name):
                        continue
                    param.requires_grad = False
                    frozen_count += 1
            print(f"[PTV3] Froze {frozen_count} encoder/decoder parameters")
        
        if self.freeze_embedding:
            frozen_count = 0
            for name, param in self.ptv3.named_parameters():
                if name.startswith('embedding.'):
                    if self.keep_lora_trainable and ('lora_A' in name or 'lora_B' in name):
                        continue
                    param.requires_grad = False
                    frozen_count += 1
            print(f"[PTV3] Froze {frozen_count} embedding parameters")
        
        # Print trainable parameter summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[PTV3] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def load_pretrained(self, pretrained_path):
        """
        Load pretrained weights for the PTV3 backbone.
        
        Supports loading from:
        1. Pointcept checkpoint format (with 'state_dict' key and 'module.backbone.' prefix)
        2. Direct state dict format
        
        Handles SpConv weight format differences between versions by permuting weights.
        For 'reinit_embedding' mode, skips loading embedding weights.
        
        Args:
            pretrained_path (str): Path to pretrained checkpoint file.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        print(f"[PTV3] Loading pretrained weights from {pretrained_path}")
        print(f"[PTV3] Input mode: {self.input_mode}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter and rename keys for PTV3 backbone
        # Checkpoint format: module.backbone.xxx -> xxx
        ptv3_state_dict = {}
        prefixes_to_strip = ['module.backbone.', 'backbone.', 'module.']
        skip_prefixes = ['seg_head.', 'criteria.', 'head.', 'module.seg_head.']
        
        for key, value in state_dict.items():
            # Skip non-backbone keys
            if any(key.startswith(prefix) or key.replace('module.', '').startswith(prefix) 
                   for prefix in skip_prefixes):
                continue
            
            new_key = key
            # Strip prefixes in order of specificity
            for prefix in prefixes_to_strip:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            
            ptv3_state_dict[new_key] = value
        
        print(f"[PTV3] Extracted {len(ptv3_state_dict)} backbone keys from checkpoint")
        
        # Get model state dict
        model_state_dict = self.ptv3.state_dict()
        print(f"[PTV3] Model has {len(model_state_dict)} keys")
        
        # Manually load weights with shape fixing
        loaded_count = 0
        skipped_count = 0
        permuted_count = 0
        reinit_count = 0
        
        with torch.no_grad():
            for name, param in self.ptv3.named_parameters():
                # For reinit_embedding mode, skip embedding layer weights
                if self.input_mode == "reinit_embedding" and name.startswith('embedding.'):
                    print(f"[PTV3] REINIT (not loading): {name}")
                    reinit_count += 1
                    continue
                
                if name in ptv3_state_dict:
                    ckpt_weight = ptv3_state_dict[name]
                    
                    if ckpt_weight.shape == param.shape:
                        # Direct copy
                        param.copy_(ckpt_weight)
                        loaded_count += 1
                    elif len(ckpt_weight.shape) == 5 and len(param.shape) == 5:
                        # Try permutations for 5D conv weights
                        permutations = [
                            (0, 2, 3, 4, 1),  # [out, in, d, h, w] -> [out, d, h, w, in]
                            (0, 4, 1, 2, 3),  # reverse
                        ]
                        matched = False
                        for perm in permutations:
                            permuted = ckpt_weight.permute(*perm).contiguous()
                            if permuted.shape == param.shape:
                                param.copy_(permuted)
                                loaded_count += 1
                                permuted_count += 1
                                matched = True
                                print(f"[PTV3] Permuted {name}: {ckpt_weight.shape} -> {permuted.shape}")
                                break
                        if not matched:
                            print(f"[PTV3] SKIP {name}: ckpt {ckpt_weight.shape} vs model {param.shape}")
                            skipped_count += 1
                    else:
                        print(f"[PTV3] SKIP {name}: shape mismatch ckpt {ckpt_weight.shape} vs model {param.shape}")
                        skipped_count += 1
                else:
                    skipped_count += 1
            
            # Also load buffers (running_mean, running_var, etc.)
            for name, buffer in self.ptv3.named_buffers():
                # Skip embedding buffers for reinit_embedding mode
                if self.input_mode == "reinit_embedding" and name.startswith('embedding.'):
                    reinit_count += 1
                    continue
                    
                if name in ptv3_state_dict:
                    ckpt_buffer = ptv3_state_dict[name]
                    if ckpt_buffer.shape == buffer.shape:
                        buffer.copy_(ckpt_buffer)
                        loaded_count += 1
        
        print(f"[PTV3] Loaded {loaded_count} weights, permuted {permuted_count}, "
              f"skipped {skipped_count}, re-initialized {reinit_count}")

    def forward(self, x: spconv.SparseConvTensor):
        """
        Forward pass.
        
        Args:
            x (spconv.SparseConvTensor): Input sparse tensor from ForestFormer's collate.
                Contains features and coordinates.
        
        Returns:
            List[Tensor]: List of feature tensors, one per batch item.
                Each tensor has shape (n_points_i, out_channels).
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
            # Use learnable adapter to project to 6 channels
            features = self.input_adapter(features)
        elif self.input_mode == "zero_pad":
            # Zero-pad features to 6 channels if needed
            if features.shape[1] < self.ptv3_in_channels:
                padding = torch.zeros(
                    features.shape[0], 
                    self.ptv3_in_channels - features.shape[1],
                    device=features.device,
                    dtype=features.dtype
                )
                features = torch.cat([features, padding], dim=1)
        # For "reinit_embedding" mode, features are passed as-is (3 channels)
        
        # Compute offset from batch indices
        # offset[i] = cumulative count of points up to batch i
        batch_counts = torch.bincount(batch_indices, minlength=batch_size)
        offset = torch.cumsum(batch_counts, dim=0).long()
        
        # Create Point data dict for PTV3
        data_dict = {
            "feat": features,
            "coord": coords,  # Using grid coordinates as coord
            "grid_coord": coords.int(),  # Grid coordinates for serialization
            "batch": batch_indices,
            "offset": offset,
        }
        
        # Add condition for PDNorm if enabled
        if self.use_pdnorm:
            data_dict["condition"] = self.pdnorm_condition
        
        # Enable checkpointing inside attention/MLP blocks (safe for Point structure)
        if self.use_gradient_checkpointing and self.training:
            # Flash attention is incompatible with non-reentrant checkpointing
            use_reentrant = self.enable_flash
            self.ptv3.set_checkpointing(True, use_reentrant=use_reentrant)
        else:
            self.ptv3.set_checkpointing(False, use_reentrant=False)

        # Normal forward pass (checkpointing handled inside blocks if enabled)
        point = self.ptv3(data_dict)
        
        # Apply output normalization
        output_feat = self.output_layer(point.feat)
        
        # Split features by batch
        out = []
        for i in range(batch_size):
            mask = batch_indices == i
            out.append(output_feat[mask])
        
        return out
