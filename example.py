from collections import OrderedDict

import timm
import torch
from torch import nn

from model import zero
from timesformer_pytorch import TimeSformer
from variables import EMBED_DIM, DIM_TS, PATCH_SIZE_TS, NUM_FRAMES, DIM_HEAD, DEPTH_TS, IN_CHANS


def change_key(self, old, new):
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v


def initialize_timesformer_weights():
    base_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True)
    mapping = {
        'cls_token': 'timesformer.cls_token',
        'patch_embed\.(.*)': 'timesformer.patch_embed.\1',
        r'blocks\.(\d+).norm1\.(.*)': r'timesformer.layers.\1.1.norm.\2',
        r'blocks\.(\d+).norm2\.(.*)': r'timesformer.layers.\1.2.norm.\2',
        r'blocks\.(\d+).attn\.qkv\.weight': r'timesformer.layers.\1.1.fn.to_qkv.weight',
        r'blocks\.(\d+).attn\.proj\.(.*)': r'timesformer.layers.\1.1.fn.to_out.0.\2',
        r'blocks\.(\d+).mlp\.fc1\.(.*)': r'timesformer.layers.\1.2.fn.net.0.\2',
        r'blocks\.(\d+).mlp\.fc2\.(.*)': r'timesformer.layers.\1.2.fn.net.3.\2',
    }

    print(timesformer.state_dict())
    print(base_model.state_dict().keys())
    od = base_model.state_dict()
    change_key(od, "patch_embed.proj.weight", "patch_embed.weight")
    change_key(od, "patch_embed.proj.bias", "patch_embed.bias")
    change_key(od, "blocks.0.attn.qkv.weight", "blocks.0.0.fn.to_qkv.weight")
    change_key(od, "blocks.0.attn.qkv.bias", "blocks.0.0.fn.to_qkv.bias")
    change_key(od, "blocks.0.norm1.weight", "blocks.0.0.norm.weight")
    change_key(od, "blocks.0.norm1.bias", "blocks.0.0.norm.bias")
    change_key(od, "blocks.0.mlp.fc1.weight", "blocks.0.0.fn.to_out.0.weight")
    change_key(od, "blocks.0.mlp.fc1.bias", "blocks.0.0.fn.to_out.0.bias")

    print(od.keys())
    timesformer.load_state_dict(base_model.state_dict(), strict=False)
    print(timesformer.state_dict())
    for block in timesformer.blocks:
        prenorm_temporal_attn: nn.Module = block[0]
        prenorm_temporal_attn.apply(zero)


timesformer = TimeSformer(dim=EMBED_DIM, image_size=DIM_TS, patch_size=PATCH_SIZE_TS, num_frames=NUM_FRAMES,
                          channels=IN_CHANS, depth=DEPTH_TS, dim_head=DIM_HEAD)

initialize_timesformer_weights()
