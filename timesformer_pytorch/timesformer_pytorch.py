import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import register_model
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from TimeSformer.timesformer.models.vit import _cfg
from timesformer_pytorch.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding

# helpers
from variables import DIM_TS


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224'
            '-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


def exists(val):
    return val is not None


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# time token shift

def shift(t, amt):
    if amt is 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))


class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        flows_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f=f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim=-1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim=-1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((flows_x, x), dim=1)
        return self.fn(x, *args, **kwargs)


# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


# attention

def attn(q, k, v, mask=None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask=None, flows_mask=None, rot_emb=None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # splice out flows tokens at index 10
        (flows_q, q_), (flows_k, k_), (flows_v, v_) = map(lambda t: (t[:, :10], t[:, 10:]), (q, k, v))

        # let flows tokens attend to key / values of all patches across time and space
        flows_out = attn(flows_q, k, v, mask=flows_mask)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # expand flows token keys and values across time or space and concat
        # print(q_.shape)
        # print(flows_k.shape)
        r = q_.shape[0] // flows_k.shape[0]
        # print(r)
        flows_k, flows_v = map(lambda t: repeat(t, 'b (f) d -> (b r) (f) d', r=r), (flows_k, flows_v))

        k_ = torch.cat((flows_k, k_), dim=1)
        v_ = torch.cat((flows_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_, mask=mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the flows token
        out = torch.cat((flows_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print("OUT = " + str(out.shape))
        # combine heads out
        return self.to_out(out)


# main classes

class TimeSformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_frames,
            # num_classes,
            image_size=224,
            patch_size=16,
            channels=3,
            depth=12,
            heads=8,
            dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.,
            rotary_emb=True,
            shift_tokens=False
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        # Compute N = HW/P^2
        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        # Compute D = C * P^2
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.flows_tokens = nn.Parameter(torch.randn(10, dim))
        # print(self.flows_tokens.shape)

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 10, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            time_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            spatial_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, DIM_TS ** 2)
        )

    def forward(self, video, mask=None):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        # print(*video.shape)

        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        n = hp * wp

        # video to patch embeddings

        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        tokens = self.patch_embed(video)

        # add flows tokens

        flows_tokens = repeat(self.flows_tokens, 'n d -> b n d', b=b)
        x = torch.cat((flows_tokens, tokens), dim=1)

        # positional embedding

        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device=device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device=device)
            image_pos_emb = self.image_rot_emb(hp, wp, device=device)

        # calculate masking for uneven number of frames

        frame_mask = None
        flows_attn_mask = None
        if exists(mask):
            mask_with_flows = F.pad(mask, (1, 0), value=True)

            frame_mask = repeat(mask_with_flows, 'b f -> (b h n) () f', n=n, h=self.heads)

            flows_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n=n, h=self.heads)
            flows_attn_mask = F.pad(flows_attn_mask, (1, 0), value=True)

        # time and space attention

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n, mask=frame_mask, flows_mask=flows_attn_mask,
                          rot_emb=frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f, flows_mask=flows_attn_mask, rot_emb=image_pos_emb) + x
            x = ff(x) + x

        flows_tokens = x[:, :10]

        out = self.to_out(flows_tokens)

        # out = rearrange(out, 'b f (h w) -> b f h w', b=b, f=10, h=DIM_TS, w=DIM_TS)
        # print(out.shape)
        return out


