import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from timesformer_pytorch import TimeSformer
from variables import NUM_FRAMES, PATCH_SIZE_TS, IN_CHANS, EMBED_DIM, DEPTH_TS, \
    DIM_HEAD, HEIGHT_TS, WIDTH_TS, BE_CHANNELS


class PFTS(nn.Module):
    def __init__(self, load_weights=False, batch_size=1):
        super(PFTS, self).__init__()
        self.batch_size = batch_size

        self.timesformer = TimeSformer(
            dim=EMBED_DIM,
            num_locations=WIDTH_TS * HEIGHT_TS,
            height=HEIGHT_TS,
            width=WIDTH_TS,
            patch_size=PATCH_SIZE_TS,
            num_frames=NUM_FRAMES,
            channels=IN_CHANS,
            depth=DEPTH_TS,
            dim_head=DIM_HEAD,
            attn_dropout=0.1,
            ff_dropout=0.1)

        #self.backend_feat = [512, 512, 512, 256, 128, 64]
        #self.backend = make_layers(self.backend_feat, in_channels=BE_CHANNELS, batch_norm=True, dilation=True)

        self.output_layer = nn.Conv2d(10, 10, kernel_size=1)
        self.relu = nn.ReLU()

        self._initialize_timesformer_weights()

    def forward(self, x, inverse=False):

        if inverse:
            x = torch.flip(x, [1])
        #xx = torch.FloatTensor().cuda()

        #xx = torch.cat((xx, x[:, 0]))
        #xx = torch.cat((xx, x[:, x.shape[1]-1]))
        #xx = xx.unsqueeze(0)

        #x = xx
        x = self.timesformer(x)

        x = rearrange(x, 'b fl (h w) -> b fl h w', b=self.batch_size, fl=10, h=HEIGHT_TS, w=WIDTH_TS)
        #x = rearrange(x, 'b (h w) d -> b d h w', b=self.batch_size, d=BE_CHANNELS, h=HEIGHT_TS, w=WIDTH_TS)
        #x = rearrange(x, 'b f c h w -> (b f) c h w')

        #xx = torch.FloatTensor().cuda()
        #for map in x:
        #    xx = torch.cat((xx, map), 0)
        #x = xx.unsqueeze(0)

        #x = self.backend(x)

        x = self.output_layer(x)
        x = self.relu(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_timesformer_weights(self):
        for layer in self.timesformer.layers:
            prenorm_temporal_attn: nn.Module = layer[0]
            prenorm_temporal_attn.apply(zero)


# initialize module's weights to zero
def zero(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
