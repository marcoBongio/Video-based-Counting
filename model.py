import timm
import torch.nn as nn
import torch
from einops import rearrange
from torch.nn import functional as F
from torchvision import models
# from timesformer.models.vit import TimeSformer
# import TimeSformerCC
from timesformer_pytorch import TimeSformer
from variables import HEIGHT, WIDTH, NUM_FRAMES, DIM_TS, BE_CHANNELS, PATCH_SIZE_TS, IN_CHANS, EMBED_DIM, DEPTH_TS, \
    NUM_HEADS, DIM_HEAD
from utils import save_net, load_net


class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                        self.scales]
        weights = [self.__make_weight(feats, scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0] * weights[0] + multi_scales[1] * weights[1] + multi_scales[2] * weights[
            2] + multi_scales[3] * weights[3]) / (weights[0] + weights[1] + weights[2] + weights[3])] + [feats]

        bottle = self.bottleneck(torch.cat(overall_features, 1))

        return self.relu(bottle)


class CANNet2s(nn.Module):
    def __init__(self, load_weights=False, batch_size=1):
        super(CANNet2s, self).__init__()
        self.batch_size = batch_size
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]  # , 'M', 1024, 1024, 1024]
        self.frontend = make_layers(self.frontend_feat)

        self.context = ContextualModule(512, 512)  # (1024, 1024)

        # self.timesformer = TimeSformer(img_size=DIM_TS, num_frames=NUM_FRAMES,
        #                                            attention_type='divided_space_time')

        self.timesformer = TimeSformer(dim=EMBED_DIM, image_size=DIM_TS, patch_size=PATCH_SIZE_TS,
                                       num_frames=NUM_FRAMES,
                                       channels=IN_CHANS, depth=DEPTH_TS, dim_head=DIM_HEAD)

        # self.backend_feat = [256, 256, 256, 128, 64]
        # self.backend = make_layers(self.backend_feat, in_channels=BE_CHANNELS, batch_norm=True, dilation=True)

        self.output_layer = nn.Conv2d(10, 10, kernel_size=1)
        self.relu = nn.ReLU()
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self._initialize_timesformer_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x_prev, x):
        #print(x.shape)
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)
        # print(x.shape)
        x_prev = self.context(x_prev)
        x = self.context(x)

        # print(x.shape)
        # x = x[0, None, :]
        x_prev = torch.unsqueeze(x_prev, 1)
        x = torch.unsqueeze(x, 1)

        # print(x.shape)
        x = torch.cat((x_prev, x), 1)

        # print(x.shape)
        x = self.timesformer(x)
        # print(x.shape)
        # x = torch.cat((x_prev, x), 1)
        # print(x.shape)
        # x = self.backend(x)
        # print(x.shape)

        # print(x.shape)
        x = rearrange(x, 'b f (h w) -> b f h w', b=self.batch_size, f=10, h=DIM_TS, w=DIM_TS)
        x = self.output_layer(x)
        x = self.relu(x)
        # print("x_final = " + str(x))
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
        base_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True)

        self.timesformer.load_state_dict(base_model.state_dict(), strict=False)

        for layer in self.timesformer.layers:
            prenorm_temporal_attn: nn.Module = layer[0]
            prenorm_temporal_attn.apply(zero)


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


# initialize module's weights to zero
def zero(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def change_key(self, old, new):
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v
