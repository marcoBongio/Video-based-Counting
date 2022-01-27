import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
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


class FeatureExtractor(nn.Module):
    def __init__(self, load_weights=False):
        super(FeatureExtractor, self).__init__()
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        if not load_weights:
            mod = torch.load('fdst.pth.tar')['state_dict']
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k: v for k, v in mod.items() if k in self.state_dict()}
            self.load_state_dict(pretrained_dict, strict=True)

            for param in self.frontend.parameters():
                param.requires_grad = False
            for param in self.context.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.frontend(x)
        x = self.context(x)

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