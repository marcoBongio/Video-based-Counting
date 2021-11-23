import timm
import torch.nn as nn
import torch
from einops import rearrange
from torch.nn import functional as F
from torchvision import models
# from timesformer.models.vit import TimeSformer
#import TimesformerCC
from timesformer_pytorch import TimeSformer
from variables import HEIGHT, WIDTH, NUM_FRAMES, PATCH_SIZE_TS, IN_CHANS, EMBED_DIM, DEPTH_TS, \
    NUM_HEADS, DIM_HEAD, HEIGHT_TS, WIDTH_TS, BE_CHANNELS
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
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        #self.timesformer = TimesformerCC.TimeSformer(height=HEIGHT_TS, width=WIDTH_TS, num_frames=NUM_FRAMES,
        #                                            attention_type='divided_space_time')

        self.timesformer = TimeSformer(dim=EMBED_DIM, height=HEIGHT_TS, width=WIDTH_TS, patch_size=PATCH_SIZE_TS,
                                       num_frames=NUM_FRAMES, channels=IN_CHANS, depth=DEPTH_TS, dim_head=DIM_HEAD)


        #self.backend_feat = [512, 512, 512, 256, 128, 64]
        #self.backend = make_layers(self.backend_feat, in_channels=BE_CHANNELS, batch_norm=True, dilation=True)

        self.output_layer = nn.Conv2d(10, 10, kernel_size=1)
        self.relu = nn.ReLU()

        self._initialize_timesformer_weights()
        if not load_weights:
            #mod = models.vgg16(pretrained=True)
            mod = torch.load('fdst.pth.tar')['state_dict']
            self._initialize_weights()
            # address the mismatch in key names for python 3
            #pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            #self.frontend.load_state_dict(pretrained_dict)
            pretrained_dict = {k: v for k, v in mod.items() if k in self.state_dict()}

            pretrained_dict.pop('output_layer.weight')
            pretrained_dict.pop('output_layer.bias')
            self.load_state_dict(pretrained_dict, strict=False)

            for param in self.frontend.parameters():
                param.requires_grad = False
            for param in self.context.parameters():
                param.requires_grad = False


    def forward(self, x, inverse=False):
        #print(x.shape)
        xx = torch.cuda.FloatTensor()

        if inverse:
            for i in reversed(range(x.shape[0])):
                x_prev = self.frontend(x[i])
                x_prev = self.context(x_prev).unsqueeze(0)
                xx = torch.cat((x_prev, xx), 1)
        else:
            for i in range(x.shape[0]):
                x_prev = self.frontend(x[i])
                x_prev = self.context(x_prev).unsqueeze(0)
                xx = torch.cat((x_prev, xx), 1)
        x = xx
        del xx, x_prev
        torch.cuda.empty_cache()

        #x = self.backend(x)
        #x = rearrange(x, 'b c (f1 h) (f2 w) -> b (f1 f2) c h w', f1=NUM_FRAMES//2, f2=NUM_FRAMES//2)

        """ _, _, _, h, w = x.shape
        xx = torch.empty(1, 10, h, w, dtype=torch.float).cuda()
        pad = nn.ZeroPad2d(1)
        x = pad(x)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                xx[:, :, i-1, j-1] = self.timesformer(x[:, :, :, (i-1):(i+2), (j-1):(j+2)])

        x = xx
        del xx
        torch.cuda.empty_cache()"""
        x = self.timesformer(x)
        x = rearrange(x, 'b fl (h w) -> b fl h w', b=self.batch_size, fl=10, h=HEIGHT_TS, w=WIDTH_TS)

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
        # base_model = timm.create_model(
        #    'vit_base_patch16_224',
        #    pretrained=True)

        # self.timesformer.load_state_dict(base_model.state_dict(), strict=False)

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
