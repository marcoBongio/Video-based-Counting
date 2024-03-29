from einops import rearrange
from fastai.torch_imports import *
from torchvision import models

from timesformer_pytorch import TimeSformer
from variables import BE_CHANNELS, EMBED_DIM, WIDTH_TS, HEIGHT_TS, PATCH_SIZE_TS, \
    NUM_FRAMES, IN_CHANS, DEPTH_TS, NUM_HEADS, DIM_HEAD, MODE


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


class FeatureFusionModel(nn.Module):
    def __init__(self, mode, img_feat_dim, txt_feat_dim, common_space_dim):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            pass
        elif mode == 'weighted':
            self.alphas = nn.Sequential(
                nn.Linear(img_feat_dim + txt_feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 2))
            self.img_proj = nn.Linear(img_feat_dim, common_space_dim)
            self.txt_proj = nn.Linear(txt_feat_dim, common_space_dim)
            self.post_process = nn.Sequential(
                nn.Linear(common_space_dim, common_space_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(common_space_dim, common_space_dim)
            )

    def forward(self, img_feat, txt_feat):
        if self.mode == 'concat':
            out_feat = torch.cat((img_feat, txt_feat), 1)
            return out_feat
        elif self.mode == 'weighted':
            b, c, h, w = img_feat.shape
            img_feat = rearrange(img_feat, 'b c h w -> (b h w) c')
            txt_feat = rearrange(txt_feat, 'b c h w -> (b h w) c')
            concat_feat = torch.cat((img_feat, txt_feat), dim=1)
            alphas = torch.sigmoid(self.alphas(concat_feat))  # B x 2
            img_feat_norm = F.normalize(self.img_proj(img_feat), p=2, dim=1)
            txt_feat_norm = F.normalize(self.txt_proj(txt_feat), p=2, dim=1)
            out_feat = img_feat_norm * alphas[:, 0].unsqueeze(1) + txt_feat_norm * alphas[:, 1].unsqueeze(1)
            out_feat = self.post_process(out_feat)
            out_feat = rearrange(out_feat, '(b h w) c -> b c h w', b=b, h=h, w=w)

            return out_feat


class BMM(nn.Module):
    def __init__(self):
        super(BMM, self).__init__()

    def forward(self, q, k):
        return torch.bmm(q, k)


class SelfAttention(nn.Module):
    " Self attention Layer"

    def __init__(self, in_dim, activation='relu'):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bmm = BMM()
        self.softmax = nn.Softmax(dim=1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = self.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out


class CANNet2s(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet2s, self).__init__()
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=BE_CHANNELS, batch_norm=True, dilation=True)
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        self.relu = nn.ReLU()

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x_prev, x):
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)

        x_prev = self.context(x_prev)
        x = self.context(x)

        x = torch.cat((x_prev, x), 1)

        x = self.backend(x)

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


class TSCANNet2s(nn.Module):
    def __init__(self, load_weights=False, batch_size=1):
        super(TSCANNet2s, self).__init__()
        self.batch_size = batch_size
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        self.context = ContextualModule(512, 512)

        self.timesformer = TimeSformer(
            dim=EMBED_DIM,
            num_locations=WIDTH_TS * HEIGHT_TS,
            height=HEIGHT_TS,
            width=WIDTH_TS,
            patch_size=PATCH_SIZE_TS,
            num_frames=NUM_FRAMES,
            channels=IN_CHANS,
            depth=DEPTH_TS,
            heads=NUM_HEADS,
            dim_head=DIM_HEAD,
            attn_dropout=0.1,
            ff_dropout=0.1,
            rotary_emb=True)

        self.output_layer = nn.Conv2d(10, 10, kernel_size=1)
        self.relu = nn.ReLU()

        self._initialize_timesformer_weights()
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x, inverse=False):

        xx = torch.FloatTensor().cuda()
        x = rearrange(x, 'f b c h w -> b f c h w')
        if inverse:
            x = torch.flip(x, [1])

        for i in range(x.shape[1]):
            x_prev = self.frontend(x[:, i])
            x_prev = self.context(x_prev)

            xx = torch.cat((xx, x_prev), 0)

        x = xx.unsqueeze(0)
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
        for layer in self.timesformer.layers:
            prenorm_temporal_attn: nn.Module = layer[0]
            prenorm_temporal_attn.apply(zero)


class FETSCANNet2s(nn.Module):
    def __init__(self, load_weights=False, batch_size=1):
        super(FETSCANNet2s, self).__init__()
        self.batch_size = batch_size
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        self.context = ContextualModule(512, 512)

        self.timesformer = TimeSformer(
            dim=EMBED_DIM,
            num_locations=WIDTH_TS * HEIGHT_TS,
            height=HEIGHT_TS,
            width=WIDTH_TS,
            patch_size=PATCH_SIZE_TS,
            num_frames=NUM_FRAMES,
            channels=IN_CHANS,
            depth=DEPTH_TS,
            heads=NUM_HEADS,
            dim_head=DIM_HEAD,
            attn_dropout=0.1,
            ff_dropout=0.1,
            rotary_emb=True)

        self.output_layer = nn.Conv2d(10, 10, kernel_size=1)
        self.relu = nn.ReLU()

        self._initialize_timesformer_weights()
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x, inverse=False):
        if inverse:
            x = torch.flip(x, [1])
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
        for layer in self.timesformer.layers:
            prenorm_temporal_attn: nn.Module = layer[0]
            prenorm_temporal_attn.apply(zero)


class ZTTSCANNet2s(nn.Module):
    def __init__(self, load_weights=False, batch_size=1):
        super(ZTTSCANNet2s, self).__init__()
        self.batch_size = batch_size
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        self.context = ContextualModule(512, 512)

        self.timesformer = TimeSformer(
            dim=EMBED_DIM,
            num_locations=WIDTH_TS * HEIGHT_TS,
            height=HEIGHT_TS,
            width=WIDTH_TS,
            patch_size=PATCH_SIZE_TS,
            num_frames=NUM_FRAMES,
            channels=IN_CHANS,
            depth=DEPTH_TS,
            heads=NUM_HEADS,
            dim_head=DIM_HEAD,
            attn_dropout=0.1,
            ff_dropout=0.1,
            rotary_emb=True)

        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=EMBED_DIM, batch_norm=True, dilation=True)

        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        self.relu = nn.ReLU()

        self._initialize_timesformer_weights()
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self, x, inverse=False):

        xx = torch.FloatTensor().cuda()
        x = rearrange(x, 'f b c h w -> b f c h w')
        if inverse:
            x = torch.flip(x, [1])

        for i in range(x.shape[1]):
            x_prev = self.frontend(x[:, i])
            x_prev = self.context(x_prev)

            xx = torch.cat((xx, x_prev), 0)

        fin_h, fin_w = xx.shape[2:]
        x = xx.unsqueeze(0)

        x = self.timesformer(x)
        x = rearrange(x, 'b (h w) d -> b d h w', b=self.batch_size, h=fin_h // PATCH_SIZE_TS, w=fin_w // PATCH_SIZE_TS)

        x = self.backend(x)

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


class SACANNet2s(nn.Module):
    def __init__(self, load_weights=False, fine_tuning=False):
        super(SACANNet2s, self).__init__()
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=BE_CHANNELS, batch_norm=True, dilation=True)
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        self.relu = nn.ReLU()

        self.FMM = FeatureFusionModel(mode=MODE, img_feat_dim=512, txt_feat_dim=512, common_space_dim=512)

        self.sacnn = SelfAttention(BE_CHANNELS)
        self.sacnn2 = SelfAttention(BE_CHANNELS)
        self.sacnn3 = SelfAttention(64)
        self.sacnn4 = SelfAttention(64)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

        if fine_tuning:
            model_best_JTA = torch.load('JTA/models/model_best_4sacnn_concat_mse_JTA.pth.tar', map_location='cpu')
            self.load_state_dict(model_best_JTA['state_dict'])

    def forward(self, x_prev, x):
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)

        x_prev = self.context(x_prev)
        x = self.context(x)

        x = self.FMM(x_prev, x)
        x = self.sacnn(x)
        x = self.sacnn2(x)

        x = self.backend(x)

        x = self.sacnn3(x)
        x = self.sacnn4(x)

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
