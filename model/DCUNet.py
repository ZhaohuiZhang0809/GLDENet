import cv2
import pywt
import torch.nn as nn
import torch
from einops import rearrange
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
from timm.models.layers import DropPath

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# 2D DWT滤波
class Dwt2d(nn.Module):
    def __init__(self):
        super(Dwt2d, self).__init__()
        self.requires_grad = False

    def dwt(self, x):
        with torch.no_grad():
            x = x.cpu()    ##
            # LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
            LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')

            LL = torch.tensor(LL).cuda()
            LH = torch.tensor(LH).cuda()
            HL = torch.tensor(HL).cuda()
            HH = torch.tensor(HH).cuda()

        return torch.cat((LL, LH, HL, HH), 1)

    def forward(self, x):
        out = self.dwt(x)
        return out




# 2D IWT滤波
class Iwt2d(nn.Module):
    def __init__(self):
        super(Iwt2d, self).__init__()
        self.requires_grad = False

    def iwt(self, x):
        with torch.no_grad():
            in_batch, in_channel, in_height, in_width = x.size()
            ch = in_channel // 4
            LL = x[:, 0: ch, :, :]
            LH = x[:, ch: ch * 2, :, :]
            HL = x[:, ch * 2: ch * 3, :, :]
            HH = x[:, ch * 3: ch * 4, :, :]

            coeffs = LL.cpu(), (LH.cpu(), HL.cpu(), HH.cpu())

            x = pywt.idwt2(coeffs, 'haar')
            # x = pywt.idwt2(coeffs, 'db2')
            x = torch.tensor(x).cuda()

        return x

    def forward(self, x):
        out = self.iwt(x)
        return out


class Dwtpool(nn.Module):
    def __init__(self, dim, ratio=4, scale=8):
        super(Dwtpool, self).__init__()
        self.ratio = ratio

        self.pool = nn.MaxPool2d(2)
        # self.pool = nn.AdaptiveAvgPool2d((size // 2, size // 2))
        self.dwt = Dwt2d()

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//ratio, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//ratio),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(dim, 1, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=False)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=False)
        # self.conv5 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, stride=1, bias=False)

        self.channel_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(dim, dim // scale, kernel_size=1),
            nn.LayerNorm([dim // scale, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // scale, dim, kernel_size=1)
        )

        self.conv1x1 = nn.Conv2d(dim * 6, dim, 3, padding=1, stride=1,bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
            p = self.pool(x)

            B, C, H, W = list(x.size())
            qkv0 = self.dwt(self.reduce(x))

            # 多尺度感受野
            qkv1 = self.conv1(qkv0)
            qkv2 = self.conv2(qkv0)
            qkv3 = self.conv3(qkv0)
            qkv4 = self.conv4(qkv0)

            qkv = self.conv1x1(torch.cat([qkv0, qkv1, qkv2, qkv3, qkv4, p], dim=1))

            content_feature = self.conv(qkv).view(B, -1, W//2 * H//2).permute(0, 2, 1)  # HW x 1
            content_feature = self.softmax(content_feature)

            channel_feature = self.channel_conv(qkv).view(B, -1, W//2 * H//2)  # C x HW
            channel_pooling = torch.bmm(channel_feature, content_feature).view(B, -1, 1, 1)  # C x 1 x 1
            channel_weight = self.channel_trans_conv(channel_pooling)
            att_cha = qkv * channel_weight

            x = self.proj(att_cha)
            # v = self.voting_gate(att_cha)
            # x = self.proj(att_cha + p)
            # x = self.proj(torch.cat([att_cha, p], dim=1))
            # x = v * qkv

            return x


class IwtUp(nn.Module):
    def __init__(self, dim, ratio=2, scale=8):
        super(IwtUp, self).__init__()
        self.ratio = ratio

        self.up = nn.ConvTranspose2d(dim, dim//2, 2, stride=2)
        self.Iwt = Iwt2d()

        self.add = nn.Sequential(
            nn.Conv2d(dim, dim*ratio, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim*ratio),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(dim//2, 1, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(dim//2, dim//2, kernel_size=5, padding=2, stride=1, bias=False)
        self.conv4 = nn.Conv2d(dim//2, dim//2, kernel_size=7, padding=3, stride=1, bias=False)
        # self.conv5 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, stride=1, bias=False)

        self.channel_conv = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(dim//2, dim // scale, kernel_size=1),
            nn.LayerNorm([dim // scale, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // scale, dim//2, kernel_size=1)
        )

        self.conv1x1 = nn.Conv2d(dim * 3, dim//2, 3, padding=1, stride=1,bias=False)
        self.proj = nn.Conv2d(dim//2, dim // 2, 1, bias=False)

    def forward(self, x):
            up = self.up(x)

            B, C, H, W = list(x.size())
            qkv0 = self.Iwt(self.add(x))

            # 多尺度感受野
            qkv1 = self.conv1(qkv0)
            qkv2 = self.conv2(qkv0)
            qkv3 = self.conv3(qkv0)
            qkv4 = self.conv4(qkv0)

            qkv = self.conv1x1(torch.cat([qkv0, qkv1, qkv2, qkv3, qkv4, up], dim=1))

            content_feature = self.conv(qkv).view(B, -1, W*2 * H*2).permute(0, 2, 1)  # HW x 1
            content_feature = self.softmax(content_feature)

            channel_feature = self.channel_conv(qkv).view(B, -1, W*2 * H*2)  # C x HW
            channel_pooling = torch.bmm(channel_feature, content_feature).view(B, -1, 1, 1)  # C x 1 x 1
            channel_weight = self.channel_trans_conv(channel_pooling)
            att_cha = qkv * channel_weight

            x = self.proj(att_cha)

            return x


class CFVB(nn.Module):
    r"""cross-layer fusion voting block"""
    def __init__(self, in_channels:int, ratio=8):
        super(CFVB, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.channel_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1),
            nn.LayerNorm([in_channels//ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1)
        )

        # self.linear = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.voting_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x1, x2):
        B, C, W, H = x1.size()

        content_feature1 = self.conv(x1).view(B, -1, W * H).permute(0, 2, 1)  # HW x 1
        content_feature1 = self.softmax(content_feature1)
        channel_feature1 = self.channel_conv(x1).view(B, -1, W * H)  # C x HW
        channel_pooling1 = torch.bmm(channel_feature1, content_feature1).view(B, -1, 1, 1)  # C x 1 x 1
        channel_weight1 = self.channel_trans_conv(channel_pooling1)

        content_feature2 = self.conv(x2).view(B, -1, W * H).permute(0, 2, 1)  # HW x 1
        content_feature2 = self.softmax(content_feature2)
        channel_feature2 = self.channel_conv(x2).view(B, -1, W * H)  # C x HW
        channel_pooling2 = torch.bmm(channel_feature2, content_feature2).view(B, -1, 1, 1)  # C x 1 x 1
        channel_weight2 = self.channel_trans_conv(channel_pooling2)

        # channel_weight = self.linear(torch.cat([channel_weight1, channel_weight2], dim=1))
        channel_weight = channel_weight1 + channel_weight2

        # att_cha1 = x1 * channel_weight
        # att_cha2 = x2 * channel_weight

        # out = att_cha1 + att_cha2
        x2 = x2 * channel_weight
        # channel_weight = self.linear(torch.cat([channel_weight1, channel_weight2], dim=1))
        #
        # att_cha1 = x1 * channel_weight
        # att_cha2 = x2 * channel_weight
        #
        # att = torch.cat([att_cha1, att_cha2], dim=1)
        v1 = self.voting_gate(x2)
        # v2 = self.voting_gate(x1)

        # v = self.voting_gate(att)

        return v1


class ResConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, res=True):
        super(ResConvBlock, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, dilation=1),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, dilation=1),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True)
        # )
        self.large_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        # self.large_conv = nn.Sequential(
        #         nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, dilation=1),
        #         nn.BatchNorm2d(ch_out),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, dilation=1),
        #         nn.BatchNorm2d(ch_out),
        #         nn.ReLU(inplace=True)
        # )


        self.small_conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.res = res

        self.downsample = nn.Sequential()
        if ch_in != ch_out:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # identity = self.small_conv(x)
        identity = x
        x = self.large_conv(x)

        if self.res == True:
            x = x + self.downsample(identity)

        return x


class RCB(nn.Module):
    def __init__(self, ch_in, res=True):
        super(RCB, self).__init__()
        self.RCB = nn.Sequential(
            # nn.Conv2d(ch_in, ch_in, kernel_size=11, stride=1, padding=5, dilation=1),
            ResConvBlock(ch_in, ch_in),
            ResConvBlock(ch_in, ch_in),
            ResConvBlock(ch_in, ch_in),
            ResConvBlock(ch_in, ch_in),
            ResConvBlock(ch_in, ch_in),
            # nn.Conv2d(ch_in, ch_in, kernel_size=11, stride=1, padding=5, dilation=1),
        )

    def forward(self, x):
        x = self.RCB(x)
        return x


class MAAB(nn.Module):
    r"""multi-scale attention aggregation block"""
    def __init__(self, in_channels_1:int , in_channels_2:int, ratio=8):
        super(MAAB, self).__init__()
        self.query_conv = nn.Conv2d(in_channels_1, in_channels_1 // ratio, kernel_size=3, padding=1, bias=False)       # in_ch1 < in_ch2
        self.key_conv = nn.Conv2d(in_channels_2, in_channels_1 // ratio, kernel_size=3, padding=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels_2, in_channels_1, kernel_size=3, padding=1, bias=False)

        self.conv = nn.Conv2d(in_channels_1, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.channel_conv = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=3, padding=1, bias=False)

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(in_channels_1, in_channels_1//ratio, kernel_size=1),
            nn.LayerNorm([in_channels_1//ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_1//ratio, in_channels_1, kernel_size=1)
        )


    def forward(self, x1, x2):
        B, C1, W1, H1 = x1.size()
        B, C2, W2, H2 = x2.size()
        query = self.query_conv(x1).view(B, -1, W1 * H1)
        key = self.key_conv(x2).view(B, -1, W2 * H2).permute(0, 2, 1)
        value = self.value_conv(x2).view(B, -1, W2 * H2)

        content = torch.bmm(key, query)
        content = self.softmax(content)
        att_con = torch.bmm(value, content).view(B, -1, W1, H1) + x1

        content_feature = self.conv(x1).view(B, -1, W1 * H1).permute(0, 2, 1)       # HW x 1
        content_feature = self.softmax(content_feature)
        channel_feature = self.channel_conv(x1).view(B, -1, W1 * H1)                # C x HW
        channel_pooling = torch.bmm(channel_feature, content_feature).view(B, -1, 1, 1)       # C x 1 x 1
        channel_weight = self.channel_trans_conv(channel_pooling)
        att_cha = x1 * channel_weight

        out = att_con + att_cha

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocalFFN(nn.Module):
    r"""Local Feed-Forward Network"""
    def __init__(self, dim, stride=1, padding=2, dilation=2, expand_ratio=4):
        super(LocalFFN, self).__init__()
        hidden_dim = dim * expand_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, dilation, groups=hidden_dim),         # 深度卷积（Depth-wise Convolution）
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),                                                       # 逐点卷积（Point-wise Convolution）
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.DWConv(x)
        x = self.conv2(x)

        return x


class Att(nn.Module):
    def __init__(self, in_channels:int, height, width, ratio=8):
        super(Att, self).__init__()

        self.rel_h = nn.Parameter(torch.randn([1, in_channels // ratio, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, in_channels // ratio, 1, width]), requires_grad=True)

        self.query_conv = nn.Linear(in_channels, in_channels // ratio)
        self.key_conv = nn.Linear(in_channels, in_channels // ratio)
        self.value_conv = nn.Linear(in_channels, in_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, L, C = x.shape
        query = self.query_conv(x)
        key = self.key_conv(x).permute(0, 2, 1)
        value = self.value_conv(x)

        position = (self.rel_h + self.rel_w).view(1, -1, L)
        content_position = torch.matmul(query, position)

        content = torch.bmm(query, key) + content_position
        content = self.softmax(content)
        att_con = torch.bmm(content, value)

        return att_con


class TransformerBlock(nn.Module):
    def __init__(self, in_features, height, width):
        super(TransformerBlock, self).__init__()

        self.GC = DoubleConv(in_features, in_features)
        self.alpha = nn.Parameter(torch.ones(1))

        self.norm1 = nn.LayerNorm(in_features)
        self.att = Att(in_features, height, width)
        # self.norm2 = nn.LayerNorm(in_features)
        # self.mlp = Mlp(in_features)
        self.norm2 = nn.BatchNorm2d(in_features)
        self.LocalFFN = LocalFFN(in_features)

    def forward(self, x, H, W):
        # gx = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        # gx = self.GC(gx)

        tx = self.att(self.norm1(x)) + x
        tx = rearrange(tx,'b (h w) c -> b c h w', h=H, w=W)
        mx = self.norm2(self.LocalFFN(tx)) + tx

        # return mx + gx * self.alpha
        return mx


class PatchMerged(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerged, self).__init__()

        self.dwt = Dwt2d()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape

        x = self.dwt(x)

        x = x.flatten(2).transpose(1, 2)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, H//2, W//2


class PatchExpanded(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchExpanded, self).__init__()

        self.Iwt = Iwt2d()
        self.expand = nn.Linear(dim // 4, dim // 2, bias=False)
        self.norm = norm_layer(dim // 4)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape

        x = self.Iwt(x)

        x = x.flatten(2).transpose(1, 2)  # B H*2*W*2 C//4

        x = self.norm(x)
        x = self.expand(x)

        return x, H * 2, W * 2


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels):
        super(PatchEmbed, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class BasicStem(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, act=nn.LeakyReLU()):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.act = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        return x

class DCUNet(nn.Module):
    def __init__(self, in_ch, out_ch, size=320):
        super(DCUNet, self).__init__()

        self.stem = BasicStem(in_ch=32, out_ch=128)
        self.PatchEmbed = PatchEmbed(4, 128)

        self.Magnify_Transformer_1 = TransformerBlock(128, size // 4, size // 4)
        self.SimpleConv_1 = nn.ConvTranspose2d(128, 128, 1)

        self.PatchMerged_2 = PatchMerged(128)
        self.Magnify_Transformer_2 = TransformerBlock(256, size // 8, size // 8)
        self.SimpleConv_2 = nn.ConvTranspose2d(256, 256, 1)

        self.PatchMerged_3 = PatchMerged(256)
        self.Magnify_Transformer_3 = TransformerBlock(512, size // 16, size // 16)
        self.SimpleConv_3 = nn.ConvTranspose2d(512, 512, 1)

        #
        self.PatchExpanded_4 = PatchExpanded(512)
        self.Magnify_Transformer_4 = TransformerBlock(256, size // 8, size // 8)
        self.SimpleConv_4 = nn.ConvTranspose2d(256, 256, 1)

        self.PatchExpanded_5 = PatchExpanded(256)
        self.Magnify_Transformer_5 = TransformerBlock(128, size // 4, size // 4)
        self.SimpleConv_5 = nn.ConvTranspose2d(128, 128, 1)

        self.FinalPatchExpand_X4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.FinalConv = self.conv2 = DoubleConv(128, 32)


        ##
        self.conv1 = DoubleConv(in_ch, 32)
        # self.pool1 = nn.MaxPool2d(2)
        self.pool1 = Dwtpool(32)
        self.conv2 = DoubleConv(32, 64)
        # self.pool2 = nn.MaxPool2d(2)
        self.pool2 = Dwtpool(64)
        self.conv3 = DoubleConv(64, 128)
        # self.pool3 = nn.MaxPool2d(2)
        self.pool3 = Dwtpool(128)
        self.conv4 = DoubleConv(128, 256)
        # self.pool4 = nn.MaxPool2d(2)
        self.pool4 = Dwtpool(256)
        self.conv5 = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.up1 = IwtUp(512)
        self.conv6 = DoubleConv(1024, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.up2 = IwtUp(256)
        self.conv7 = DoubleConv(512, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.up3 = IwtUp(128)
        self.conv8 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.up4 = IwtUp(64)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

        self.maab1 = MAAB(128, 256)
        self.maab2 = MAAB(256, 512)
        # self.maab3 = MAAB(512, 128)

        self.rcb1 = RCB(32)
        self.rcb2 = RCB(64)

        self.CFVB1 = CFVB(256)
        self.CFVB2 = CFVB(128)
        self.CFVB3 = CFVB(64)
        self.CFVB4 = CFVB(32)
        self.CFVB5 = CFVB(512)


    def forward(self, x):

        # ## -------------------Recode-------------------
        # #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print(p2.shape)

        #
        c3 = self.conv3(p2)  # 80
        p3 = self.pool3(c3)
        # print(p3.shape)
        c4 = self.conv4(p3)  # 40
        p4 = self.pool4(c4)
        # print(p4.shape)
        c5 = self.conv5(p4)  # 20

        rcb1 = self.rcb1(c1)
        rcb2 = self.rcb2(c2)
        # maab1 = self.maab1(c2, c3)
        maab1 = self.maab1(c3, c4)
        maab2 = self.maab2(c4, c5)


        ## -------------Feature Magnification-------------
        h1 = self.stem(c1)

        h1, H, W = self.PatchEmbed(h1)
        xh1 = self.Magnify_Transformer_1(h1, H, W)      # 128
        xh1 = xh1 + self.SimpleConv_1(c3)

        h2, H, W = self.PatchMerged_2(xh1)
        xh2 = self.Magnify_Transformer_2(h2, H, W)      # 256
        xh2 = xh2 + self.SimpleConv_2(c4)

        # h3, H, W = self.PatchMerged_3(xh2)
        # xh3 = self.Magnify_Transformer_3(h3, H, W)      # 512
        # xh3 = xh3 + self.SimpleConv_3(c5)

        # h4, H, W = self.PatchMerged_4(xh3)
        # xh4 = self.Magnify_Transformer_4(h4, H, W)

        # maab3 = self.maab3(xh3, xh1)

        # h4, H, W = self.PatchExpanded_4(maab3)
        # xh4 = self.Magnify_Transformer_4(h4, H, W)
        # xh4 = torch.cat([xh4, xh2], dim=1)              # 512
        #
        # h5, H, W = self.PatchExpanded_5(xh4)
        # xh5 = self.Magnify_Transformer_5(h5, H, W)      # 256
        # xh5 = torch.cat([xh5, xh1], dim=1)
        #
        # xh6 = self.FinalPatchExpand_X4(xh5)             # 32
        # xh6 = self.FinalConv(xh6)


        # ## -------------------Decode-------------------
        # merge0 = torch.cat([c5, maab3], dim=1)
        # v0 = self.CFVB5(c5, maab3)
        # merge0_vote = merge0 * v0
        # up_1 = self.up1(merge0_vote)

        up_1 = self.up1(c5)
        c_v1 = self.CFVB1(up_1, maab2)
        t_v1 = self.CFVB1(up_1, xh2)
        c_merge1 = torch.cat([up_1, maab2], dim=1)      # 512
        c_merge1_vote = c_merge1 * c_v1
        t_merge1 = torch.cat([up_1, xh2], dim=1)
        t_merge1_vote = t_merge1 * t_v1
        v1 = self.CFVB5(c_merge1_vote, t_merge1_vote)
        merge1 = torch.cat([c_merge1_vote, t_merge1_vote], dim=1)
        merge1_vote = merge1 * v1
        # merge1_vote = torch.max(merge1_vote, merge1)
        c6 = self.conv6(merge1_vote)

        up_2 = self.up2(c6)
        c_v2 = self.CFVB2(up_2, maab1)
        t_v2 = self.CFVB2(up_2, xh1)
        c_merge2 = torch.cat([up_2, maab1], dim=1)      # 256
        c_merge2_vote = c_merge2 * c_v2
        t_merge2 = torch.cat([up_2, xh1], dim=1)
        t_merge2_vote = t_merge2 * t_v2
        v2 = self.CFVB1(c_merge2_vote, t_merge2_vote)
        merge2 = torch.cat([c_merge2_vote, t_merge2_vote], dim=1)
        merge2_vote = merge2 * v2
        # merge2_vote = torch.max(merge2_vote, merge2)
        c7 = self.conv7(merge2_vote)

        up_3 = self.up3(c7)
        v3 = self.CFVB3(up_3, rcb2)
        merge3 = torch.cat([up_3, rcb2], dim=1)       # 128
        merge3_vote = merge3 * v3
        # merge3_vote = torch.max(merge3_vote, merge3)
        c8 = self.conv8(merge3_vote)

        up_4 = self.up4(c8)
        v4 = self.CFVB4(up_4, rcb1)
        merge4 = torch.cat([up_4, rcb1], dim=1)       # 64
        merge4_vote = merge4 * v4
        # merge4_vote = torch.max(merge4_vote, merge4)
        c9 = self.conv9(merge4_vote)

        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)
        return out



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DCUNet(1, 1).to(device)
    # 打印网络结构和参数
    # summary(net, (1, 320, 320), batch_size=4)
    print(net)


# 阶段性连续地投票加权对OCTA图像是有效的