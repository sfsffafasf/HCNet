import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
# from toolbox.models.SPNet.lib.res2net_v1b_base import Res2Net_model
from backbone.backbone.mix_transformer import mit_b5,mit_b1
from torch.nn.parameter import Parameter


# backbone/backbone/mix_transformer.py

# from toolbox.models.l2model.mmcv_ops_saconv import SAConv2dB
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 卷积
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        # self.conv = Dynamic_conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=dilation, bias=False)  ##改了动态卷积
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class SAEd(nn.Module):
    def __init__(self, channels_in):
        super(SAEd, self).__init__()
        self.se_rgb = BasicConv2d(channels_in // 2, channels_in // 2, 1)
        self.se_depth = nn.Sequential(BasicConv2d(channels_in, channels_in * 2, 1),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      BasicConv2d(channels_in * 2, channels_in // 2, 3, padding=7, dilation=7)
                                      )
        self.layer_ful2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 4, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channels_in // 4, channels_in // 2, kernel_size=1),
        )
        self.secovn = BasicConv2d(channels_in // 2, channels_in // 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels_in // 2, channels_in // 4),
            nn.LeakyReLU(),
            nn.Linear(channels_in // 4, channels_in // 2),
            nn.LeakyReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.LeakyReLU()
    def forward(self, rgb, depth):  # 可以改为其他的权重

        b, c, _, _ = rgb.size()
        depth = self.se_depth(depth)
        rgb1 = self.se_rgb(rgb)
        rgbweight = self.avg_pool(depth + rgb1 - torch.mean(depth + rgb1)).view(b, -1)
        weighting = self.fc(rgbweight).view(b, c, 1, 1)
        jout = rgb * torch.mean(rgb, dim=1, keepdim=True) + rgb * (nn.Sigmoid()(weighting))
        Cout = depth * torch.mean(depth, dim=1, keepdim=True) + depth * (nn.Sigmoid()(weighting))
        out = self.layer_ful2(torch.cat([jout, Cout], dim=1))
        out = out * (rgb + depth)
        out = self.relu(self.secovn(out))
        return out

       # print(co0.shape,co3.shape,s1.shape,s2.shape)
       # torch.Size([2, 64, 104, 104]) torch.Size([2, 512, 13, 13]) torch.Size([2, 256, 26, 26]) torch.Size([2, 64, 104, 104])
# torch.Size([2, 128, 26, 26]) torch.Size([2, 256, 13, 13])
class decode(nn.Module):
    def __init__(self, img1channel, img2channel):
        super(decode, self).__init__()
        # self.fu_1 = CIM(img2channel, img2channel)  # MixedFusion_Block_IMfusion 128,256

        self.m = BasicConv2d(img1channel*2, img1channel, kernel_size=3, stride=1, padding=1)
        self.co = BasicConv2d(img2channel, img1channel, kernel_size=3, stride=1, padding=1)
        self.out = BasicConv2d(img1channel, img1channel, 1)
        self.relu = nn.LeakyReLU()
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self,s1,co0):
        ################################[2, 32, 28, 28]
        _, _, height, width = s1.size()
        if height != co0.size(2) or width != co0.size(2):
            # s1 = F.interpolate(s1, size=.size(2), mode='bilinear', align_corners=False)
            co0 = F.interpolate(co0, size=(s1.size(2), s1.size(3)), mode='bilinear', align_corners=False)
            co0 = self.co(co0)
        cat = self.m(torch.cat([s1,co0],dim=1))
        fea_c = co0 * cat + s1 * cat
        fea_c = self.upsample_2(fea_c)
        out = self.relu(self.out(fea_c))
        return out


class BidirectionalAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, ):
        super(BidirectionalAttention, self).__init__()

        # self.query_conv = BasicConv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.query_conv = BasicConv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = BasicConv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.conv_adj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=in_channels // reduction_ratio),
            nn.Conv1d(in_channels // reduction_ratio, in_channels // reduction_ratio* 2, 3, 1, 1, groups=1),

            nn.BatchNorm1d(in_channels // reduction_ratio * 2),
        )

        self.co_conv = nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1)
    def forward(self, x):
        # x = self.ful2(torch.cat([x,y],dim=1))
        batch_size, channels, height, width = x.size()
        # query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        query = self.query_conv(x).view(batch_size, -1, height * width)
        query = self.conv_adj(query).reshape(batch_size, -1, 2, height * width).permute(2, 0, 1, 3)
        query_y, key_y = query.unbind(0)
        query_y = query_y.permute(0, 2, 1)

        value = self.value_conv(x).view(batch_size, -1, height * width)
        # 计算 query 和 key 的相似度得分
        attn = torch.bmm(query_y, key_y)

        attn_f = F.softmax(attn, dim=0)
        attn_b = F.softmax(attn, dim=1)
        fusion_f = torch.bmm(value, attn_f.permute(0, 2, 1))
        fusion_b = torch.bmm(value, attn_b.permute(0, 2, 1))
        fusion = self.gamma * (fusion_f + fusion_b)
        spatial_attn = self.ap(torch.flatten(x, 2))
        fusion = fusion * spatial_attn

        fusion = fusion.view(batch_size, channels, height, width)+x
        fusion = self.co_conv(fusion)
        # one_block = torch.ones_like(fusion1)
        # fea_out = fusion * (one_block - fusion1) + x * fusion1
        # 返回融合后的特征
        return fusion
"""
rgb和d分别与融合的做乘法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
rgb和d分别与融合的做加法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
输出就是融合
"""
class CO(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CO, self).__init__()
        self.rgb = BasicConv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.dep = BasicConv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_ful2 = BasicConv2d(2 * out_dim, out_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, rgb, dep):
        rgb = self.rgb(rgb)
        dep = self.dep(dep)
        mm_cat2 = torch.cat([rgb, dep], dim=1)
        cim_out = self.layer_ful2(mm_cat2) + rgb + dep
        return cim_out
####################################################自适应1,2,3,6###########################
class LiSPNetx22(nn.Module):
    def __init__(self, channel=32):
        super(LiSPNetx22, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # Backbone model   In
        res_channels = [32, 64, 160, 256, 256]
        channels = [64, 128, 256, 512, 512]
        self.resnet = mit_b1()
        # self.resnet_depth = mit_b1()
        self.resnet.init_weights("/home/li/桌面/kuangjia/backbone/backbone/mit_b1.pth")
        # self.resnet_depth.init_weights("/media/user/7810057410053B20/kuangjia/backbone/backbone/mit_b1.pth")
        ###############################################
        # funsion encoders #
        ## rgb
        # self.rgb1 = BasicConv2d(64, 1, 1)
        self.conv_out1 = nn.Conv2d(64, 1, 1)
        self.conv_out2 = nn.Conv2d(64, 1, 1)
        self.conv_out3 = nn.Conv2d(32, 1, 1)
        self.rSE = SAEd(256)
        self.rSE1 = SAEd(128)
        # self.dualgcn = DualGCN(256)
        # self.d1to3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #                            nn.BatchNorm2d(128), nn.LeakyReLU(inplace=True))
        # self.d1to1 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.d2to3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #                            nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True))
        # self.d1to4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 1),
        #                            nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True))
        # self.d1to2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True))
        self.CO0 = CO(512, 256)
        self.CO1 = CO(320, 128)
        self.CO2 = CO(128, 64)
        self.CO3 = CO(64, 32)
        res_channels = [32, 64, 160, 256, 256]
        self.decode2 = decode(128,256)
        self.decode1 = decode(64,128)
        # self.SPA = BidirectionalAttention(64)
        # self.conv = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.convf1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.convf2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
    def forward(self, imgs,depths):
        # depths = imgs
        img_0, img_1, img_2, img_3 = self.resnet.forward_features(imgs)
        ####################################################
        ## decoder rgb     ful_2.shape[2, 256, 14, 14]   img_3.shape [2, 256, 7, 7]
        ####################################################
        dep_0, dep_1, dep_2, dep_3 = self.resnet.forward_features(depths)
        co0 = self.CO3(img_0, dep_0)
        co1 = self.CO2(img_1, dep_1)
        co2 = self.CO1(img_2, dep_2)
        co3 = self.CO0(img_3, dep_3)
        # print(co1, co2)
        s1 = self.rSE1(co1, co2)
        s2 = self.rSE(co2, co3)  # torch.Size([2, 256, 26, 26]) torch.split(input_tensor, split_size_or_sections=num_channels, dim=1)
        # torch.Size([2, 32, 104, 104]) torch.Size([2, 32, 104, 104])
        # print(s1.shape, co2.shape)#torch.Size([2, 64, 52, 52]) torch.Size([2, 128, 26, 26])
        ful1 = self.decode1(s1,co2)
        ful2 = self.decode2(s2,co3)
        # print(ful1.shape,ful2.shape)
        ful2 = self.conv2(ful2)
        ful = self.convf1(ful1)*co0+ self.convf2(ful2)*co0
        # print(ful.shape)
        # ful = self.SPA(ful)
        # img_02 = self.d2to3(s1 * s2)
        # ful = img_02* co0 + co0 * self.upsample_2((self.d1to1(s1))) + co0
        # print(imgs.shape, img_0.shape, img_1.shape, img_2.shape, img_3.shape)torch.Size([2, 3, 416, 416]) torch.Size([2, 32, 104, 104]) torch.Size([2, 64, 52, 52]) torch.Size([2, 160, 26, 26]) torch.Size([2, 256, 13, 13])
        # 512,13 #此处加了se注意力
        # ful_0 = self.ful_0(ful_1, img_01, dep_01)
        ########2，256，13      32,208     64,104
        ful1 = self.conv_out1(self.upsample_4(ful1))
        # ful2 = self.conv_out2(self.upsample_8(ful2))
        ful3 = self.conv_out3(self.upsample_4(ful))

        return ful1, ful3,ful2,co0, co1,co2, co3,s1,s2


if __name__ == "__main__":
    model = LiSPNetx22().cuda()
    rgb = torch.randn(2, 3, 416, 416).cuda()
    t = torch.randn(2, 3, 416, 416).cuda()
    out = model(rgb)
    for i in range(len(out)):
        print(out[i].shape)
# from toolbox import compute_speed
# from ptflops import get_model_complexity_info
# with torch.cuda.device(0):
#     net = LiSPNetx22().cuda()
#     flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=False)
#     print('Flops:' + flops)
#     print('Params:' + params)
# print(a.shape)
# Flops:33.52 GMac
# Params:190.26 M
# print(a[1].shape)Flops:16.37 GMac
# Params:20.05 M
# print(a[2].shape)
# print(a[3].shape)
# compute_speed(net,input_size=(1, 3, 416, 416), iteration=500)
# Flops:48.56 GMac
# Params:93.47 M
