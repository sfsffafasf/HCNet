import torch
import torch.nn.functional as F
# from util import *
from math import exp, pow
import numpy as np
from PIL import Image
import torch.nn as nn
from toolbox import Dice_loss


def get_contour(label):
    lbl = label.gt(0.5).float() #mask=y_pred.ge(0.5).float().squeeze() # 以0.5为阈值进行分类 correct=(mask==train_y).sum() # 计算正确预测的样本个数 acc=correct.item()/train_y.size(0) # 计算分类准确率
    ero = 1 - F.max_pool2d(1 - lbl, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(lbl, kernel_size=5, stride=1, padding=2)            # dilation
    edge = dil - ero
    return edge

# Boundary-aware Texture Matching Loss
def BTMLoss(pred, image, radius, config=None):
        alpha = config['rgb']
        modal = config['trset']
        num_modal = len(modal) if 'c' in modal else len(modal)+1

        slices = range(0, 3*num_modal+1, 3)
        sal_map =  F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=True)
        image_ = F.interpolate(image, size=sal_map.shape[-2:], mode='bilinear', align_corners=True)
        mask = get_contour(sal_map)
        features = torch.cat([image_, sal_map], dim=1)

        N, C, H, W = features.shape
        diameter = 2 * radius + 1
        kernels = F.unfold(features, diameter, 1, radius).view(N, C, diameter, diameter, H, W)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        dis_modal = 1
        for idx, slice in enumerate(slices):
            if idx == len(slices) - 1:
                continue
            dis_map = (-alpha * kernels[:, slice:slices[idx+1]] ** 2).sum(dim=1, keepdim=True).exp()
            # Only RGB
            if config['only_rgb'] and idx > 0:
                dis_map = dis_map * 0 + 1
            dis_modal = dis_modal * dis_map

        dis_sal = torch.abs(kernels[:, slices[-1]:])
        distance = dis_modal * dis_sal

        loss = distance.view(N, 1, (radius * 2 + 1) ** 2, H, W).sum(dim=2)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

# Confidenceaware Saliency Distilling Loss
def CSDloss(pred, feat, mask=False, epoch=1, config=None, name=None):
    mul = 2 ** (1 - (epoch - 1) / config['epoch'])
    loss_map = torch.abs(pred - 0.5)
    loss_map = torch.pow(loss_map, mul)
    
    # The pow(0.5, mul) is used to keep the loss greater than 0. It has no impact on the training process.
    loss = pow(0.5, mul) - loss_map.mean() 
    return loss

def Loss(pre1, img1, pre2, img2, epoch, ws, config, name):
    sal1 = pre1['sal'][0]
    sal2 = pre2['sal'][0]
    
    p1 = torch.sigmoid(sal1)
    p2 = torch.sigmoid(sal2)
    
    adb_loss = CSDloss(p1, pre1['feat'][0], False, epoch, config, name) + CSDloss(p2, pre2['feat'][0], False, epoch, config, name)
    
    if ws[1] > 0:
        ac_loss = BTMLoss(p1, img1, 5, config) + BTMLoss(p2, img2, 5, config)
    else:
        ac_loss = 0
    
    p2 = F.interpolate(p2, size=p1.size()[2:], mode='bilinear', align_corners=True)
    mse_loss = torch.mean(torch.pow(p1 - p2, 2))
    
    return adb_loss * ws[0], ac_loss * ws[1], mse_loss * ws[2]

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# writer = SummaryWriter("logs")
# ==============================蒸馏损失===============================
class DistillKLm(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKLm, self).__init__()
        self.T = T
        self.fc1 = nn.Conv2d(128, 64, 1)
        self.fc2 = nn.Conv2d(64, 1, 1)
        # self.fc2 = nn.Conv2d(K, K, 1, )
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    def forward(self, ful_2,y_sr, y_sd,ful):
        # student网络输出软化后结果
        # log_softmax与softmax没有本质的区别，只不过log_softmax会得到一个正值的loss结果。
        batch_size, in_planes, _, _ = y_sr.size()

        x= F.max_pool2d(ful_2, kernel_size=5, stride=1, padding=2)
        y_sr = self.upsample_8(self.fc1(y_sr))
        y_sr = self.fc2(y_sr)
        y_sd = self.upsample_8(self.fc1(y_sd))
        y_sd = self.fc2(y_sd)
        # y_1r = F.interpolate(self.fc2(y_1r), size=ful.size()[2:], mode='bilinear', align_corners=True)
        # y_1d = F.interpolate(self.fc2(y_1d), size=ful.size()[2:], mode='bilinear', align_corners=True)
        r1 = y_sr * x + y_sr
        r1 = F.relu(r1)
        # r = r1+r2
        r2 = y_sd * x + y_sd
        r2 = F.relu(r2)
        p_s = F.logsigmoid(r1+r2 / self.T)
        p_s1 = F.logsigmoid(ful_2 / self.T)
        # # teacher网络输出软化后结果
        p_t = F.sigmoid(ful / self.T)
        # print(p_s.shape, p_t.shape)
        # 蒸馏损失采用的是KL散度损失函数
        # loss = torch.mean(p_s-p_t)+torch.mean(p_s1-p_t)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / ful.shape[0]
        loss1 = F.kl_div(p_s1, p_t, size_average=False) * (self.T ** 2) / ful.shape[0]
        return loss+loss1


class DistillKLml(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T,in_dim, out_dim):
        super(DistillKLml, self).__init__()
        self.T = T
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.conv1_rgb = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv1_dep = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_ful2 = nn.Conv2d(2 * out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.out_dim = nn.Conv2d(out_dim, out_dim , kernel_size=3, stride=1, padding=1)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.Dice_loss = nn.BCEWithLogitsLoss()

    def forward(self, rgb, dep,at):
        x_rgb = self.conv1_rgb(rgb)
        x_dep = self.conv1_dep(dep)
        mm_cat2 = torch.cat([self.upsample_8(x_dep),self.upsample_8(x_rgb)], dim=1)
        cim_out = self.out_dim(self.layer_ful2(mm_cat2))
        p_s = nn.Sigmoid()(cim_out / self.T)
        p_t = nn.Sigmoid()(at / self.T)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / at.shape[0]
        loss = self.Dice_loss(p_s, p_t)
        return torch.mean(loss)




class Similarity(nn.Module):
    ##Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author##
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss






class DistillKLml3(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T, in_dim, out_dim):
        super(DistillKLml3, self).__init__()
        self.T = T
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.conv1_rgb = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv1_dep = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_ful2 = nn.Conv2d(2 * out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.out_dim = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')
    def forward(self, rgb, dep, at):
        x_rgb = self.conv1_rgb(rgb)
        x_dep = self.conv1_dep(dep)
        mm_cat2 = torch.cat([self.upsample_8(x_dep), self.upsample_8(x_rgb)], dim=1)
        cim_out = self.out_dim(self.layer_ful2(mm_cat2))
        p_s = nn.Sigmoid()(cim_out / self.T)
        p_t = nn.Sigmoid()(at / self.T)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / at.shape[0]
        loss_AT = self.mse(p_s.mean(dim=1), p_t.mean(dim=1))

        x_student = F.log_softmax(p_s, dim=1)
        x_teacher = F.softmax(p_t, dim=1)

        loss_PD = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[1])
        loss = loss_AT + loss_PD

        return loss


import torch
import torch.nn as nn

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()

        # Query, Key, Value参数矩阵
        self.query_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Dropout层
        self.dropout = nn.Dropout(0.1)

        # 注意力分数归一化的比例系数
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, x):
        """
        x: 输入的特征矩阵，维度为(batch_size, seq_len, hidden_dim)
        """
        # 获取batch_size和seq_len
        batch_size, seq_len, hidden_dim = x.size()

        # 得到Query, Key, Value
        Q = self.query_matrix(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key_matrix(x)  # (batch_size, seq_len, hidden_dim)
        V = self.value_matrix(x)  # (batch_size, seq_len, hidden_dim)

        # 计算注意力分数（内积）
        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行缩放
        scaled_scores = scores / self.scale  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行softmax，得到注意力权重
        attn_weights = torch.softmax(scaled_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # 对注意力权重进行dropout
        attn_weights = self.dropout(attn_weights)

        # 将注意力权重与Value相乘，得到self-attention后的表示
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        return attn_output, attn_weights


a = torch.randn(3, 1, 416, 416)
b = torch.randn(3, 1, 52, 52)
c = torch.randn(3, 1, 52, 52)
d = torch.randn(3, 1, 416, 416)
e = torch.randn(3, 128, 52, 52)
f = torch.randn(3, 128, 52, 52)
# contrastive_loss = DistillKLml3(2,128,1)
# loss = contrastive_loss(e,f,a)
# Similarity = Similarity()
# loss= Similarity(a,d)
# print(loss)
SelfAttention = SelfAttention(128)
SelfAttention = SelfAttention(e)
print(SelfAttention)


#
# train_dataset = MyDataset(...)  # 构建训练数据集
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
