from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=1,
                     padding=0, bias=False)

'''
Modified from https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py
'''


class VID(nn.Module):
    '''
    Variational Information Distillation for Knowledge Transfer
    https://zpascal.net/cvpr2019/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf
    '''

    def __init__(self, in_channels, mid_channels, out_channels, init_var, eps=1e-6):
        super(VID, self).__init__()
        self.eps = eps
        self.regressor = nn.Sequential(*[
            conv1x1(in_channels, mid_channels),
            nn.ReLU(),
            conv1x1(mid_channels, mid_channels),
            nn.ReLU(),
            conv1x1(mid_channels, out_channels),
        ])
        self.alpha = nn.Parameter(
            np.log(np.exp(init_var - eps) - 1.0) * torch.ones(out_channels)#np 取对数操作  np.exp(x)函数返回以e为底，以x为指数的指数值。
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        pred_mean = self.regressor(fm_s)
        pred_var = torch.log(1.0 + torch.exp(self.alpha)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (torch.log(pred_var) + (pred_mean - fm_t) ** 2 / pred_var)
        loss = torch.mean(neg_log_prob)

        return

import torch
import torch.nn as nn

class CustomBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CustomBinaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # 将输入通过一个激活函数，例如Sigmoid

        # input.clamp(min=0)
        # 表示对输入input进行下界修剪，将小于等于0的值替换为0
        # 它可以用于引入非线性特性或者进行数值调整。(1 + neg_abs.exp()).log()  neg_abs = - input.abs()
        probabilities = torch.sigmoid(input)

        # 计算二分类交叉熵损失
        loss = -torch.log(probabilities) * target - torch.log(1 - probabilities) * (1 - target)

        # 根据权重调整损失
        if self.weight is not None:
            loss = loss * self.weight

        # 根据reduction参数计算最终损失值
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss



e = torch.randn(3, 128, 52, 52)
f = torch.randn(3, 128, 52, 52)
# contrastive_loss = DistillKLml3(2,128,1)
# loss = contrastive_loss(e,f,a)
# Similarity = Similarity()
# loss= Similarity(a,d)
# print(loss)
CustomBinaryCrossEntropyLoss = CustomBinaryCrossEntropyLoss(reduction='mean')
CustomBinaryCrossEntropyLoss = CustomBinaryCrossEntropyLoss(e,f)
print(CustomBinaryCrossEntropyLoss)