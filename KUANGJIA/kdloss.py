
from torch.cuda import amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from toolbox import get_dataset, Ranger, AdamW
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import setup_seed
# from toolbox.losses.loss import DiceLoss, IOU, edge_hold_loss, hcl?
from toolbox.losses import focal_loss
from toolbox.msg import runMsg
# toolbox/models/zhengliu/n11gongsgaisede.py
from toolbox.models.zhengliu.n11gongsgaisede import LiSPNetx22
from toolbox import Dice_loss
from toolbox.losses.ssim_loss import SSIMLoss, IOU
# from toolbox.losses.sunshi1 import Dis

setup_seed(33)


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)


# writer = SummaryWriter("logs")L2(f_) = sqrt(sum((f_ ** 2), dim=1)) + 1e-8
# ==============================蒸馏损失===============================
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Networkzij"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # G_s = y_s-torch.mean(y_s)
        # G_t = y_t-torch.mean(y_t)
        #
        # G_s = L2(G_s).detach()
        # G_s = torch.nn.functional.normalize(G_s)
        #
        # G_t = L2(G_t).detach()
        # G_t = torch.nn.functional.normalize(G_t)
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # 蒸馏损失采用的是KL散度损失函数
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]

        return loss



    # writer = SummaryWriter("logs")
# ==============================蒸馏损失===============================


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')

    def _resize(self, x, x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear', align_corners=False)
        return x

    def forward(self, x_student, x_teacher):
        # x_student = self._resize(x_student,x_teacher)
        loss_AT = self.mse(x_student.mean(dim=1), x_teacher.mean(dim=1))

        x_student = F.log_softmax(x_student, dim=1)
        x_teacher = F.softmax(x_teacher, dim=1)

        loss_PD = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[1])
        loss = loss_AT + loss_PD
        return loss

def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8
class Dis(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(Dis, self).__init__()
        self.T = T
        self.self_attention_map = self_attention_map
        self.mse = nn.MSELoss(reduce='mean')
        self.alpha = nn.Parameter()
        # self.WeightedFocalLoss = WeightedFocalLoss()

        self.Similarity = Similarity()

    def forward(self, y_s, y_t):
        # student网络输出软化后结果

        # 蒸馏损失采用的是KL散度损失函数
        kl_loss = F.kl_div(y_s, y_t)
        y_s_attention_map = self.self_attention_map(y_s)
        y_t_attention_map = self.self_attention_map(y_t)
        mse = self.mse(y_s_attention_map, y_t_attention_map)
        loss = kl_loss + mse
        # print(kl_loss,mse,WeightedFocalLoss)
        return loss.mean()

def self_attention_map(input_features):
    # 输入特征形状：[batch_size, num_heads, seq_len, embedding_dim]
    # batch_size, num_heads, seq_len, embedding_dim = input_features.size()
    # 计算注意力分数
    # input_features = torch.nn.functional.normalize(input_features)
    scores = torch.matmul(input_features,
                          input_features.transpose(-2, -1))  # 形状：[batch_size, num_heads, seq_len, seq_len]
    # 归一化注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    # 计算自注意力图
    self_attention_map = torch.matmul(attention_weights,
                                      input_features)  # 形状：[batch_size, num_heads, seq_len, embedding_dim]

    return self_attention_map
# torch.Size([2, 32, 104, 104]) torch.Size([2, 64, 52, 52]) torch.Size([2, 128, 26, 26]) torch.Size([2, 256, 13, 13])
class ca(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.conv1_rgb = nn.Sequential(nn.Conv2d(in_dim, in_dim//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim//2))
        self.conv1_dep = nn.Sequential(nn.Conv2d(in_dim//2, in_dim//4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim//4))
        self.conv2_dep = nn.Sequential(nn.Conv2d(in_dim // 4, in_dim // 8, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim//8))
        self.layer_ful2 = nn.Sequential(nn.Conv2d(in_dim // 4, in_dim//8, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim//8),nn.LeakyReLU())
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.self_attention_map = self_attention_map
    def forward(self, co0, co1,co2, co3):
        # x_student = self._resize(x_student,x_teacher)
        co3 = self.upsample_2(self.conv1_rgb(co3.float()))
        co2 = self.upsample_2(self.conv1_dep(co3+co2.float()))
        co1 = self.upsample_2(self.conv2_dep(co2+co1.float()))
        mm_cat2 = torch.cat([co1, co0.float()], dim=1)
        cim_out = self.self_attention_map(mm_cat2)
        return cim_out
def SCL(y_s,y):
    # Normalise representations
    #print(sum(y_s[0][0][0]))
    y_s = F.interpolate(y_s, size=(13, 13), mode='bilinear')
    z_s_norm = F.normalize(y_s, dim=1)
    b,c,h,w=z_s_norm.shape
    z_s_norm=z_s_norm.reshape(b,c,h*w).permute(0,2,1)
    z_s_norm_t=z_s_norm.permute(0,2,1)
    c_ss = torch.bmm(z_s_norm, z_s_norm_t)
    y_d = F.interpolate(y,size=(13,13),mode='bilinear')
    b, c, h, w = y_d.shape
    y_d = y_d.reshape(b, c, h * w).permute(0, 2, 1)
    y_d_t = y_d.permute(0, 2, 1)
    yy = torch.bmm(y_d, y_d_t)
    loss=0.0
    loss += torch.log2(c_ss.pow(2).sum()) / (h*h*w*w)
    loss -= torch.log2((c_ss * yy).pow(2).sum()) / (h*h*w*w)
    return loss

class Similarity(nn.Module):
    ##Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author##
    def __init__(self):
        super(Similarity, self).__init__()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        # self.conv1_dep = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        # self.layer_ful2 = nn.Conv2d(2 * out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        # self.out_dim = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, g_s, g_t):
        # x_rgb = self.conv1_rgb(g_s)
        # x_dep = self.conv1_dep(g_t)
        # mm_cat2 = torch.cat([x_dep, x_rgb], dim=1)
        # cim_out = self.out_dim(self.layer_ful2(mm_cat2))
        # g_s = g_s*cim_out
        # g_t = g_t*cim_out
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
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)+self.SmoothL1Loss(f_s,f_t).mean(-1)
        return loss


class Loss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(Loss, self).__init__()
        self.binary_cross_entropy = nn.BCELoss()
        self.iou = IOU()
        # self.SSIMLoss = focal_loss()
        self.semantic_loss = nn.BCEWithLogitsLoss()
        self.ca = ca(256)
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = DistillKL(1)
        # self.KLD1 = DistillKL(2)
        self.Dice_loss = Dice_loss.BinaryDiceLoss()
        self.Similarity = Similarity()
        self.smooth_loss = nn.SmoothL1Loss()
        self.Dis = Dis(1)
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.BCE = F.binary_cross_entropy_with_logits
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0)
        self.SCL = SCL

    def forward(self, inputs, inputsT, targets):
        out1s, out2s,out3s,co0s, co1s,co2s, co3s,s1s,s2s= inputs
        out1, out2,out3,co0, co1,co2, co3,s1,s2,ful = inputsT
        target = targets
        bce = torch.unsqueeze(target, 1).float()
        # teacher = nn.Sigmoid()(teacher)
        # print(img.shape, dep.shape, img2.shape, dep2.shape,c.shape,d.shape,e.shape)
        loss1 = self.iou(torch.sigmoid(out1s), bce) + self.semantic_loss(out1s,bce)  # + self.edge_looss(out1, semantic_gt)
        loss2 = self.iou(torch.sigmoid(out2s), bce) + self.semantic_loss(out2s, bce)  # + self.edge_looss(out2, semantic_gt)下次去了out3
        # #
        loss11 = self.mse(torch.sigmoid(out1s.float()),torch.sigmoid(out1.float()))#+self.Dice_loss(torch.sigmoid(out1s.float()),torch.sigmoid(out1.float()))
        loss21 = self.mse(torch.sigmoid(out2s.float()),torch.sigmoid(out2.float()))#+self.Dice_loss(torch.sigmoid(out2s.float()),torch.sigmoid(out2.float()))
        loss31 = self.mse(torch.sigmoid(out3s.float()),torch.sigmoid(out3.float()))#+self.Dice_loss(torch.sigmoid(out3s.float()),torch.sigmoid(out3.float()))# + self.SCL(torch.sigmoid(out3s.float()),torch.sigmoid(out3.float()))
        loss41 = self.KLD(torch.sigmoid(s1s.float()), torch.sigmoid(s1.float()))
        loss51 = self.KLD(torch.sigmoid(s2s.float()), torch.sigmoid(s2.float()))

        loss61 = self.Similarity(torch.sigmoid(self.ca(co0s, co1s,co2s, co3s)), torch.sigmoid(self.ca(co0, co1,co2, co3)))+self.SmoothL1Loss(torch.sigmoid(co0s),torch.sigmoid(co0))+self.SmoothL1Loss(torch.sigmoid(co1s),torch.sigmoid(co1))+self.SmoothL1Loss(torch.sigmoid(co2s), torch.sigmoid(co2))+self.SmoothL1Loss(torch.sigmoid(co3s),torch.sigmoid(co3))

        # print((self.ca(co0s, co1s,co2s, co3s)).shape, out2s.shape)


        # loss71 = self.KLD(torch.sigmoid(self.ca(co0s, co1s,co2s, co3s).float()),torch.sigmoid(ful.float()))


        # loss71 = self.smooth_loss(torch.sigmoid(co1s.float()), torch.sigmoid(co1.float()))

        # loss81 = self.smooth_loss(torch.sigmoid(co3s.float()), torch.sigmoid(co3.float()))
        # loss91 = self.smooth_loss(torch.sigmoid(co0s.float()), torch.sigmoid(co0.float()))
        # loss41 = self.KLD(torch.sigmoid(img.float()), torch.sigmoid(imgT.float(
        # loss51 = self.KLD(torch.sigmoid(dep.float()), torch.sigmoid(depT.float()))
        # # loss111 = self.KLD(torch.sigmoid(img_3.float()), torch.sigmoid(img_3T.float()))
        # # loss112 = self.KLD(torch.sigmoid(dep_3.float()), torch.sigmoid(dep_3T.float()))
        # loss61 = self.KLD(torch.sigmoid(d.float()), torch.sigmoid(dT.float()))+ self.smooth_loss(torch.sigmoid(d.float()), torch.sigmoid(dT.float()))
        # loss71 = self.KLD(torch.sigmoid(e.float()), torch.sigmoid(eT.float()))+self.smooth_loss(torch.sigmoid(e.float()), torch.sigmoid(eT.float()))
        # loss81 = self.semantic_loss(torch.sigmoid(self.ca(img_3.float(),dep_3.float())), torch.sigmoid(self.ca(img_3T.float(),dep_3T.float())))
        # loss91 = self.semantic_loss(torch.sigmoid(self.ca3(img.float(), dep.float())),torch.sigmoid(self.ca3(imgT.float(), depT.float())))
        #+self.iou(torch.sigmoid(img2),torch.sigmoid(img2T.float()))
        # loss71 = self.semantic_loss(torch.sigmoid(dep2), torch.sigmoid(dep2T.float()))+self.iou(torch.sigmoid(dep2), torch.sigmoid(dep2T.float()))
        # loss61 = self.smooth_loss(img2.float(), img2T.float())
        # loss71 = self.smooth_loss(dep2.float(), dep2T.float())
        loss = 1 * (loss1 + loss2) +loss41+loss51+loss61+loss21+loss11+loss31#+loss51+loss21+loss31#+loss71+loss81+loss91
        # print(loss1,loss21,loss31,loss3)
        return loss

