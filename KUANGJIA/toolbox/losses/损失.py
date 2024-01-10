import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss

# 定义蒸馏函数
class Dis(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, alpha,in_dim):
        super(Dis, self).__init__()
        self.T = T
        self.mse = nn.MSELoss(reduce='mean')
        self.alpha = alpha
        self.conv1_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv1_dep = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_ful2 = nn.Conv2d(2 * in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.out_dim = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, y_s, y_t):

        p_s = F.log_softmax(y_s / self.T, dim=1)+1e-7
        # # teacher网络输出软化后结果
        p_t = F.softmax(y_t / self.T, dim=1)
        # 蒸馏损失采用的是KL散度损失函数
        kl_loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        y_s_attention_map = self.self_attention_map(y_s)
        y_t_attention_map = self.self_attention_map(y_t)
        x_rgb = self.conv1_rgb(y_s_attention_map)
        x_dep = self.conv1_dep(y_t_attention_map)
        mm_cat2 = torch.cat([x_dep, x_rgb], dim=1)
        cim_out = self.out_dim(self.layer_ful2(mm_cat2))
        y_s_attention_map = y_s_attention_map * cim_out
        y_t_attention_map = y_t_attention_map * cim_out
        mse = self.mse(y_s_attention_map, y_t_attention_map)
        loss = self.alpha*kl_loss + (1 - self.alpha)*mse
        # print(kl_loss,mse,WeightedFocalLoss)
        # 如果你需要将这个概率分布再进行额外的处理或计算，可以直接使用这个已经归一化的张量。但如果你需要将概率分布转换为具体的类别标签，可以使用
        # argmax
        # 函数找到最大概率对应的类别索引。例如，对于每个样本，可以使用
        # torch.argmax(tensor, dim=1)
        # 来获取最大概率对应的类别索引。



        return loss.mean()