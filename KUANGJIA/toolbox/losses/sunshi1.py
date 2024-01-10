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
        self.self_attention_map = self_attention_map
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

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # print(targets.shape)
        targets = targets.squeeze(3).squeeze(2).type(torch.long)

        at = self.alpha.gather(0, targets.data.view(-1).long())
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

e = torch.randn(3, 2, 1, 1)
f = torch.randn(3, 2, 1, 1)
loss = WeightedFocalLoss()
loss_value = loss(e, f)
print(loss_value)




class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.6, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# e = torch.randn(3, 2,26,26)
# f = torch.randn(3, 2,26,26)
# loss = BCEFocalLoss()
# loss = loss(e,f)
# print(loss)
# DistillKL = Dis(2,0.7)
# Dis = Dis(e,f)
# print(Dis)


# 将皮尔逊损失用于特征注意力图，而将分位数损失和KL损失用于特征本身是可行的，因为它们在不同的层面上对任务有所贡献。
#
#     皮尔逊损失用于特征注意力图：皮尔逊相关系数在这里用于衡量特征注意力图与目标之间的线性相关性。通过最小化皮尔逊损失，模型可以学习到如何将特征注意力图与目标正相关或负相关。这对于理解哪些特征对目标更重要是有帮助的。
#
#     分位数损失用于特征本身：分位数损失函数在这里用于衡量预测值与真实值之间的分位数偏差。通过最小化分位数损失，模型可以更好地拟合数据集的分位数特征，例如中位数或高分位数。这对于一些特定的任务，如异常检测或风险评估，可能更有意义。
#
#     KL损失用于特征本身：KL散度（Kullback-Leibler divergence）常用于衡量两个概率分布之间的差异。将KL损失用于特征本身可以帮助模型更准确地学习到数据的分布特征。通过最小化KL损失，模型可以更好地拟合数据的分布，并提高对数据的生成能力。
#
# 总之，将皮尔逊损失用于特征注意力图，而将分位数损失和KL损失用于特征本身是有意义的，因为它们在不同的层面上对任务有贡献。但是，仍然需要根据具体任务和数据集来评估和调整这些损失函数，以确保它们能够合理地衡量和满足需求。
# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()
