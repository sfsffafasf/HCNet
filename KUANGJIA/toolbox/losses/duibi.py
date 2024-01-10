import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
# 定义蒸馏函数
class Dis(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, alpha):
        super(Dis, self).__init__()
        self.T = T
        self.mse = nn.MSELoss(reduce='mean')
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        # self.WeightedFocalLoss = WeightedFocalLoss()
    def forward(self, y_s, y_t):
        # student网络输出软化后结果
        p_s = F.log_softmax(y_s / self.T, dim=1)+1e-7
        # # teacher网络输出软化后结果
        p_t = F.softmax(y_t / self.T, dim=1)
        # 蒸馏损失采用的是KL散度损失函数
        kl_loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        y_s_attention_map = self.self_attention_map(y_s)
        y_t_attention_map = self.self_attention_map(y_t)
        mse = self.mse(y_s_attention_map, y_t_attention_map)
        loss = self.alpha*kl_loss + (1 - self.alpha)*mse
        # print(kl_loss,mse,WeightedFocalLoss)
        return loss.mean()

class Dis1(nn.Module):
    def __init__(self):
        super(Dis1, self).__init__()

        self.threshold = nn.Parameter(torch.Tensor(1))
    def forward(self, y_s,t):#使用前先进行softmax操作
        pool = nn.AdaptiveAvgPool2d(1)
        y_s = F.normalize(pool(y_s), dim=1)
        t = F.normalize(pool(t), dim=1)
        distances = F.pairwise_distance(t, y_s)
        distances = torch.clamp(distances, 0, 0.8)
        b = torch.clamp(self.threshold.data, min=0.4, max=0.60)
        # 将阈值限制在指定范围内
        margin = torch.full_like(distances, b.item())
        labels = torch.where(distances < margin, torch.tensor([-1]), torch.tensor([1])).flatten()
        # 根据距离阈值设置样本标签   distances为一个3,52,52的张量,无法于margin = 0.2比较大小
        loss_contrastive = torch.mean(labels * torch.pow(distances, 2) + (1 - labels) * torch.pow(torch.clamp(margin - distances, min=0.0), 2))
        return loss_contrastive
# labels * torch.pow(distances, 2) 计算了正样本（labels = 1）的损失，它将距离 distances 平方作为损失值。这样做的目的是让正样本之间的距离尽可能接近0，以增强它们的相似性。
# (1 - labels) * torch.pow(torch.clamp(margin - distances, min=0.0), 2) 计算了负样本（labels = -1）的损失，它使用了一个余项来定义损失。这个余项是通过将距离 distances 与阈值 margin 进行比较，并截断为非负值来计算的。这样做的目的是使负样本之间的距离大于阈值，增加它们的差异。
e = torch.randn(3, 128,52,52)
f = torch.randn(3, 128,52,52)
Dis1 = Dis1()
loss = Dis1(e,f)
print(loss,"Dis1")
# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma
#
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()


#



#
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
#
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()

# def gram_matrix(input):
#     """
#     计算输入特征图的Gram矩阵
#     参数:
#         input: 输入特征图, shape为 [batch_size, channels, height, width]
#     返回:
#         gram: Gram矩阵, shape为 [batch_size, channels, channels]
#     """
#     batch_size, channels, height, width = input.size()
#     features = input.view(batch_size * channels, height * width)
#     gram = torch.matmul(features, features.t())
#
#     # 标准化Gram矩阵（除以特征图的元素个数）
#     gram = gram.div(batch_size * channels * height * width)
#
#     return gram