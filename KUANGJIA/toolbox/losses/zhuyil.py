import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss

# 定义蒸馏函数
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, alpha,q):
        super(DistillKL, self).__init__()
        self.T = T

        self.pearson_distance_loss = pearson_distance_loss
        self.alpha = alpha
        self.quantile_loss = quantile_loss
        self.q = q
    def forward(self, y_s, y_t):
        # student网络输出软化后结果
        y_s = torch.nn.functional.normalize(y_s)
        y_s = nn.LeakyReLU()(y_s)
        y_t = torch.nn.functional.normalize(y_t)
        y_t = nn.LeakyReLU()(y_t)
        # log_softmax与softmax没有本质的区别，只不过log_softmax会得到一个正值的loss结果。
        p_s = F.log_softmax(y_s / self.T, dim=1)
        # # teacher网络输出软化后结果
        p_t = F.softmax(y_t / self.T, dim=1)
        # 蒸馏损失采用的是KL散度损失函数
        kl_loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        # y_s_attention_map = self.DynamicSelfAttention(y_s,y_t)

        scores1 = torch.matmul(y_s, y_s.transpose(-2, -1))  # 形状：[batch_size, num_heads, seq_len, seq_len]
        # 归一化注意力权重
        attention_weights1 = F.softmax(scores1, dim=-1)
        # 计算自注意力图
        scores2 = torch.matmul(y_t, y_t.transpose(-2, -1))  # 形状：[batch_size, num_heads, seq_len, seq_len]
        # 归一化注意力权重
        attention_weights2 = F.softmax(scores2, dim=-1)
        quantile_loss= self.quantile_loss(y_s, y_t,self.q)

        loss = self.alpha*kl_loss + (1 - self.alpha)*quantile_loss

        return loss.mean()


def pearson_distance_loss(y_pred, y_true):
    # 计算均值

    y_pred_mean = torch.mean(y_pred)
    y_true_mean = torch.mean(y_true)
    # 中心化
    y_pred_centered = y_pred - y_pred_mean
    y_true_centered = y_true - y_true_mean
    # 计算标准差
    y_pred_std = torch.std(y_pred)
    y_true_std = torch.std(y_true)
    # 计算协方差
    covariance = torch.mean(y_pred_centered * y_true_centered)
    # 计算皮尔逊相关系数
    pearson_corr = covariance / (y_pred_std * y_true_std)
    # 将皮尔逊相关系数转化为损失
    distance_loss = mse_loss(pearson_corr, torch.zeros_like(pearson_corr))

    return distance_loss
def quantile_loss(y_true, y_pred, q):
    """
    分位数损失函数
    """

    error = y_true - y_pred
    loss = torch.max(q * error, (q - 1) * error)
#     return loss.mean()
# e = torch.randn(3, 128, 52, 52)
# f = torch.randn(3, 128, 52, 52)
# # contrastive_loss = DistillKLml3(2,128,1)
# DistillKL = DistillKL(2,0.02,0.7)
# DistillKL = DistillKL(e,f)
# print(DistillKL)

#
# class DynamicSelfAttention(nn.Module):
#     def __init__(self, embedding_dim, num_heads):
#         super(DynamicSelfAttention, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#
#         self.query = nn.Linear(embedding_dim, embedding_dim)
#         self.key = nn.Linear(embedding_dim, embedding_dim)
#         self.value = nn.Linear(embedding_dim, embedding_dim)
#         self.proj = nn.Linear(embedding_dim, embedding_dim)
#
#     def forward(self, input_features, target_feature):
#         # 输入特征形状：[batch_size, seq_len, embedding_dim]
#         batch_size, seq_len, embedding_dim = input_features.size()
#
#         # 使用线性变换获取查询、键和值
#         query = self.query(input_features).view(batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads).transpose(1, 2)
#         key = self.key(target_feature).view(batch_size, 1, self.num_heads, embedding_dim // self.num_heads).transpose(1, 2)
#         value = self.value(target_feature).view(batch_size, 1, self.num_heads, embedding_dim // self.num_heads).transpose(1, 2)
#
#         # 计算注意力分数
#         scores = torch.matmul(query, key.transpose(-2, -1))  # 形状：[batch_size, num_heads, seq_len, 1]
#         # 归一化注意力权重
#         attention_weights = torch.softmax(scores / (embedding_dim ** 0.5), dim=-1)
#         # 计算加权和表示
#         weighted_sum = torch.matmul(attention_weights, value)  # 形状：[batch_size, num_heads, seq_len, embedding_dim // num_heads]
#         # 进行投影变换
#         weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
#         self_attention_map = self.proj(weighted_sum)
#
#         return self_attention_map

# class SpatialGCN(nn.Module):
#     def __init__(self, plane):
#         super(SpatialGCN, self).__init__()
#         inter_plane = plane // 2
#         self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
#         self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
#         self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
#
#         self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
#         self.bn_wg = nn.BatchNorm1d(inter_plane)
#         self.softmax = nn.Softmax(dim=2)
#
#         self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
#                                  nn.BatchNorm2d(plane))
#
#     def forward(self, x):
#         # b, c, h, w = x.size()
#         node_k = self.node_k(x)
#         node_v = self.node_v(x)
#         node_q = self.node_q(x)
#         b, c, h, w = node_k.size()
#         node_k = node_k.view(b, c, -1).permute(0, 2, 1)
#         node_q = node_q.view(b, c, -1)
#         node_v = node_v.view(b, c, -1).permute(0, 2, 1)
#         # A = k * q
#         # AV = k * q * v
#         # AVW = k *(q *v) * w
#         AV = torch.bmm(node_q, node_v)
#         AV = self.softmax(AV)
#         MV = torch.bmm(node_k, AV)
#
#         return AV,MV
# e = torch.randn(3, 128, 52, 52)
# SpatialGCN = SpatialGCN(128)
# SpatialGCN = SpatialGCN(e)
# print(SpatialGCN.shape)
class CrossEntropyLoss2dLabelSmooth():
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector可以在前向传播方法中使用它来计算模型的输出与目标之间的损失
    """

    def __init__(self, weight=None, ignore_label=255, epsilon=0.1, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.nll_loss = nn.PoissonNLLLoss(reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        output = output.permute((0, 2, 3, 1)).contiguous().view(-1, output.size(1))
        target = target.view(-1)
        n_classes = output.size(1)
        # batchsize, num_class = input.size()
        # log_probs = F.log_softmax(inputs, dim=1)
        targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes

        return self.nll_loss(output, targets)


def dice_loss(target, predictive, ep= 1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss