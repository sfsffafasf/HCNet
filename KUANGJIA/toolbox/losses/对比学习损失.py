import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].批次和每个样本特征的维度。
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        features = features.squeeze(1).view(batch_size,-1).float()
        labels = (self.pool(labels)).squeeze().view(batch_size).float()
        # features = (features.view(batch_size,-1,1)).squeeze().float()
        # labels = (labels.view(batch_size,-1,1)).squeeze().float()
        features = F.normalize(features.float(), p=2, dim=1)#p=2 表示使用的范数是 L2 范数，也就是欧氏距离范数。dim=1 表示在第一个维度上进行归一化操作，即对每个样本的特征向量进行归一化。
#归一化的目的是为了将特征向量缩放到单位长度，从而消除不同样本之间的尺度差异。这对于对比学习任务非常重要，因为在计算样本之间的距离或相似度时，尺度的影响应该被消除。
        # batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度，较低的温度会使相似度更加尖锐，而较高的温度则会使相似度更加平滑
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        # 这个一维张量的长度为 batch_size，对应特征矩阵中每个样本的数量。
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) \
                      + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)#它会将输入张量中的每个元素 x，计算 log(x) 并返回一个与输入张量相同形状的张量
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        #是检查张量 log_probs 中的每个元素是否为 NaN
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:#决定最终的输出是否要乘以温度系数
            loss *= self.temperature
        loss = loss.mean()
        return loss

a = torch.randn(3, 1, 416, 416)
b = torch.randn(3, 1, 52, 52)
c = torch.randn(3, 1, 52, 52)
d = torch.randn(3, 1, 416, 416)
e = torch.randn(3, 1, 52, 52)
f = torch.randn(3, 1, 52, 52)
# contrastive_loss = DistillKLml3(2,128,1)
# loss = contrastive_loss(e,f,a)
# Similarity = Similarity()
# loss= Similarity(a,d)
# print(loss)
SupConLoss = SupConLoss(0.4)
SupConLoss = SupConLoss(e,f)
print(SupConLoss)

# Lovász - Softmax
# Loss：Lovász - Softmax
# Loss
# 是一种基于最小化平均交叉点差异的非参数化损失函数。它能够处理类别不平衡和多个标签的情况，并且在边界细节和小目标的分割上表现出色。
#
# Dice
# Loss：Dice
# Loss
# 使用
# Dice
# 系数作为相似性度量，计算预测结果与真实标签之间的重叠度。它对于类别不平衡的数据集和像素级别的语义分割任务效果较好。
#
# Focal
# Loss：Focal
# Loss
# 是针对类别不平衡问题设计的一种损失函数。它通过降低易分类样本的权重来解决类别不平衡问题，同时聚焦于难分类的样本，有助于提高模型对于边界像素的分割准确性。
#
# Hausdorff
# Loss：Hausdorff
# Loss
# 是一种基于
# Hausdorff
# 距离的损失函数，用于度量预测结果和真实标签之间的距离。它能够鲁棒地处理分割结果的稀疏性和模糊性，对于不规则目标的分割效果较好。
