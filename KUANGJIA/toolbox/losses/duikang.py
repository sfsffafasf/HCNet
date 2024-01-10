import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch
import torch.nn as nn
import torch

CrossEntropyLoss2d = nn.CrossEntropyLoss

class CrossEntropyLoss2dLabelSmooth(nn.Module):
    def __init__(self, n_classes, epsilon=0.1, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (N x C x H x W)
        :param target: torch.tensor (N x H x W)
        :return: scalar
        """

        batch_size, _, height, width = output.size()

        scaled_output = output.permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes)
        target = target.view(-1)

        n_samples = scaled_output.size(0)
        target = target.unsqueeze(1).long()
        targets = torch.zeros(n_samples, self.n_classes, device=output.device).scatter_(0, target, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.n_classes

        loss = nn.CrossEntropyLoss(reduction=self.reduction)(scaled_output, target.squeeze())

        return loss

e = torch.randn(3, 1, 52, 52)
f = torch.randint(0, 2, (3, 52, 52))  # 目标值在0和1之间随机选择
loss_func = CrossEntropyLoss2dLabelSmooth(n_classes=2)  # 将n_classes设置为2，表示有两个类别
a = loss_func(e, f)
print(a)




# class HDDTBinaryLoss(nn.Module):
#     def __init__(self):
#         """
#         compute haudorff loss for binary segmentation
#         https://arxiv.org/pdf/1904.10030v1.pdf
#         """
#         super(HDDTBinaryLoss, self).__init__()
#
#     def forward(self, net_output, target):
#         """
#         net_output: (batch_size, 2, x,y,z)
#         target: ground truth, shape: (batch_size, 1, x,y,z)
#         """
#         net_output = softmax_helper(net_output)
#         pc = net_output[:, 1, ...].type(torch.float32)
#         gt = target[:, 0, ...].type(torch.float32)
#         with torch.no_grad():
#             pc_dist = compute_edts_forhdloss(pc.cpu().numpy() > 0.5)
#             gt_dist = compute_edts_forhdloss(gt.cpu().numpy() > 0.5)
#         # print('pc_dist.shape: ', pc_dist.shape)
#
#         pred_error = (gt - pc) ** 2
#         dist = pc_dist ** 2 + gt_dist ** 2  # alpha=2 in eq(8)
#
#         dist = torch.from_numpy(dist)
#         if dist.device != pred_error.device:
#             dist = dist.to(pred_error.device).type(torch.float32)
#
#         multipled = torch.einsum("bxyz,bxyz->bxyz", pred_error, dist)
#         hd_loss = multipled.mean()
#
#         return hd_loss
