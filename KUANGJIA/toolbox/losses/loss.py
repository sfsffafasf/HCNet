import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from toolbox.losses.lovasz_losses import lovasz_softmax
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import NLLLoss2d

__all__ = ["MscCrossEntropyLoss", "CrossEntropyLoss2d", "CrossEntropyLoss2dLabelSmooth",
           "FocalLoss2d", "LDAMLoss", "ProbOhemCrossEntropy2d",
           "LovaszSoftmax"]


class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                        ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


CrossEntropyLoss2d = nn.CrossEntropyLoss


class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
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


"""
https://arxiv.org/abs/1708.02002
# Credit to https://github.com/clcarwin/focal_loss_pytorch
"""


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()

        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()

        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        print(output.shape, target.shape)
        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


"""
https://arxiv.org/pdf/1906.07413.pdf
"""


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


# Adapted from OCNet Repository (https://github.com/PkuRainBow/OCNet)
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                 weight=weight,
                                                 ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                 ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)  #
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # print('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


# ==========================================================================================================================
# ==========================================================================================================================


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


def dice_loss(pred, mask):
    # mask = F.sigmoid(mask)
    # pred = F.sigmoid(pred)
    # mask = F.softmax(mask, dim=1)
    # pred = F.softmax(pred, dim=1)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss
