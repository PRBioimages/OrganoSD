from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss, SmoothL1Loss
from reconstruction.ResAE.utils import *
import torch.nn.functional as F
from torch import nn
import torch

from torch.autograd import Variable

#结合了 Margin Loss 和 Focal Loss 的损失函数，用于处理二进制分类问题。这种损失函数旨在改善模型在难以分类的样本上的性能
# ICME.2019Skeleton-BasedActionRecognitionwithSynchronousLocalandNon-LocalSpatio-TemporalLearningandFrequencyAttention.pdf
# Soft-margin focal loss
def criterion_margin_focal_binary_cross_entropy(weight_pos=2, gamma=2):
    def _criterion_margin_focal_binary_cross_entropy(logit, truth):
        # weight_pos=2
        #注释掉即将其设置为1
        #weight_pos: 用于调整正类别的权重，默认值为2。这意味着正类别的损失将被赋予更高的权重，以便更好地处理正类别样本。
        weight_neg=1
        # gamma=2
        #Focal Loss 中的焦点参数
        margin=0.2
        em = np.exp(margin)
        #Focal Loss 的核心思想是对于容易分类的样本（概率接近于 0 或 1），降低它们的损失贡献，使模型更加关注难以分类的样本。
        # 通过调整 margin 和 em，可以影响 Focal Loss 的表现，以适应不同的任务和数据集。

        #计算 logit（模型的输出）和 truth（真实标签）的相关信息
        logit = logit.view(-1)
        truth = truth.view(-1)
        log_pos = -F.logsigmoid( logit)
        log_neg = -F.logsigmoid(-logit)

        log_prob = truth*log_pos + (1-truth)*log_neg
        prob = torch.exp(-log_prob)
        #计算损失的 margin 部分，这部分用于增强模型在难以分类的样本上的损失。
        # 它使用了 Margin Loss 中的 margin 参数
        margin = torch.log(em +(1-em)*prob)

        #计算权重部分，这部分根据样本的真实标签调整了权重，以便更好地处理正类别样本
        weight = truth*weight_pos + (1-truth)*weight_neg
        loss = margin + weight*(1 - prob) ** gamma * log_prob

        loss = loss.mean()
        return loss
    return _criterion_margin_focal_binary_cross_entropy


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma
#
#     def forward(self, input, target):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), input.size()))
#
#         max_val = (-input).clamp(min=0)
#         loss = input - input * target + max_val + \
#                ((-max_val).exp() + (-input - max_val).exp()).log()
#
#         invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss
#
#         return loss.sum(dim=1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        if not (targets.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(targets.size(), logits.size()))

        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1 - p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p) ** self.gamma)
        # loss = 19 * loss.mean()
        return loss.mean()

#标签平滑是一种正则化技术，用于改善深度学习模型的泛化性能，特别是在面对类别不平衡和过拟合问题时非常有用。
class LabelSmoothingCrossEntropy(nn.Module):
    '''
    copy from fastai

    '''
    def __init__(self, eps:float=0.1, reduction='mean'):
        super().__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

#创建交叉熵（Cross Entropy）损失函数的工厂函数。
# 它接受一个权重参数 weight，可以根据需要为不同类别分配不同的权重
def ce(weight=None):
    print('[ √ ] Using CE loss, weight is {}'.format(weight))
    ce = CrossEntropyLoss(weight=torch.tensor(weight).cuda())

    def _ce_loss(output, truth):
        return ce(output, truth)
    return _ce_loss

#带有 Online Hard Example Mining (OHEM) 的交叉熵损失函数
def ce_oheb():
    def _ce_loss(output, truth):
        ce = CrossEntropyLoss(reduction='none')
        r = ce(output, truth).view(-1)
        return r.topk(output.shape[0] - 2, largest=False)[0].mean()
    return _ce_loss

#工厂函数，用于创建针对不同问题和超参数设置的 Focal Loss 损失函数
def focal_loss(gamma=2):
    def _focal_loss(output, truth):
        focal = FocalLoss(gamma=gamma)
        return focal(output, truth)
    return _focal_loss

#工厂函数，用于创建均方误差（Mean Squared Error，MSE）损失函数
def mse(flag=True):
    if flag:
        print('[ √ ] Using loss MSE')
    def _mse_loss(output, truth):
        mse = MSELoss()
        return mse(output, truth)
    return _mse_loss

#创建二元交叉熵损失（Binary Cross Entropy，BCE）损失函数
def bce(reduction='mean'):
    # print('[ √ ] Using loss BCE')
    def _bce_loss(output, truth):
        bce = BCEWithLogitsLoss(reduction=reduction)
        return bce(output, truth)
    return _bce_loss

#带有 Online Hard Example Mining (OHEM) 的二元交叉熵损失函数
#ratio 参数可以指定保留的困难样本比例
def bce_ohem(ratio=0.5):
    def _bce_loss(output, truth):
        bce = BCEWithLogitsLoss(reduction='none')
        r = bce(output, truth).view(-1)
        return r.topk(int(r.shape[0] * ratio))[0].mean()
    return _bce_loss

#复合损失函数 _bce_mse_loss，它同时计算二元交叉熵和均方误差
def bce_mse(ratio=(0.5, 0.5)):
    print('SpecialLoss: BCE and SUM(1) MSE')
    def _bce_mse_loss(output, truth):
        bce = BCEWithLogitsLoss()
        mse = MSELoss()
        b = bce(output, truth)
        m = mse(torch.sigmoid(output).sum(1), truth.sum(1))
        return ratio[0] * b + ratio[1] * m, b, m
    return _bce_mse_loss



def sl1(k):
    # print('[ √ ] Using loss MSE')
    def _sl1_loss(output, truth):
        _sl1 = SmoothL1Loss()
        return _sl1(k * output, k * truth)
    return _sl1_loss



def mae():
    # print('[ √ ] Using loss MSE')
    def _mae_loss(output, truth):
        mae = L1Loss()
        return mae(output, truth)
    return _mae_loss


# def label_smooth_ce(eps=0.1):
#     def _ce_loss(output, gt):
#         ce = LabelSmoothingCrossEntropy(eps=eps)
#         return ce(output, gt)
#     return _ce_loss

#标签平滑的交叉熵损失函数
def label_smooth_ce(eps=0.1, reduction='mean'):
    def _ce_loss(output, gt):
        ce = LabelSmoothingCrossEntropy(eps=eps, reduction=reduction)
        return ce(torch.cat([output, 1-output], 1), gt.view(-1).long())
    return _ce_loss

#创建带有标签平滑和Online Hard Example Mining (OHEM) 的损失函数
#eps 来调整标签平滑的强度，pa 来指定保留的困难样本比例
def label_smooth_ce_ohem(eps=0.1, pa=0.5, bs=64):

    def _ce_loss(output, gt):
        ce = LabelSmoothingCrossEntropy(eps=eps, reduction='none')
        k = min(int(pa * bs), output.size(0))
        return ce(output, gt).topk(k=k)
    return _ce_loss


