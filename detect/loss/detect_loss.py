import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import organ as cfg
from loss.endecode_nms import match1, point_form, log_sum_exp


class MultiBoxLoss(nn.Module):
    def __init__(self, overlap_thresh, prior_for_matching, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.threshold = overlap_thresh
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)# batch_size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0)) # 先验框个数
        # num_classes = self.num_classes #类别数

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx].data
            truths = truths[truths[:, 0] + truths[:, 1] + truths[:, 2] + truths[:, 3] > 0]
            defaults = priors.data
            match1(self.threshold, truths, defaults, self.variance, loc_t, conf_t, idx)
        # loc_t = Variable(loc_t, requires_grad=False)
        # conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4) #预测的正样本box信息
        loc_t = loc_t[pos_idx].view(-1, 4) #真实的正样本box信息
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) #Smooth L1 损失

        pos_prior = np.zeros([num, 420, 4], dtype=np.float32)
        for i in range(num):
            pos_one = point_form(priors[pos_idx[i].squeeze(0)].view(-1, 4))
            pos_prior[i, :len(pos_one), :] = pos_one
        pos_prior = torch.as_tensor(pos_prior)

        '''
        Target；
            下面进行hard negative mining
        '''
        batch_conf = conf_data.view(-1, 1)
        # 使用logsoftmax，计算置信度,shape[b*M, 1]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # 把正样本排除，剩下的就全是负样本，可以进行抽样
        loss_c = loss_c.view(num, -1)# shape[b, M]
        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
         # 抽取负样本
        # 每个batch中正样本的数目，shape[b,1]
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 抽取前top_k个负样本，shape[b, M]
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # shape[b,M] --> shape[b,M,num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 计算conf交叉熵
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # 正样本个数
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c