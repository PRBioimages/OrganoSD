# -*- coding: utf-8 -*-
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import organ as cfg
from loss.endecode_nms import match, point_form, neg_select, log_sum_exp, match1, decode
from loss.detect_loss import MultiBoxLoss


class DetecLoss(nn.Module):
    def __init__(self, overlap_thresh, neg_truth):
        super(DetecLoss, self).__init__()
        self.threshold = overlap_thresh
        self.variance = cfg['variance']
        self.neg_pos_ratio = neg_truth
        self.num_classes = 2
        self.neg_truth = neg_truth

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        conf_neg = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx].data
            truths = truths[truths[:,2]-truths[:,0]>0]
            defaults = priors.data
            match1(self.threshold, truths, defaults, self.variance, loc_t, conf_t, idx)
            pos = conf_t[idx] > 0
            num_pos = pos.sum(dim=0, keepdim=True)
            neg_select(truths, priors, conf_neg, idx, self.neg_pos_ratio, num_pos)
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_t = []
        for a in range(len(targets)):
            pos_t.append((targets[a][:, -1]>0).sum().numpy())

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # pos_prior = np.zeros([num, 1500, 4], dtype=np.float32)
        # for i in range(num):
        #     pos_one = point_form(priors[pos_idx[i].squeeze(0)].view(-1, 4))
        #     pos_prior[i, :len(pos_one), :] = pos_one
        # pos_prior = torch.as_tensor(pos_prior)

        # batch_conf = conf_data.view(-1, self.num_classes)
        # loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # loss_c = loss_c.view(num, -1)
        # loss_c[pos] = 0  # filter out pos boxes for now
        # _, loss_idx = loss_c.sort(1, descending=True)
        # _, idx_rank = loss_idx.sort(1)
        # num_pos = pos.long().sum(1, keepdim=True)
        # num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # neg = idx_rank < num_neg.expand_as(idx_rank)

        neg = conf_neg > 0
        num_neg = neg.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # targets_weighted = conf_t[(pos + neg).gt(0)]
        # loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        conf_t = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, conf_t, size_average=False)
        # neg_idx1 = neg.unsqueeze(2).expand_as(loc_data)
        # neg_prior = np.zeros([num, 1500, 4], dtype=np.float32)
        # for i in range(num):
        #     neg_one = point_form(priors[neg_idx1[i].squeeze(0)].view(-1, 4))
        #     neg_prior[i, :len(neg_one), :] = neg_one
        # neg_prior = torch.as_tensor(neg_prior)

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        # print(f'-N={N}')
        # print(f'-numpos={num_pos.detach().numpy().T}---numneg={num_neg.detach().numpy().T}')
        return loss_l, loss_c, None, None
        # return loss_l, loss_c, pos_prior, neg_prior


class SegLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(SegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred, mask):
        # loss_s = F.smooth_l1_loss(pred.squeeze(1), mask.float(), size_average=True)
        # l1 = nn.L1Loss()
        # loss_s = l1(pred.squeeze(1), mask.float())

        bceloss = nn.BCELoss()
        loss_s = bceloss(pred.squeeze(1), mask.float())
        F_loss = loss_s

        # CE_loss = self.cross_entropy_loss(pred, mask.float())  # inputs可以是NxCxHxW，targets可以是NxHxW，会自动对其张量
        # pt = torch.exp(-CE_loss)
        # F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss




class ConsLoss(nn.Module):
    def __init__(self, conf_thresh, mask_thresh):
        super(ConsLoss, self).__init__()
        self.mask_thresh = mask_thresh
        self.conf_thresh = conf_thresh
        self.bceloss = nn.BCELoss()
        self.variance = cfg['variance']

    def forward(self, out_detect, out_seg):
        width = out_seg.shape[-1]
        mask_seg_ = out_seg.clone()

        loc_data_, conf_data_, priors = out_detect
        num = loc_data_.size(0)
        # IoU = 0
        consloss = 0
        l1 = nn.L1Loss()
        for idx in range(num):
            mask_detec = np.zeros((width, width))
            loc_data, conf_data = loc_data_[idx], conf_data_[idx]
            pos = conf_data[:, 1] > self.conf_thresh
            pos_idx = pos.unsqueeze(1).expand_as(loc_data)
            loc_p = decode(loc_data, priors, self.variance)
            loc_p = loc_p[pos_idx].view(-1, 4)
            loc_p = loc_p[loc_p[:, 2] - loc_p[:, 0] > 0]
            loc_p = loc_p[loc_p[:, 3] - loc_p[:, 1] > 0]
            loc_p = (loc_p.detach().numpy() * width).astype(np.uint16)
            loc_p[loc_p[:, :]>512] = 0
            for pt in loc_p:
                mask_detec[pt[1]:pt[3], pt[0]:pt[2]] = 1

            seg = mask_seg_[idx].squeeze(0)
            mask_seg = seg.clone().detach().numpy()
            mask_seg[mask_seg >= self.mask_thresh] = 1
            mask_seg[mask_seg < self.mask_thresh] = 0
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_seg.astype(np.uint8), connectivity=8)
            mask_seg[:, :] = 0
            for pt in stats:
                if pt[2]<width/2 and pt[3]<width/2:
                    mask_seg[pt[1]:pt[1] + pt[3], pt[0]:pt[0] + pt[2]] = 1
            image = torch.cat((seg * 255, torch.as_tensor(mask_seg * 255), torch.as_tensor(mask_detec * 255)), dim=1).detach().numpy()
            # cv2.imencode('.jpg', image)[1].tofile('/home/hryang/Detecseg/train_img/240409/Aseg_all.jpg')

            # itersect = (mask_detec * mask_seg).sum()
            # union = (mask_detec + mask_seg).astype(bool).astype(np.uint8).sum()
            # IoU += - itersect / union + 1
            # consloss += self.bceloss(torch.from_numpy(mask_detec).float(), torch.from_numpy(mask_seg).float())
            consloss += l1(torch.from_numpy(mask_detec).float(), torch.from_numpy(mask_seg).float())
        # loss_conf = IoU / num
        loss_cons = consloss / num
        return loss_cons


def total_loss(out_detect, out_seg, box, mask, overlap_thresh=0.2):
    loss_detec = DetecLoss(overlap_thresh, neg_truth = 2)
    loss_l, loss_c, pos_prior, neg_prior = loss_detec(out_detect, box)

    # loss_detec = MultiBoxLoss(overlap_thresh, prior_for_matching = True, neg_mining = True, neg_pos = 3, neg_overlap = 0.5, encode_target = True)
    # loss_l, loss_c, pos_prior, neg_prior = loss_detec(out_detect, box)

    loss_seg = SegLoss()
    loss_s = loss_seg(out_seg, mask)

    loss_cons = ConsLoss(0.5, 0.5)
    loss_cs = loss_cons(out_detect, out_seg)
    # loss_cs = torch.tensor(0)

    return loss_s, loss_c, loss_l, pos_prior, neg_prior, loss_cs


def seg_loss(out_seg, mask):
    loss_seg = SegLoss()
    loss_s = loss_seg(out_seg, mask)
    return loss_s