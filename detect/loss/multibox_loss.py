# -*- coding: utf-8 -*-
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import organ512 as cfg
from loss.endecode_nms import match,match_ini, point_form, neg_select, log_sum_exp, match1, decode
from loss.detect_loss import MultiBoxLoss


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][targets[idx][:, 2] > targets[idx][:, 0]].data
            labels = torch.ones(truths.size(0), 1)
            defaults = priors.data
            match_ini(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(0, conf_t.view(-1, 1))
        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # targets_weighted = conf_t[(pos+neg).gt(0)].view(-1, self.num_classes).float()
        # loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1)
        targets_weighted = conf_t[(pos + neg).gt(0)].float()
        loss_c = F.binary_cross_entropy_with_logits(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class DetecLoss(nn.Module):
    def __init__(self, overlap_thresh, neg_pos_ratio):
        super(DetecLoss, self).__init__()
        self.threshold = overlap_thresh
        self.variance = cfg['variance']
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = 2

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        device = loc_data.device
        num, num_priors = loc_data.size(0), priors.size(0)
        # 初始化目标张量（与预测张量同设备）
        loc_t = torch.zeros(num, num_priors, 4, device=device)
        conf_t = torch.zeros(num, num_priors, dtype=torch.long, device=device)
        neg_mask = torch.zeros(num, num_priors, dtype=torch.bool, device=device)
        for idx in range(num):
            truths = targets[idx][targets[idx][:, 2] > targets[idx][:, 0]]  # 过滤无效框
            defaults = priors.data
            # 假设 match1 正确实现匹配逻辑，填充 loc_t 和 conf_t
            match1(self.threshold, truths, defaults, self.variance, loc_t, conf_t, idx)
            # 假设 neg_select 正确筛选负样本，填充 neg_mask
            neg_select(truths, defaults, neg_mask, idx, self.neg_pos_ratio, (conf_t[idx] > 0).sum())
        # 计算位置损失（仅正样本）
        pos_mask = conf_t > 0
        pos_loc_pred = loc_data[pos_mask].view(-1, 4)
        pos_loc_target = loc_t[pos_mask].view(-1, 4)
        loss_l = F.smooth_l1_loss(pos_loc_pred, pos_loc_target, reduction='sum')
        # 计算分类损失（正样本 + 负样本）
        combined_mask = pos_mask | neg_mask
        conf_p = conf_data[combined_mask].view(-1, self.num_classes)
        conf_t_masked = conf_t[combined_mask]
        loss_c = F.cross_entropy(conf_p, conf_t_masked, reduction='sum')
        # 归一化损失（避免除零）
        N = max(pos_mask.sum().item(), 1e-8)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c, None, None


class SegLoss_ini(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(SegLoss_ini, self).__init__()
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

class SegLoss_focal(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(SegLoss_focal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, pred, mask):
        b, H, W = mask.shape
        pred = pred.view(-1).float()
        mask = mask.view(-1).float()
        CEloss = F.cross_entropy(pred, mask, reduction='none')
        # 计算焦点损失
        pt = torch.exp(-CEloss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CEloss
        if self.reduction == 'mean':
            return focal_loss / H / W
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss



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
            pos = conf_data[:, 0] > self.conf_thresh
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
            cv2.imencode('.jpg', image)[1].tofile('/home/hryang/Detecseg/train_img/240409/Aseg_all.jpg')

            # itersect = (mask_detec * mask_seg).sum()
            # union = (mask_detec + mask_seg).astype(bool).astype(np.uint8).sum()
            # IoU += - itersect / union + 1
            # consloss += self.bceloss(torch.from_numpy(mask_detec).float(), torch.from_numpy(mask_seg).float())
            consloss += l1(torch.from_numpy(mask_detec).float(), torch.from_numpy(mask_seg).float())
        # loss_conf = IoU / num
        loss_cons = consloss / num
        return loss_cons


def total_loss(out_detect, out_seg, box, mask, overlap_thresh=0.2):
    # loss_detec = DetecLoss(overlap_thresh, neg_pos_ratio = 2)
    # loss_l, loss_c, pos_prior, neg_prior = loss_detec(out_detect, box)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 1, 0.5,
                             False, False)
    loss_l, loss_c = criterion(out_detect, box)

    loss_seg = SegLoss_focal(alpha=0.75, gamma=10, reduction='mean')
    loss_s = loss_seg(out_seg, mask)

    loss_cons = ConsLoss(0.5, 0.5)
    loss_cs = loss_cons(out_detect, out_seg)
    # loss_cs = torch.tensor(0)

    return loss_s, loss_c, loss_l, None, None, loss_cs


def seg_loss(out_seg, mask):
    loss_seg = SegLoss_focal()
    loss_s = loss_seg(out_seg, mask)
    return loss_s