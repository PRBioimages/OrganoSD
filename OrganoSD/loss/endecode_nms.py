from numpy import random
import torch
import numpy as np

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]



def neg_select(truths, priors, conf_neg, idx, neg_pos, num_pos):
    # width_prior = list(priors[:, 2].detach().numpy())
    # N = len(width_prior)
    # dic = []
    # ratio = []
    # for i in width_prior:
    #     if i in dic:
    #         None
    #     else:
    #         dic.append(i)
    # for i in dic:
    #     x = width_prior.count(i)/N
    #     ratio.append(x)
    # ratio = np.array(ratio)
    # print(dic)
    dic = [0.05859375, 0.10546875, 0.12917231, 0.15234375, 0.17506364, 0.19921875, 0.24158822, 0.265625, 0.28811076]
    ratio = np.array([0.481, 0.241, 0.120, 0.06, 0.060, 0.015, 0.015, 0.004, 0.004])
    total_num = int(num_pos * neg_pos)
    num = np.ceil(ratio * total_num).astype(np.uint16)
    neg = np.zeros((priors.size(0), 1))

    priors_point = point_form(priors)
    inter = intersect(truths, priors_point)
    inter_sum = sum(inter, 0)
    index1 = inter_sum == 0

    for k, n in zip(dic, num):
        index2 = priors[:, 2] == k
        index = index1 * index2
        index = random_n(index, n)
        neg[index, :] = 1
    conf_neg[idx] = torch.from_numpy(neg).squeeze(1)


def match(threshold, truths, priors, loc_t, conf_t, idx):
    # jaccard index
    priors = point_form(priors)
    overlaps = jaccard(truths, priors)
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = best_prior(overlaps)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 1)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    loc = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = np.ones((len(best_truth_idx), 1))          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    # loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = torch.from_numpy(conf).squeeze(1)  # [num_priors] top class label for each prior



def best_prior(overlaps):
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    diction = []
    for k, v in enumerate(best_prior_idx):
        if v in diction:
            overlaps[k,v] = 0
        else:
            diction.append(v)
    if len(diction)==len(best_prior_idx):
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        return best_prior_overlap, best_prior_idx
    else:
        return best_prior(overlaps)


def match1(threshold, truths, priors, variances, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    flag = best_truth_overlap[0].clone()
    best_truth_overlap.index_fill_(0, best_prior_idx, 1)  # ensure best prior
    best_truth_overlap[0] = flag
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]           # Shape: [num_priors,4]
    conf = np.ones((len(best_truth_idx), 1))   # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0   # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc                           # [num_priors,4] encoded offsets to learn
    conf_t[idx] = torch.from_numpy(conf).squeeze(1)  # [num_priors] top class label for each prior


def random_n(index, n):
    idx = np.arange(0, index.size(0), dtype = np.uint16)
    idx = idx[index]
    while idx.size > n:
        i = random.randint(idx.size)
        index[idx[i]] = False
        idx = np.delete(idx, i)

    idx_ = np.arange(0, index.size(0), dtype=np.uint16)
    idx_ = idx_[index]
    return index



def encode(matched, priors, variances):
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    boxes = torch.cat((
    priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
