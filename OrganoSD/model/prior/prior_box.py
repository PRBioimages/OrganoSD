from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        # self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance']
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.prior_num = cfg['prior_num']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                cx1 = cx + 2 / f_k
                cy1 = cy + 2 / f_k
                cx2 = cx + 4 / f_k
                cy2 = cy + 4 / f_k
                cx3 = cx + 4 / f_k / 3
                cy3 = cy + 4 / f_k / 3
                s_k = self.min_sizes[k]/self.image_size
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))

                mean += [cx, cy, s_k, s_k]
                mean += [cx1, cy1, s_k, s_k]
                if self.prior_num[k] > 2:
                    mean += [cx, cy, s_k_prime, s_k_prime]
                    mean += [cx1, cy1, s_k_prime, s_k_prime]
                if self.prior_num[k] > 4:
                    mean += [cx2, cy2, s_k, s_k]
                    mean += [cx3, cy3, s_k, s_k]
                if self.prior_num[k] > 6:
                    mean += [cx2, cy2, s_k_prime, s_k_prime]
                    mean += [cx3, cy3, s_k_prime, s_k_prime]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
