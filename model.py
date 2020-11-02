import torch
import torch.nn as nn
import numpy as np


class RealNVP(nn.Module):
    def __init__(self, mask, base_dist):
        super(RealNVP, self).__init__()

        self.mask = mask
        self.base_dist = base_dist

        self.net_t = nn.Sequential(nn.Linear(mask.shape[0], 1000), 
                                  nn.ReLU(),
                                  nn.Linear(1000, 28 * 28 - mask.shape[0]))
        self.net_s = nn.Sequential(nn.Linear(mask.shape[0], 1000), 
                                  nn.ReLU(),
                                  nn.Linear(1000, 28 * 28 - mask.shape[0]))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x1 = x[:, self.mask]
        x2 = x[:, np.setdiff1d(np.arange(x.shape[1]), self.mask)]
        t = self.net_t(x1)
        s = self.net_s(x1)
        y1 = x1
        y2 = torch.exp(s) * x2 + t
        y = torch.cat((y1, y2), dim=1)
        log_det_j = torch.sum(s)
        return y, log_det_j
    
    def inverse(self, y):
        y = y.reshape(y.shape[0], -1)
        y1 = y[:, self.mask]
        y2 = y[:, np.setdiff1d(np.arange(y.shape[1]), self.mask)]
        t = self.net_t(y1)
        s = self.net_s(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat((x1, x2), dim=1)
        log_det_inv_j = torch.sum(-s)
        return x, log_det_inv_j
    
    def log_prob(self, y):
        y = y.reshape(y.shape[0], -1)
        x, log_det_inv_j = self.inverse(y)
        log_p_x = self.base_dist.log_prob(x)
        # print(log_p_x, log_det_inv_j)
        return log_p_x + log_det_inv_j
