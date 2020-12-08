import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False))

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        skipcon = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = x + skipcon

        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, kernel_size, padding):
        super(ResNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(2 * in_channels, mid_channels, kernel_size, padding, bias=True))
        self.skip1 = nn.utils.weight_norm(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True))

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([nn.utils.weight_norm(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True))
                                    for _ in range(num_blocks)])

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True))

    def forward(self, x):
        x = self.bn1(x)
        x = torch.cat((x, -x), dim=1)
        x = self.conv1(F.relu(x))
        x_skip = self.skip1(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.bn2(x_skip)
        x = F.relu(x)
        x = self.conv2(x)

        return x

class RealNVP(nn.Module):
    def __init__(self, mask, base_dist):
        super(RealNVP, self).__init__()

        self.mask = mask
        self.base_dist = base_dist

        self.net_t = nn.Sequential(nn.Linear(mask.shape[0], 1000), 
                                  nn.ReLU(),
                                  nn.Linear(1000, mask.shape[0]))
        self.net_s = nn.Sequential(nn.Linear(mask.shape[0], 1000), 
                                  nn.ReLU(),
                                  nn.Linear(1000, mask.shape[0]),
                                  nn.Tanh())

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        # x1 = x[:, self.mask]
        # x2 = x[:, np.setdiff1d(np.arange(x.shape[1]), self.mask)]
        # t = self.net_t(x1)
        # s = self.net_s(x1)
        # y1 = x1
        # y2 = torch.exp(s) * x2 + t
        # y = torch.cat((y1, y2), dim=1)
        # log_det_j = torch.sum(s)
        t = self.net_t(self.mask * x)
        s = self.net_s(self.mask * x)
        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_j = torch.sum(self.mask * s)
        return y, log_det_j

    def inverse(self, y):
        y = y.reshape(y.shape[0], -1)
        # y1 = y[:, self.mask]
        # y2 = y[:, np.setdiff1d(np.arange(y.shape[1]), self.mask)]
        # t = self.net_t(y1)
        # s = self.net_s(y1)
        # x1 = y1
        # x2 = (y2 - t) * torch.exp(-s)
        # x = torch.cat((x1, x2), dim=1)
        # log_det_inv_j = torch.sum(-s)
        t = self.net_t(self.mask * y)
        s = self.net_s(self.mask * y)
        x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det_inv_j = torch.sum(self.mask * (-s))
        return x, log_det_inv_j
    
    def log_prob(self, y):
        y = y.reshape(y.shape[0], -1)
        x, log_det_inv_j = self.forward(y)
        log_p_x = self.base_dist.log_prob(x)
        # print(log_p_x, log_det_inv_j)
        return log_p_x + log_det_inv_j
