import os
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dataset='mnist', ndf=64, dfc_dim=1024):
        super(Discriminator, self).__init__()
        
        if dataset == 'mnist':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=1, out_channels=ndf, kernel_size=5, stride=2, padding=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            self.lin = nn.Sequential(
                nn.Linear(in_features=49 * ndf, out_features=dfc_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(in_features=dfc_dim, out_features=1)
            )
        else:
            # https://github.com/ermongroup/flow-gan/blob/91b745e7811479a1d73074b3e33e24b3cbb38823/model.py#L611
            raise NotImplemented
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.lin(x)
        
#         return torch.sigmoid(x), x
        return x.squeeze(1)
