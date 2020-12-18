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
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=ndf, kernel_size=5, stride=2, padding=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.LayerNorm([ndf, 16, 16]),
                nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=5, stride=2, padding=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.LayerNorm([ndf * 2, 8, 8]),
                nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=5, stride=2, padding=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.LayerNorm([ndf * 4, 4, 4]),
                nn.Conv2d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=5, stride=2, padding=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.LayerNorm([ndf * 8, 2, 2])
            )
            
            self.lin = nn.Sequential(
                nn.Linear(512 * 2 * 2, 1)
            )
    
    def forward(self, x):
#         print(x.shape)
        x = self.conv(x)
#         print(x.shape)
        x = x.reshape(x.shape[0], -1)
#         print(x.shape)
        x = self.lin(x)
#         print(x.shape)
        
#         return torch.sigmoid(x), x
        return x
