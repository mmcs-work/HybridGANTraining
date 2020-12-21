#Experiment 5:
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from models import RealNVP, RealNVPLoss

os.makedirs("imagesLL", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, image):
        

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = np.array(image)
#         #image = image.transpose((2, 0, 1))
#         v = torch.from_numpy(image)
#         print(torch.min(v))
#         v= torch.unsqueeze(v,0)
#         #print(v.shape)
#         return v
    
    
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(opt.latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.shape[0], *img_shape)
#         return img

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
        
        return x
    
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )

#     def forward(self, img):
#         img_flat = img.view(img.shape[0], -1)
#         validity = self.model(img_flat)
#         return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
#generator = Generator()
generator = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=4).cuda()
discriminator = Discriminator()
loss_fn = RealNVPLoss()


if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size),transforms.ToTensor()]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr) #, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        
        z = Variable(Tensor(torch.randn((imgs.shape[0], 1, 28, 28), dtype=torch.float32).cuda()))
        
        # Generate a batch of images
        fake_imgs, _ = generator(z, reverse = True)

        # Real images
        alpha = 1e-7
        fake_imgs = nn.Sigmoid()(fake_imgs)
        fake_imgs = (fake_imgs-alpha)/(1-2*alpha)
        real_imgs = real_imgs*255.0
        corruption_level = 1.0
        real_imgs = real_imgs + corruption_level * torch.rand((imgs.shape[0], 1, 28, 28)).cuda()
        real_imgs = real_imgs/(255.0 + corruption_level)
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs, _ = generator(z, reverse=True)
            fake_imgs = nn.Sigmoid()(fake_imgs)
            fake_imgs = (fake_imgs-alpha)/(1-2*alpha)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            # fake_imgs = nn.Sigmoid()(fake_imgs)
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            z , sldj = generator(real_imgs.data, reverse=False)
            likelihood = loss_fn(z, sldj)
            
            g_loss += likelihood
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            fake_imgs = fake_imgs * 255.0
            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "imagesLL/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic