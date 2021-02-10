import torch
import numpy as np
import os
import torchvision
import util
from prepare_data import get_data
from models import RealNVP, RealNVPLoss
from model import Discriminator
from tqdm import tqdm
import torch.nn as nn
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def discriminator_score(FLAGS):
    real_validity_all_epochs = np.empty(0, dtype=np.float64)
    fake_validity_all_epochs = np.empty(0, dtype=np.float64)
    
    train_loader, val_loader, _ = get_data(FLAGS)
    nc = 1 if FLAGS.dataset == 'mnist' else 3
    netG = RealNVP(num_scales=2, in_channels=nc, mid_channels=64, num_blocks=FLAGS.no_of_layers).to(device)
    netD = Discriminator(dataset=FLAGS.dataset, ndf=FLAGS.df_dim).to(device)
    
    if FLAGS.netG != '':
        netG.load_state_dict(torch.load(FLAGS.netG))
    
    if FLAGS.netD != '':
        netD.load_state_dict(torch.load(FLAGS.netD))
    
    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset)) as pbar:
            for i, (real_imgs, _) in enumerate(val_loader):
                # real images
                real_imgs = real_imgs.cuda()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                z = torch.randn(real_imgs.size(), dtype=torch.float32).cuda()

                # Generate a batch of images
                fake_imgs, _ = netG(z, reverse = True)

                # Real images
                fake_imgs = nn.Sigmoid()(fake_imgs)
#                 fake_imgs = (fake_imgs-FLAGS.alpha)/(1-2*FLAGS.alpha)
#                 real_imgs = real_imgs*255.0
#                 corruption_level = 1.0
#                 real_imgs = real_imgs + corruption_level * torch.rand(real_imgs.size()).cuda()
#                 real_imgs = real_imgs/(255.0 + corruption_level)


                real_validity = netD(real_imgs)
                fake_validity = netD(fake_imgs)

                real_validity_all_epochs = np.append(real_validity_all_epochs, real_validity.reshape(-1).cpu().detach().numpy())
                fake_validity_all_epochs = np.append(fake_validity_all_epochs, fake_validity.reshape(-1).cpu().detach().numpy())
                
                pbar.update(real_imgs.shape[0])
    
    np.savez(FLAGS.filename + '.npz', real_validity_all_epochs, fake_validity_all_epochs)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--root_dir', type=str, default="./data")

    parser.add_argument('--dataset', type=str, default="mnist", help='cifar10 | mnist')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--alpha', type=float, default=1e-7)
    parser.add_argument('--no_of_layers', type=int, default=5)
    parser.add_argument('--df_dim', type=int, default=64)
    parser.add_argument('--filename', type=str, default='_name')
    
    FLAGS = parser.parse_args()
    discriminator_score(FLAGS)