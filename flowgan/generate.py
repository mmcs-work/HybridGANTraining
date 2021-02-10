import torch
import os
import torchvision
import util
from prepare_data import get_data
from models import RealNVP, RealNVPLoss
from model import Discriminator
from torch.autograd import Variable
from tqdm import tqdm
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter
from numpy import asarray
from numpy import savez_compressed


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def generate(FLAGS):
    #likelihood_store = []
    if FLAGS.dataset == 'mnist':
        nc = 1
    else:
        nc = 3
    
    netG = RealNVP(num_scales=2, in_channels=nc, mid_channels=64, num_blocks=FLAGS.no_of_layers).to(device)
    netD = Discriminator(dataset=FLAGS.dataset, ndf=FLAGS.df_dim).to(device)
    
    if FLAGS.netG != '':
        netG.load_state_dict(torch.load(FLAGS.netG))
    
    if FLAGS.netD != '':
        netD.load_state_dict(torch.load(FLAGS.netD))
    
    loss_fn = RealNVPLoss()
    likelihoods = util.AverageMeter()
    
    print(FLAGS.generate_num_imgs//FLAGS.batch_size)
    for i in tqdm(range(FLAGS.generate_num_imgs//FLAGS.batch_size)):
        if FLAGS.dataset == 'mnist':
            z = torch.randn((FLAGS.batch_size, 1, 28, 28)).to(device)
        else:
            z = torch.randn((FLAGS.batch_size, 3, 32, 32)).to(device)
      
        likelihood = sample_imgs_with_likelihood(netG, FLAGS, z, loss_fn, i)
        likelihoods.update(likelihood.item())
        #likelihood_store.append(likelihood.item())

    print(likelihoods.avg)
    
    #likelihood_store = np.array(likelihood_store)
    #savez_compressed('data.npz', likelihood_store)
    return None


def sample_imgs_with_likelihood(netG, FLAGS, z, loss_fn, epoch):
    
    img_dir = os.path.join('/nfs/students/winter-term-2020/project-5/flowgan_local/Generated_Images', 
                                           FLAGS.generate_sample_dir)
    #netG.eval()
    
    with torch.no_grad():
        x, _ = netG(z, reverse=True)
        fake = torch.sigmoid(x)
        for i in range(fake.shape[0]):
            img = fake[i, :, :, :]
            img = torchvision.transforms.ToPILImage()(img)
            lens = len(os.listdir(img_dir))
            img.save(os.path.join(img_dir, 
                                  'img_%d.png' % (epoch*z.shape[0]+i)))
    
        z, sldj = netG(fake, reverse=False)
#         assert torch.allclose(_, z, atol=1e-3)
        likelihood = loss_fn(z, sldj).mean()
        
        return likelihood