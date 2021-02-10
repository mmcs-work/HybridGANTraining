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
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def get_noise(dataset):
    np.random.seed(2021)
    
    if dataset in ['mnist', 'fmnist']:
        z = np.random.randn(25, 1, 28, 28)
    else:
        z = np.random.randn(25, 3, 32, 32)
        
    return torch.from_numpy(z).float().cuda()
    
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


def run(FLAGS):
    
    train_loader, val_loader, _ = get_data(FLAGS)
    nc = next(iter(train_loader))[0].shape[1]
    
    netG = RealNVP(num_scales=2, in_channels=nc, mid_channels=64, num_blocks=FLAGS.no_of_layers).to(device)
    netD = Discriminator(dataset=FLAGS.dataset, ndf=FLAGS.df_dim).to(device)
    
    if FLAGS.netG != '':
        netG.load_state_dict(torch.load(FLAGS.netG))
    
    if FLAGS.netD != '':
        netD.load_state_dict(torch.load(FLAGS.netD))
    
    
    optimizerD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, FLAGS.beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, FLAGS.beta2))
    
    lmbda = lambda epoch: FLAGS.lr_decay
    scheduler_G = torch.optim.lr_scheduler.MultiplicativeLR(optimizerG, lr_lambda=lmbda)
    scheduler_D = torch.optim.lr_scheduler.MultiplicativeLR(optimizerD, lr_lambda=lmbda)
    
    loss_fn = RealNVPLoss()
#     criterion = torch.nn.BCELoss()
    
    fix_z = get_noise(FLAGS.dataset)
    
    # TODO:
    # load dict
    logger = logging.getLogger('logger')
    hdlr = logging.FileHandler(FLAGS.log_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    
    writer = SummaryWriter(log_dir=FLAGS.writer_log)
    
    for epoch in range(FLAGS.start_epoch, FLAGS.epoch):
        train(netG, 
              netD, 
              optimizerG, 
              optimizerD, 
              scheduler_G, 
              scheduler_D, 
              train_loader, 
              FLAGS, 
              loss_fn, 
              epoch, 
              fix_z, 
              logger, 
              writer)
        
        val(netG, 
            FLAGS, 
            val_loader, 
            loss_fn, 
            epoch, 
            logger, 
            writer)
        
        sample(netG, FLAGS, train_loader, epoch, fix_z)
        
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (FLAGS.checkpoint_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (FLAGS.checkpoint_dir, epoch))
    
    writer.flush()
    

def train(netG, 
          netD, 
          optimizerG, 
          optimizerD, 
          scheduler_G, 
          scheduler_D, 
          train_loader, 
          FLAGS, 
          loss_fn, 
          epoch, 
          fix_z, 
          logger, 
          writer):
#     netD.train()
    
    likelihoods = util.AverageMeter()
    
    with tqdm(total=len(train_loader.dataset)) as pbar:
        for i, (imgs, _) in enumerate(train_loader):
            # real images
#             netG.train()
            
            real_imgs = Variable(imgs.type(Tensor))
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            z = Variable(Tensor(torch.randn(imgs.size(), dtype=torch.float32).cuda()))

            # Generate a batch of images
            fake_imgs, _ = netG(z, reverse = True)

            # Real images
            alpha = 1e-7
            fake_imgs = nn.Sigmoid()(fake_imgs)
            fake_imgs = (fake_imgs-alpha)/(1-2*alpha)
            real_imgs = real_imgs*255.0
            corruption_level = 1.0
            real_imgs = real_imgs + corruption_level * torch.rand(imgs.size()).cuda()
            real_imgs = real_imgs/(255.0 + corruption_level)
            
            # -----------------
            #  Train Generator
            # -----------------
            d_loss = torch.Tensor([0.])
            for c in range(FLAGS.n_critics):
                optimizerD.zero_grad()

                real_validity = netD(real_imgs)
                fake_validity = netD(fake_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + FLAGS.lambda_gp * gradient_penalty

                d_loss.backward(retain_graph=True)
                optimizerD.step()

            optimizerG.zero_grad()
            
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_validity = netD(fake_imgs)
            errG = -torch.mean(fake_validity)

            z , sldj = netG(real_imgs.data, reverse=False)
            likelihood = loss_fn(z, sldj).mean()

            if FLAGS.mode == 'hybrid':
                g_loss = errG / FLAGS.like_reg + likelihood
            elif FLAGS.mode == 'adv':
                g_loss = errG
            elif FLAGS.mode == 'mle':
                g_loss = likelihood
            else:
                raise NotImplemented
            
            g_loss.backward()
            optimizerG.step()
            likelihoods.update(likelihood.item(), imgs.shape[0])

            pbar.set_postfix(Dloss=d_loss.item(), 
                             Gloss=g_loss.item(), 
                             LL=likelihoods.avg,
                             bpd=util.bits_per_dim(np.prod(imgs.size()[1:]), likelihoods.avg))

            pbar.update(imgs.shape[0])
            
            if i % 400 == 0:
                sample(netG, FLAGS, train_loader, epoch, fix_z)
                
    bpd_score = util.bits_per_dim(np.prod(imgs.size()[1:]), likelihoods.avg)
    logger.info(f'epoch: {epoch},'
                f'train_likelihood: {likelihoods.avg},' 
                f'train_bpd: {bpd_score}')
    
    writer.add_scalar('train_likelihood', likelihoods.avg, epoch)
    writer.add_scalar('train_bpd', bpd_score, epoch)


def val(netG, 
        FLAGS, 
        val_loader, 
        loss_fn, 
        epoch, 
        logger, 
        writer):
    
    #netG.eval()
    val_likelihoods = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset)) as pbar:
            for i, (data, _) in enumerate(val_loader):
                
                data = Variable(data.type(Tensor))
                
                data = data*255.0
                corruption_level = 1.0
                data = data + corruption_level * torch.rand(data.size()).cuda()
                data = data/(255.0 + corruption_level)

                z, sldj = netG(data.to(device), reverse=False)
                likelihood = loss_fn(z, sldj).mean()
                val_likelihoods.update(likelihood.item(), data.shape[0])
                
                pbar.set_postfix(LL_val=val_likelihoods.avg,
                                 bpd=util.bits_per_dim(np.prod(data.size()[1:]), val_likelihoods.avg))
                pbar.update(data.shape[0])
    
    bpd_score = util.bits_per_dim(np.prod(data.size()[1:]), val_likelihoods.avg)
    logger.info(f'epoch: {epoch},'
                f'val_likelihood: {val_likelihoods.avg},' 
                f'val_bpd: {bpd_score}')
    
    writer.add_scalar('val_likelihood', val_likelihoods.avg, epoch)
    writer.add_scalar('val_bpd', bpd_score, epoch)
            

def sample(netG, FLAGS, loader, epoch, fix_z):
    
    #netG.eval()
    with torch.no_grad():
        x, _ = netG(fix_z, reverse=True)
        fake = torch.sigmoid(x)
        torchvision.utils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%06d.png' % (FLAGS.sample_dir, epoch), 
                                     nrow=5, 
                        normalize=True)
