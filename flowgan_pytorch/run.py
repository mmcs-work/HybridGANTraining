import torch
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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
    
    optimizerD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, FLAGS.beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, FLAGS.beta2))
    
    loss_fn = RealNVPLoss()
    criterion = torch.nn.BCELoss()
    
    # TODO:
    # load dict
    
    for epoch in range(FLAGS.epoch):
        train(netG, netD, optimizerG, optimizerD, train_loader, FLAGS, loss_fn, epoch, logger=None)
        val(netG, FLAGS, val_loader, loss_fn)
        sample(netG, FLAGS, train_loader, epoch)
        
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (FLAGS.checkpoint_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (FLAGS.checkpoint_dir, epoch))


def train(netG, netD, optimizerG, optimizerD, train_loader, FLAGS, loss_fn, epoch, logger):
    netG.train()
    netD.train()
    
    likelihoods = util.AverageMeter()
    
    with tqdm(total=len(train_loader.dataset)) as pbar:
        for i, (data, _) in enumerate(train_loader):
            # real images
            netG.train()
            if data.min().item() < -0.001:
                data = (data + 1.) / 2
            
            optimizerD.zero_grad()
            data = Variable(data.type(Tensor))
            output_real = netD(data)
            
            # fake images
            z = Variable(Tensor(torch.randn(data.size(), dtype=torch.float32).cuda()))
            x, _ = netG(z, reverse=True)
            fake = torch.sigmoid(x)
            output_fake = netD(fake)
            
            div_gp = compute_gradient_penalty(netD, data.data, fake.data)
            d_loss = -torch.mean(output_real) + torch.mean(output_fake) + FLAGS.lambda_gp * div_gp
            d_loss.backward()
            optimizerD.step()
            
            # update G
            if i % FLAGS.n_critics == 0:
                optimizerG.zero_grad()
                x, _ = netG(z, reverse=True)
                fake = torch.sigmoid(x)
                output = netD(fake)
                errG = -torch.mean(output)
                z, sldj = netG(data, reverse=False)
                likelihood = loss_fn(z, sldj)
                
                if FLAGS.mode == 'hybrid':
                    g_loss = errG / 20  +  likelihood
                elif FLAGS.mode == 'adv':
                    g_loss = errG
                elif FLAGS.mode == 'mle':
                    g_loss = likelihood
                else:
                    raise NotImplemented
                    
                g_loss.backward()
                optimizerG.step()
                likelihoods.update(likelihood.item(), data.shape[0])
        
            pbar.set_postfix(Dloss=d_loss.item(), 
                             Gloss=g_loss.item(), 
                             LL=likelihoods.avg,
                             bpd=util.bits_per_dim(torch.randn(data.size(), dtype=torch.float32), likelihoods.avg))

            pbar.update(data.shape[0])
            
            if i % 400 == 1:
                sample(netG, FLAGS, train_loader, epoch)


def val(netG, FLAGS, val_loader, loss_fn):
    netG.eval()
    val_likelihoods = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset)) as pbar:
            for i, (data, _) in enumerate(val_loader):
                data = data.to(device)
                if data.min().item() < -0.001:
                    data = (data + 1.) / 2
                z, sldj = netG(data.to(device), reverse=False)
                likelihood = loss_fn(z, sldj)
                val_likelihoods.update(likelihood.item(), data.shape[0])
                
                pbar.set_postfix(LL_val=val_likelihoods.avg,
                                 bpd=util.bits_per_dim(torch.randn(data.size(), dtype=torch.float32), val_likelihoods.avg))
                pbar.update(data.shape[0])
            

def sample(netG, FLAGS, loader, epoch):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(next(iter(loader))[0].size(), dtype=torch.float32, device=device)
        x, _ = netG(z, reverse=True)
        fake = torch.sigmoid(x)
        torchvision.utils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (FLAGS.sample_dir, epoch),
                        normalize=True)