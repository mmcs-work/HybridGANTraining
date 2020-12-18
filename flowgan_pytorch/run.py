import torch
import torchvision
import util
from prepare_data import get_data
from models import RealNVP, RealNVPLoss
from model import Discriminator
from torch.autograd import Variable
from tqdm import tqdm
import torch.autograd as autograd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_gradient_penalty(netD, batch_size, real_images, fake_images):
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.cuda()
    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.cuda()

    interpolated = Variable(interpolated, requires_grad=True)

    prob_interpolated = torch.sigmoid(netD(interpolated))

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() #* self.lambda_term
    return grad_penalty


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
        train(netG, netD, optimizerG, optimizerD, train_loader, FLAGS, loss_fn, device, logger=None)
        val(netG, FLAGS, val_loader, loss_fn, device)
        sample(netG, FLAGS, train_loader, epoch, device=device)
        
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (FLAGS.checkpoint_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (FLAGS.checkpoint_dir, epoch))


def train(netG, netD, optimizerG, optimizerD, train_loader, FLAGS, loss_fn, device, logger):
    netG.train()
    netD.train()
    
    likelihoods = util.AverageMeter()
    
    with tqdm(total=len(train_loader.dataset)) as pbar:
        for i, batch in enumerate(train_loader):
            # real images
            optimizerD.zero_grad()
            data, _ = batch
            data = data.to(device)
            batch_size = data.shape[0]
            output_real = netD(data)
            
            # fake images
            z = torch.randn(data.size(), dtype=torch.float32, device=device)
            x, _ = netG(z, reverse=True)
            fake = torch.sigmoid(x)
            #output_fake = netD(fake.detach())
            output_fake = netD(fake)
            
            div_gp = calculate_gradient_penalty(netD, batch_size, data, fake)
            d_loss = -torch.mean(output_real) + torch.mean(output_fake) + 0.5 * div_gp
            d_loss.backward(retain_graph=True)
            optimizerD.step()
            
            # update G
            if i % FLAGS.n_critics == 0:
                optimizerG.zero_grad()
                z = torch.randn(data.size(), dtype=torch.float32, device=device)
                x, _ = netG(z, reverse=True)
#                 print(x.min().item(), x.max().item())
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
        
            pbar.set_postfix(LL=likelihoods.avg,
                             bpd=util.bits_per_dim(torch.randn(data.size(), dtype=torch.float32), likelihoods.avg))
            pbar.update(batch_size)


def val(netG, FLAGS, val_loader, loss_fn, device):
    netG.eval()
    val_likelihoods = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset)) as pbar:
            for i, batch in enumerate(val_loader):
                data, _ = batch
                data = data.to(device)
                batch_size = data.shape[0]
                z, sldj = netG(data.to(device), reverse=False)
                likelihood = loss_fn(z, sldj)
                val_likelihoods.update(likelihood.item(), data.shape[0])
                
                pbar.set_postfix(LL_val=val_likelihoods.avg,
                                 bpd=util.bits_per_dim(torch.randn(data.size(), dtype=torch.float32), val_likelihoods.avg))
                pbar.update(batch_size)
            

def sample(netG, FLAGS, loader, epoch, device):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(next(iter(loader))[0].size(), dtype=torch.float32, device=device)
        x, _ = netG(z, reverse=True)
        fake = torch.sigmoid(x)
        torchvision.utils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (FLAGS.sample_dir, epoch),
                        normalize=True)