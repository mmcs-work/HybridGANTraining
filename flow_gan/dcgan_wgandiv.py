
from __future__ import print_function
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
import util
import logging
import numpy as np

from models import RealNVP, RealNVPLoss
from tqdm.autonotebook import tqdm
import torchvision
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
"""done"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=28)
parser.add_argument('--ndf', type=int, default=28)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--log_path', default='./flow_gan.log', help='path to store the log file')
parser.add_argument('--start_epoch', type=int, default=0, help='continue training')
parser.add_argument('--root_dir', type=str, help='load self-generated imgs')
#######################################################################################
parser.add_argument('--generate', type=int, default=0, help='generate images')
parser.add_argument('--generate_loc', default='', help='location of generate images')
parser.add_argument('--generate_img_nums', type=int, default=100, help='Number of images to generate')
#######################################################################################
parser.add_argument('--likelihood', type=int, default=0, help='calculate likelihood')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: r", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

root_dir = opt.root_dir
def my_dataloader(nc, transform):
    
    lst = os.listdir(root_dir)
    imgs = []
    for n in range(len(lst)):
        img = transform(Image.open(os.path.join(root_dir, lst[n])))
        img = img.unsqueeze(0)
        imgs.append(img)

        if len(imgs) == opt.batchSize or n == len(lst):
            batch = torch.cat(imgs, dim = 0)
            assert batch.shape != [opt.batchSize, nc, opt.imageSize, opt.imageSize]
            imgs = []
            yield batch
            
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                              #  transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1
        dataset, _ = torch.utils.data.random_split(dataset, [500, 59500])

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
#dataloader = my_dataloader(nc, transform=transforms.Compose([transforms.Resize(opt.imageSize), 
#                                                     transforms.ToTensor()]))
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
p = 6
k = 2
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = RealNVP(num_scales=2, in_channels=nc, mid_channels=64, num_blocks=8).to(device)
# netG = Generator(ngpu).to(device)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
########################################################################
if opt.generate == 1:
    batch_size = opt.generate_img_nums
    index = 0
    num_of_times = int(batch_size / 64) + int((batch_size % 64) > 0) 
    likelihoods = util.AverageMeter()
    loss_fn = RealNVPLoss()
    with torch.no_grad():
        for i in tqdm(range(num_of_times)):
            z = torch.randn((64, 1, 28, 28), dtype=torch.float32).to(device)
            x, _ = netG(z, reverse=True)
            
            fake = torch.sigmoid(x)
            z, sldj = netG(fake, reverse=False)
            likelihood = loss_fn(z, sldj)
            likelihoods.update(likelihood.item(), fake[0].size(0))
            print("bpd\n")
            print(util.bits_per_dim(x, likelihoods.avg))
            print("\n")
            fake = fake.detach()
            for i in range(fake.shape[0]):
                img = fake[i, :, :, :]
                img = torchvision.transforms.ToPILImage()(img)
                img.save(f'{opt.generate_loc}/img_{index}.png')
                index = index + 1
        print(likelihoods.avg)
        
        
    import sys
    sys.exit()
########################################################################
# print(netG)
#################################
if opt.likelihood == 1:
    dataloader = my_dataloader(nc, transform=transforms.Compose([transforms.Resize(opt.imageSize), 
                                                    transforms.ToTensor()]))
    likelihoods = util.AverageMeter()
    loss_fn = RealNVPLoss()
    #logll = []
    with tqdm(total=len(os.listdir(root_dir))) as pbar:
        for i, data in enumerate(dataloader):
            netG.zero_grad()
            #print(data.shape)
            data = data.to(device)
            z, sldj = netG(data, reverse=False)
            likelihood = loss_fn(z, sldj)
            #print(likelihood)
            #logll.append(-likelihood.item())
            likelihoods.update(likelihood.item(), data[0].size(0))
            #print("---")
            pbar.update(opt.batchSize)
    print(likelihoods.avg)
    #print(np.mean(np.array(logll)))
    import sys
    sys.exit()       
#################################

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.ReLU(),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)



netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
# print(netD)

criterion = nn.BCELoss()
loss_fn = RealNVPLoss()

fixed_noise = torch.randn((opt.batchSize, nc, opt.imageSize, opt.imageSize), dtype=torch.float32, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1

logger = logging.getLogger('logger')
hdlr = logging.FileHandler(opt.log_path)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

alpha = 1.0

for epoch in range(opt.start_epoch, opt.niter):
#     alpha -= 0.01
        
    loss_d = util.AverageMeter()
    loss_g = util.AverageMeter()
#     Dx = util.AverageMeter()
#     DGz1 = util.AverageMeter()
#     DGz2 = util.AverageMeter()
    likelihoods = util.AverageMeter()
    with tqdm(total=len(dataloader.dataset)) as pbar:
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            real_imgs = Variable(real_cpu.type(Tensor), requires_grad=True)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)
            # print(real_cpu.shape)
            output_real = netD(real_imgs)
#             print(output_real.shape)
#             print(real_cpu.size(0))
            #errD_real = criterion(output_real, label)
#             errD_real.backward()
#             D_x = output_real.mean().item()
            
            # train with fake
            # noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # fake = netG(noise)
            z = Variable(torch.randn((batch_size, nc, opt.imageSize, opt.imageSize), dtype=torch.float32, device=device))
            x, _ = netG(z, reverse=True)
            fake = torch.sigmoid(x)
            label.fill_(fake_label)
            #output_fake = netD(fake.detach())
            output_fake = netD(x)
            #errD_fake = criterion(output, label)
#             errD_fake.backward()
#             D_G_z1 = output.mean().item()
            #errD = errD_real + errD_fake
#             optimizerD.step()
            
            #######################################
            real_grad_out = Variable(Tensor(real_cpu.size(0), 1).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(
            output_real.unsqueeze(1), real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(fake.size(0), 1).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(
            output_fake.unsqueeze(1), x, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

            # Adversarial loss
            d_loss = -torch.mean(output_real) + torch.mean(output_fake) + div_gp
            #######################################
            d_loss.backward()
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            print("---------------")
            #printf(output.size())
            #pp = output.squeeze()
            errG = criterion(nn.Sigmoid()(output), label)
            z, sldj = netG(data[0].to(device), reverse=False)
            likelihood = loss_fn(z, sldj)
            hybrid =  (1 - alpha) * likelihood
            hybrid.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # compute likelihood
#             with torch.no_grad():
#                 z, sldj = netG(data[0].to(device), reverse=False)
#                 likelihood = -loss_fn(z, sldj)
            
#             loss_d.update(errD.item(), data[0].size(0))
#             loss_g.update(errG.item(), data[0].size(0))
            likelihoods.update(likelihood, data[0].size(0))

            pbar.set_postfix(epoch=epoch,
                             batch=i, 
                             likelihood=likelihood)
            pbar.update(batch_size)

    logger.info(f'epoch: {epoch}, Loss_D: {loss_d.avg}, Loss_G: {loss_g.avg}, likelihood: {likelihoods.avg}')
    vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            # fake = netG(fixed_noise)
    x, _ = netG(fixed_noise, reverse=True)
    fake = torch.sigmoid(x)
    vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

#        if opt.dry_run:
#            break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    
