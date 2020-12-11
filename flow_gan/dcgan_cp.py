
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset
from PIL import Image
import torch.utils.data as data
import util
import glob
import logging
from skimage import io


from models import RealNVP, RealNVPLoss
from tqdm.autonotebook import tqdm
###################################
import torchvision
###################################

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


# root_dir="/nfs/students/winter-term-2020/project-5/project-5-manna/flow_gan/GAANDataGenforFlow"
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
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = my_dataloader(nc, transform=transforms.Compose([transforms.Resize(opt.imageSize), 
                                                         transforms.ToTensor()]))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


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

netG = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8).to(device)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
########################################################################
if opt.generate == 1:
    batch_size = opt.generate_img_nums
    index = 0
    num_of_times = int(batch_size / 64) + int((batch_size % 64) > 0) 
    
    for i in tqdm(range(num_of_times)):
        z = torch.randn((64, 1, 28, 28), dtype=torch.float32).to(device)
        x, _ = netG(z, reverse=True)
        fake = torch.sigmoid(x)
        fake = fake.detach()
        for i in range(fake.shape[0]):
            img = fake[i, :, :, :]
            img = torchvision.transforms.ToPILImage()(img)
            img.save(f'{opt.generate_loc}/img_{index}.png')
            index = index + 1

        
    import sys
    sys.exit()
########################################################################
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
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
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
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

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
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
#################################
if opt.likelihood == 1:
    dataloader = my_dataloader(nc, transform=transforms.Compose([transforms.Resize(opt.imageSize), 
                                                         transforms.ToTensor()]))
    likelihoods = util.AverageMeter()
    with tqdm(total=len(os.listdir(root_dir))) as pbar:
        for i, data in enumerate(dataloader):
            z, sldj = netG(data.to(device), reverse=False)
            likelihood = loss_fn(z, sldj)
            ###### Changing the loss to likelihood only.
            #hybrid = errG / 20 + likelihood
            
            # compute likelihood
#             with torch.no_grad():
#                 z, sldj = netG(data[0].to(device), reverse=False)
#                 likelihood = -loss_fn(z, sldj)
            
            likelihoods.update(likelihood, data[0].size(0))
        print(f'likelihood: {likelihoods.avg}')
    
    import sys
    sys.exit()       
#################################
for epoch in range(opt.start_epoch, opt.niter):
    dataloader = my_dataloader(nc, transform=transforms.Compose([transforms.Resize(opt.imageSize), 
                                                         transforms.ToTensor()]))
#     if epoch % 5 == 4:
#         alpha *= 0.9
        
    loss_d = util.AverageMeter()
    loss_g = util.AverageMeter()
#     Dx = util.AverageMeter()
#     DGz1 = util.AverageMeter()
#     DGz2 = util.AverageMeter()
    likelihoods = util.AverageMeter()
    with tqdm(total=len(os.listdir(root_dir))) as pbar:
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
#             print(i, data.shape)
            netD.zero_grad()
#             print(data.shape)
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)
#             print(real_cpu.shape)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            # noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # fake = netG(noise)
            z = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device=device)
            x, _ = netG(z, reverse=True)
            fake = torch.sigmoid(x)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            z, sldj = netG(data.to(device), reverse=False)
            likelihood = loss_fn(z, sldj)
            ###### Changing the loss to likelihood only.
            #hybrid = errG / 20 + likelihood
            hybrid = likelihood
            hybrid.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # compute likelihood
#             with torch.no_grad():
#                 z, sldj = netG(data[0].to(device), reverse=False)
#                 likelihood = -loss_fn(z, sldj)
            
            loss_d.update(errD.item(), data[0].size(0))
            loss_g.update(errG.item(), data[0].size(0))
            likelihoods.update(likelihood, data[0].size(0))

            pbar.set_postfix(epoch=epoch,
                             batch=i, 
                             errD=errD.item(),
                             errG=errG.item(),
                             likelihood=likelihood)
            pbar.update(batch_size)

    logger.info(f'epoch: {epoch}, Loss_D: {loss_d.avg}, Loss_G: {loss_g.avg}, likelihood: {likelihoods.avg}')
    vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            # fake = netG(fixed_noise)
    z = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device=device)
    x, _ = netG(z, reverse=True)
    fake = torch.sigmoid(x)
    vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

#        if opt.dry_run:
#            break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    
