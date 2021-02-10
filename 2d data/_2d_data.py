import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import distributions
from tqdm import tqdm
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
###################################
import torchvision
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd

###################################

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
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


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

nets = lambda: nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 2), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 2))
masks = torch.from_numpy(np.array([[0, 1], [1, 0]]).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2).cuda(), torch.eye(2).cuda())


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 9), 
            nn.LeakyReLU(),
            nn.Linear(9, 2)
        )
    
    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 10), 
            nn.LeakyReLU(),
            nn.Linear(10, 10), 
            nn.LeakyReLU(),
            nn.Linear(10, 1),
#             nn.LeakyReLU(),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 1)
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)

    
netD = Discriminator().cuda()
# netG = Generator().cuda()
netG = RealNVP(nets, nett, masks, prior).cuda()

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=3e-4)
optimizerG = optim.Adam(netG.parameters(), lr=3e-4)

pbar = tqdm(range(100000))
for epoch in pbar:
    
    # ADV training
    for _ in range(3):
        netD.zero_grad()
        real = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
        real = torch.from_numpy(real).cuda()

        label = torch.ones(1000).cuda()

        real_output = netD(real)
        errD_real = criterion(real_output.squeeze(1), label)
        errD_real.backward()

        # train with fake
        noise = torch.randn(1000, 2).cuda()
        if isinstance(netG, RealNVP):
            fake = netG.g(noise)
        else:
            fake = netG(noise)
            
        label = torch.zeros(1000).cuda()
        fake_output = netD(fake.detach())
        errD_fake = criterion(fake_output.squeeze(1), label)
        errD_fake.backward()

#         gradient_penalty = compute_gradient_penalty(netD, real, fake)
#         d_loss = -torch.mean(real_output) + torch.mean(fake_output) + 10. * gradient_penalty
#         d_loss.backward(retain_graph=True)
        optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    optimizerG.zero_grad()
    label = torch.ones(1000).cuda()  # fake labels are real for generator cost
    output = netD(fake)
    errG = criterion(output.squeeze(1), label)
#     errG = -torch.mean(output)
    errG.backward()
    optimizerG.step()



    # MLE training
#     noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)
#     loss = -netG.log_prob(torch.from_numpy(noisy_moons).cuda()).mean()
    
#     optimizerG.zero_grad()
#     loss.backward(retain_graph=True)
#     optimizerG.step()

    loss = -netG.log_prob(real).mean()
    pbar.set_postfix(ll=loss.item())
    
    if epoch % 1000 == 0:
        plt.figure()
        if isinstance(netG, RealNVP):
            fake = netG.sample(1000).cpu().detach().numpy()
            plt.scatter(fake[:, 0, 0], fake[:, 0, 1], c='r')
        else:
            noise = torch.randn(1000, 2).cuda()
            fake = netG(noise).cpu().detach().numpy()
            plt.scatter(fake[:, 0], fake[:, 1], c='r')
            
        plt.ylim((-0.7, 1))
        plt.xlim((-1, 2))
        plt.savefig('foo_{}.png'.format(epoch))
        plt.close()
        
        if isinstance(netG, RealNVP):
            plt.figure()
            x = np.linspace(-1, 2, 100)
            y = np.linspace(-0.7, 1, 100)
            xx, yy = np.meshgrid(x, y, sparse=True)
            # xx = xx[0]
            to_cal_list = np.transpose([np.tile(xx[0], len(yy)), np.repeat(yy, len(xx[0]))])
            loss = -netG.log_prob(torch.from_numpy(to_cal_list).float().cuda()).cpu().detach().numpy().reshape(100, 100)
            plt.contourf(x,y,loss)
            plt.savefig('heatmap_{}.png'.format(epoch))
            plt.close()

        
plt.figure()
real = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
plt.scatter(real[:, 0], real[:, 1], c='r')
plt.savefig('real.png')
plt.close()

real = torch.from_numpy(real).cuda()
loss = -netG.log_prob(real).mean()
print(f'NLL of real image: {loss.item()}')

plt.figure()
fake = netG.sample(1000).cpu().detach().numpy()
plt.scatter(fake[:, 0, 0], fake[:, 0, 1], c='r')
plt.savefig('foo.png')
fake = fake.squeeze(1)
loss = -netG.log_prob(torch.from_numpy(fake).cuda()).mean()
print(f'NLL of fake image: {loss.item()}')
plt.close()

plt.figure()
x = np.linspace(-1, 2, 100)
y = np.linspace(-0.7, 1, 100)
xx, yy = np.meshgrid(x, y, sparse=True)
# xx = xx[0]
to_cal_list = np.transpose([np.tile(xx[0], len(yy)), np.repeat(yy, len(xx[0]))])
loss = -netG.log_prob(torch.from_numpy(to_cal_list).float().cuda()).cpu().detach().numpy().reshape(100, 100)
plt.contourf(x,y,loss)
plt.savefig('heatmap.png')
plt.close()