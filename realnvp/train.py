"""Train Real NVP on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import logging
from models import RealNVP, RealNVPLoss
from tqdm import tqdm


def main(args):
    logger = logging.getLogger('logger')
    hdlr = logging.FileHandler('/nfs/students/winter-term-2020/project-5/real-nvp/realnvp_log.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('Building model..')
    net = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('/nfs/students/winter-term-2020/project-5/real-nvp/ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        
        ########################################################################
        if args.generate == 1:
            batch_size = args.generate_img_nums
            index = 0
            num_of_times = int(batch_size / 64) + int((batch_size % 64) > 0) 

            for i in tqdm(range(num_of_times)):
                x = sample(net, 64, device)
                fake = x.detach()
                for i in range(fake.shape[0]):
                    img = fake[i, :, :, :]
                    img = torchvision.transforms.ToPILImage()(img)
                    img.save(f'{args.generate_loc}/img_{index}.png')
                    index = index + 1

            import sys
            sys.exit()
        ########################################################################
        
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm, logger)
        test(epoch, net, testloader, device, loss_fn, args.num_samples, logger)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, logger):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    #likelihood_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            #likelihood = -loss_fn(z, 0)
            loss_meter.update(loss.item(), x.size(0))
            #likelihood_meter.update(likelihood.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
    __bpd = util.bits_per_dim(x,loss_meter.avg)
    logger.info(f'epoch: {epoch}, loss: {loss_meter.avg}, loss(act): {loss.item()} bpd: {__bpd}')


def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


def test(epoch, net, testloader, device, loss_fn, num_samples, logger):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
#    likelihood_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
 #               likelihood = -loss_fn(z, 0)
                loss_meter.update(loss.item(), x.size(0))
#                likelihood_meter.update(likelihood.item(), 1)
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))
    
    #logger.info(f'epoch: {epoch}, val_loss: {loss_meter.avg}')
    __bpd = util.bits_per_dim(x,loss_meter.avg)
    logger.info(f'epoch: {epoch}, val_loss: {loss_meter.avg}, loss(act): {loss.item()} bpd: {__bpd}')
    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    #######################################################################################
    parser.add_argument('--generate', type=int, default=0, help='generate images')
    parser.add_argument('--generate_loc', default='', help='location of generate images')
    parser.add_argument('--generate_img_nums', type=int, default=100, help='Number of images to generate')
    #######################################################################################

    best_loss = 1e8

    main(parser.parse_args())
