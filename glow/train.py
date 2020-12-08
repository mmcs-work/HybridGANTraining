from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse
import os
import torch
import logging
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--epochs", default=20, type=int, help="epochs")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--dataset", type=str, help="dataset to train on")
parser.add_argument("--logdir", type=str, help="log path")
parser.add_argument("--path_sample", type=str)
parser.add_argument("--path_model", type=str)


def sample_data(dataset, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    
    if dataset == "mnist":
        dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    elif dataset == "cifar":
        dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    val_set = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_set


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, in_channel, model, optimizer, logger):
    train_loader, val_set = sample_data(args.dataset, args.batch, args.img_size)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(in_channel, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        train_losses = []
        log_ps = []
        log_dets = []
        for i, data in enumerate(pbar):
            image, _ = data
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if epoch == 0 and i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            train_losses.append(loss.item())
            log_ps.append(log_p.item())
            log_dets.append(log_det.item())
            
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_postfix(
                train_loss='{:.3f}'.format(loss), 
                logP='{:.3f}'.format(log_p), 
                logdet='{:.3f}'.format(log_det), 
                lr='{:f}'.format(warmup_lr)
            )
        
        pbar = tqdm(val_set)
        val_losses = []
        val_log_ps = []
        val_log_dets = []
        with torch.no_grad():
            for i, data in enumerate(pbar):
                image, _ = data
                image = image.to(device)

                image = image * 255

                if args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5

                if epoch == 0 and i == 0:
                    with torch.no_grad():
                        log_p, logdet, _ = model.module(
                            image + torch.rand_like(image) / n_bins
                        )

                        continue

                else:
                    log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

                logdet = logdet.mean()

                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                val_losses.append(loss.item())
                val_log_ps.append(log_p.item())
                val_log_dets.append(log_det.item())

                pbar.set_postfix(
                    val_loss='{:.3f}'.format(loss), 
                    logP='{:.3f}'.format(log_p), 
                    logdet='{:.3f}'.format(log_det), 
                    lr='{:f}'.format(warmup_lr)
                )

        logger.info(f'train_loss: {np.array(train_losses).mean()}, val_loss: {np.array(val_losses).mean()}, logP: {np.array(log_ps).mean()}, logdet: {np.array(log_dets).mean()}, val_logP: {np.array(val_log_ps).mean()}, val_logdet: {np.array(val_log_dets).mean()}')

        with torch.no_grad():
            utils.save_image(
                model_single.reverse(z_sample).cpu().data,
                os.path.join(args.path_sample, "model_" + str(epoch) + ".png"),
                normalize=True,
                nrow=5,
                range=(-0.5, 0.5)
            )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.path_model, "model_" + str(epoch + 1) + ".pt")
            )
            torch.save(
                optimizer.state_dict(), os.path.join(args.path_model, "optim_" + str(epoch + 1) + ".pt")
            )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.path_sample):
        os.makedirs(args.path_sample)
    if not os.path.isdir(args.path_model):
        os.makedirs(args.path_model)
    
    in_channel = 1 if args.dataset == "mnist" else 3
    model_single = Glow(
        in_channel, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = logging.getLogger('logger')
    hdlr = logging.FileHandler(args.logdir)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

    train(args, in_channel, model, optimizer, logger)
