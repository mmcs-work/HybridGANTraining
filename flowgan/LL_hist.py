import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import RealNVP, RealNVPLoss
from prepare_data import get_data
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_LL(FLAGS):
    _, loader, _ = get_data(FLAGS)
    
    nc = 1 if FLAGS.dataset in ['mnist', 'fmnist'] else 3
    try:
        netG = RealNVP(num_scales=2, in_channels=nc, mid_channels=64, num_blocks=5).to(device)
        netG.load_state_dict(torch.load(FLAGS.netG))
    except:
        netG = RealNVP(num_scales=2, in_channels=nc, mid_channels=64, num_blocks=8).to(device)
        netG.load_state_dict(torch.load(FLAGS.netG))
                       
    loss_fn = RealNVPLoss()
    
    LL_lst = torch.empty(len(loader.dataset), dtype=torch.float64).to(device)
    
    pointer = 0
    with tqdm(total=len(loader.dataset)) as pbar:
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            
            with torch.no_grad():
                z , sldj = netG(imgs, reverse=False)
                likelihood = loss_fn(z, sldj)
            
            LL_lst[pointer: pointer + FLAGS.batch_size] = likelihood
            pointer += FLAGS.batch_size
            
            pbar.update(FLAGS.batch_size)
    
    fig, ax = plt.subplots(figsize =(10, 7))
    LL_lst_np = LL_lst.cpu().detach().numpy()
    ax.hist(LL_lst_np)
    file = os.path.join(FLAGS.plot_folder, FLAGS.filename)
    plt.title(FLAGS.filename.split('.')[0])
    plt.savefig(file)
    np.save(os.path.join(FLAGS.plot_folder, FLAGS.filename.split('.')[0] + '.npy'), LL_lst_np)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--root_dir', type=str, default="./data")
    parser.add_argument('--dataset', type=str, default="mnist", help='cifar10 | mnist | fmnist')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--checkpoint_dir', default='./flow_gan', help='folder to output images and model checkpoints')
    parser.add_argument('--plot_folder', type=str, default='LL_hist')
    parser.add_argument('--filename', type=str)
    
    FLAGS = parser.parse_args()
    print(FLAGS)
    
    if not os.path.exists(FLAGS.plot_folder):
        os.makedirs(FLAGS.plot_folder)
    
    get_LL(FLAGS)