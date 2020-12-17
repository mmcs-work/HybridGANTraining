import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0, help='continue training')
parser.add_argument('--root_dir', type=str, help='load self-generated imgs')

parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--epoch', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, default=28)
parser.add_argument('--c_dim', type=int, default=3)
parser.add_argument('--checkpoint_dir', default='./flow_gan', help='folder to output images and model checkpoints')
parser.add_argument('--log_path', default='./flow_gan.log', help='path to store the log file')
parser.add_argument('--sample_dir', default='./flow_gan_sample', help='folder to output images and model checkpoints')
parser.add_argument('--f_div', type=str, default='wgan')
parser.add_argument('--alpha', type=float, default=1e-7)
parser.add_argument('--lr_decay', type=float, default=1.0)
parser.add_argument('--min_lr', type=float, default=0.0)
parser.add_argument('--reg', type=float, default=1.0, help='for wgan')
parser.add_argument('--model_type', type=str, default="real_nvp")
parser.add_argument('--n_critics', type=int, default=5)
parser.add_argument('--no_of_layers', type=int, default=8)
parser.add_argument('--hidden_layers', type=int, default=1000)
parser.add_argument('--like_reg', type=float, default=0.0)
parser.add_argument('--df_dim', type=int, default=64)

FLAGS = parser.parse_args()
print(FLAGS)


def main():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
        
    model = DCGAN(
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        c_dim=FLAGS.c_dim,
        z_dim=FLAGS.c_dim * FLAGS.input_height * FLAGS.input_width,
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        f_div=FLAGS.f_div,
        prior=FLAGS.prior,
        lr_decay=FLAGS.lr_decay,
        min_lr=FLAGS.min_lr,
        model_type=FLAGS.model_type,
        log_dir=FLAGS.log_dir,
        alpha=FLAGS.alpha,
        batch_norm_adaptive=FLAGS.batch_norm_adaptive,
        init_type=FLAGS.init_type,
        reg=FLAGS.reg,
        n_critic=FLAGS.n_critic,
        hidden_layers=FLAGS.hidden_layers,
        no_of_layers=FLAGS.no_of_layers,
        like_reg=FLAGS.like_reg,
        df_dim=FLAGS.df_dim)


if __name__ == '__main__':
    main()