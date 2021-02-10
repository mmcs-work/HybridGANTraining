from run import run
from generate import generate
import argparse
import os


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="hybrid", help='hybrid | adv | mle')

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--start_epoch', type=int, default=0, help='continue training')
    parser.add_argument('--root_dir', type=str, default="./data")

    parser.add_argument('--dataset', type=str, default="mnist", help='cifar10 | mnist')
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--checkpoint_dir', default='./flow_gan', help='folder to output images and model checkpoints')
    parser.add_argument('--log_path', default='./flow_gan.log', help='path to store the log file')
    parser.add_argument('--writer_log', type=str, default='runs')
    parser.add_argument('--sample_dir', default='./flow_gan_sample', help='folder to output images and model checkpoints')
    parser.add_argument('--alpha', type=float, default=1e-7)
    parser.add_argument('--lr_decay', type=float, default=0.99999)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--model_type', type=str, default="real_nvp")
    parser.add_argument('--n_critics', type=int, default=5)
    parser.add_argument('--no_of_layers', type=int, default=5)
    parser.add_argument('--hidden_layers', type=int, default=1000)
    parser.add_argument('--df_dim', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_gp', type=float, default=10.)
    parser.add_argument('--like_reg', type=float, default=10.)
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--generate_sample_dir', default='./flowgan_sampled_images', help='folder to output generated images')
    parser.add_argument('--generate_num_imgs', type=int, default=10000)
    

    FLAGS = parser.parse_args()
    print(FLAGS)
        
    if FLAGS.generate == 0:
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)
        if not os.path.exists(FLAGS.root_dir):
            os.makedirs(FLAGS.root_dir)
            
        run(FLAGS)
        
    else:
        if not os.path.exists('./Generated_Images'):
            os.makedirs('./Generated_Images')
        if not os.path.exists(os.path.join('./Generated_Images', FLAGS.generate_sample_dir)):
            os.makedirs(os.path.join('./Generated_Images', FLAGS.generate_sample_dir))
            
        generate(FLAGS)