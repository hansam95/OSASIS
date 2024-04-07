import os
import json
import argparse
from pathlib import Path

def make_args():
    parser = argparse.ArgumentParser()
    # directory
    parser.add_argument('--diffae_ckpt', default='diffae/checkpoints', type=str, help='diffusion autoencoder checkpoints')
    parser.add_argument('--style_domA_dir', default='imgs_style_domA', type=str, help='style images (domain A)')
    parser.add_argument('--style_domB_dir', default='imgs_style_domB', type=str, help='style images (domain B)')
    parser.add_argument('--infer_dir', default='imgs_input_domA', type=str, help='input images (domain A)')
    # data
    parser.add_argument('--ref_img', default='digital_painting_jing.png', type=str, help='reference image file name')
    parser.add_argument('--work_dir', default='exp0', type=str, help='experiment working directory')
    # train
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n_iter', default=200, type=int, help='training iteration')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    # reverse process
    parser.add_argument('--T_train_for', default=50, type=int, help='forward timesteps during training')
    parser.add_argument('--T_train_back', default=20, type=int, help='backward timesteps during training')
    parser.add_argument('--T_infer_for', default=100, type=int, help='forward timesteps during inference')
    parser.add_argument('--T_infer_back', default=50, type=int, help='backward timesteps during inference')
    parser.add_argument('--T_latent', default=200, type=int, help='latent ddim teimsteps')
    parser.add_argument('--t0_ratio', default=0.5, type=float, help='return step ratio')
    # loss coefficients
    parser.add_argument('--cross_dom', default=1, type=float, help='cross domain loss')
    parser.add_argument('--in_dom', default=0.5, type=float, help='in domain loss')
    parser.add_argument('--recon_clip', default=30, type=float, help='clip reconstruction loss')
    parser.add_argument('--recon_l1', default=10, type=float, help='l1 reconstruction')
    parser.add_argument('--recon_lpips', default=10, type=float, help='lpips reconstruction')
    # utils
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--train_img_freq', default=50, type=int)
    parser.add_argument('--ckpt_freq', default=200, type=int)
    parser.add_argument('--train', action='store_true', help='train flag')
    # mapping net
    parser.add_argument('--map_net', action='store_true', help='using mappingnet')
    parser.add_argument('--map_time', action='store_true', help='using mappingnet time')
    parser.add_argument('--lambda_map', default=0.1, type=float, help='weigth of mapping net output')

    args = parser.parse_args()

    args.device = 'cuda'
    args.ref_img_name = Path(args.ref_img).stem

    if args.train:
        os.makedirs(args.work_dir, exist_ok=True)
        with open(os.path.join(args.work_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return args

