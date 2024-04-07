"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import glob
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import lpips

import torch as th
import torchvision.transforms as transforms
from torchvision import utils

from P2_weighting.guided_diffusion import dist_util, logger
from P2_weighting.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

"""
Using P2 / rtaio_0.5 / eta 1.0 / respacing 50
"""


def main():
    args = create_argparser().parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    dist_util.setup_dist()
    logger.configure(dir=args.sample_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")

    device = dist_util.dev()
    transform_256 = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
    ])

    # imgs_ref_domainB = glob.glob(f'{args.input_dir}/*')
    imgs_ref_domainB = [f'{args.input_dir}/art_{args.seed}.png']
    percept_loss = lpips.LPIPS(net='vgg').to(device)
    l1 = th.nn.L1Loss(reduction='none')

    # for img_file in tqdm(imgs_ref_domainB):
    #     # file_name = img_file.split('/')[-1]
    # file_name = Path(img_file).name
    
    img_file = os.path.join(args.input_dir, args.img_name)
    img_ref_B = Image.open(img_file).convert('RGB')
    img_ref_B = transform_256(img_ref_B)
    img_ref_B = img_ref_B.unsqueeze(0).to(device)
    img_ref_B_all = img_ref_B.repeat(args.n,1,1,1)

    t_start = int(diffusion.num_timesteps*args.t_start_ratio)
    t_start = th.tensor([t_start], device=device)
    
    # forward DDPM
    xt = diffusion.q_sample(img_ref_B_all.clone(), t_start)

    # reverse DDPM
    indices = list(range(t_start))[::-1]
    for i in indices:
        t = th.tensor([i] * img_ref_B_all.shape[0], device=device)
        with th.no_grad():
            out = diffusion.p_sample(
                model,
                xt,
                t,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )
            xt = out["sample"]

        # # reverse DDIM
        # indices = list(range(t_start))[::-1]
        # for i in indices:
        #     t = th.tensor([i] * img_ref_B_all.shape[0], device=device)
        #     with th.no_grad():
        #         out = diffusion.ddim_sample(
        #             model,
        #             xt,
        #             t,
        #             clip_denoised=True,
        #             denoised_fn=None,
        #             cond_fn=None,
        #             model_kwargs=None,
        #         )
        #         xt = out["sample"]

        # compute loss
        # from torchvision.utils import save_image
        # save_image(xt/2+0.5, os.path.join('step1_tmp', file_name))
        l1_loss = l1(xt, img_ref_B.repeat(int(args.n),1,1,1))
        l1_loss = l1_loss.mean(dim=(1,2,3))
        lpips_loss = percept_loss(xt, img_ref_B.repeat(int(args.n),1,1,1)).squeeze()
        loss = 10*l1_loss + lpips_loss

        # pick best image
        img_idx = th.argmin(loss)
        img_ref_A = xt[img_idx]
        
        os.makedirs(args.sample_dir, exist_ok=True)
        utils.save_image(img_ref_A/2+0.5, os.path.join(args.sample_dir, args.img_name))

def create_argparser():
    defaults = dict(
        model_path="",
        input_dir="",
        sample_dir="",
        img_name="",
        n=1,
        t_start_ratio=0.5,
        eta=1.0,
        seed=1
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()