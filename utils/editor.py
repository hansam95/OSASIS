import os
import glob
import copy
import random
import math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from diffae.templates_latent import ffhq256_autoenc_latent
from diffae.templates_cls import ffhq256_autoenc_cls
from diffae.experiment import LitModel
from diffae.experiment_classifier import ClsModel
from diffae.dataset import CelebAttrDataset
from utils.map_net import MappingNet, MappingNetTime


class Editor:
    def __init__(self, args):
        self.args = args

        # set seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # load diffae
        conf = ffhq256_autoenc_latent()
        conf.pretrain.path = os.path.join(self.args.diffae_ckpt, 'ffhq256_autoenc/last.ckpt')
        conf.latent_infer_path = os.path.join(self.args.diffae_ckpt, 'ffhq256_autoenc/latent.pkl')

        model_diffae = LitModel(conf)
        state = torch.load(os.path.join(self.args.diffae_ckpt, f'{conf.name}/last.ckpt'), map_location='cpu')
        model_diffae.load_state_dict(state['state_dict'], strict=False)

        # make diffae for domainA (photo) / freeze
        self.diffae_A = copy.deepcopy(model_diffae.ema_model)
        self.diffae_A = self.diffae_A.to(self.args.device)
        self.diffae_A.eval()
        self.diffae_A.requires_grad_(False)

        # make diffae for domainB (style) / train 
        self.diffae_B = copy.deepcopy(model_diffae.ema_model)
        self.diffae_B = self.diffae_B.to(self.args.device)
        self.diffae_B.eval()
        self.diffae_B.requires_grad_(False)

        # mapping net
        if self.args.map_net:
            if self.args.map_time:
                self.model_map = MappingNetTime().to(args.device)
            else:
                self.model_map = MappingNet().to(args.device)

        # load classifier
        cls_conf = ffhq256_autoenc_cls()
        cls_conf.pretrain.path = os.path.join(self.args.diffae_ckpt, 'ffhq256_autoenc/last.ckpt')
        cls_conf.latent_infer_path = os.path.join(self.args.diffae_ckpt, 'ffhq256_autoenc/latent.pkl')
        self.cls_model = ClsModel(cls_conf)
        state = torch.load(os.path.join(self.args.diffae_ckpt, f'{cls_conf.name}/last.ckpt'), map_location='cpu')
        self.cls_model.load_state_dict(state['state_dict'], strict=False)
        self.cls_model.to(args.device)

        self.infer_samp_for = conf._make_diffusion_conf(self.args.T_infer_for).make_sampler()
        self.infer_samp_back = conf._make_diffusion_conf(self.args.T_infer_back).make_sampler()

        self.transform_img = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
            ])        

    def manipulate(self, style, img_dir, z_style_B):
        # imgs = glob.glob(os.path.join(img_dir, '*'))
        imgs = [os.path.join(img_dir, 'Morgan.png')]

        # # inference images
        for input_path in imgs:
            img_cont_A = Image.open(input_path)
            img_cont_A = self.transform_img(img_cont_A).to(self.args.device)
            img_cont_A = img_cont_A.unsqueeze(0)
        
            z_cont_A = self.diffae_A.encode(img_cont_A)
            z_cont_A = z_cont_A.detach().clone()
            xt_cont_A = img_cont_A.clone()

            # manipulate z_sem of content image
            cls_id = CelebAttrDataset.cls_to_id[self.args.attribute]
            z_cont_A = self.cls_model.normalize(z_cont_A)
            z_cont_A = z_cont_A + self.args.lambda_dir * math.sqrt(512) *\
                        F.normalize(self.cls_model.classifier.weight[cls_id][None, :], dim=1)
            z_cont_A = self.cls_model.denormalize(z_cont_A)

            with torch.no_grad():
                # forward ddim (content A)
                forwad_indices = list(range(self.args.T_infer_for))\
                    [:int(self.args.T_infer_for*self.args.t0_ratio)]
                for j in forwad_indices:
                    t = torch.tensor([j]*len(img_cont_A), device=self.args.device)
                    out = self.infer_samp_for.ddim_reverse_sample(self.diffae_A,
                                                                  xt_cont_A,
                                                                  t,
                                                                  model_kwargs={'cond': z_cont_A})
                    xt_cont_A = out['sample']

                xt_cont_B = xt_cont_A.detach().clone()

                # reverse ddim (mix)
                reverse_indices = list(range(self.args.T_infer_back))[::-1]\
                    [int(self.args.T_infer_back*(1-self.args.t0_ratio)):]
                for j in reverse_indices:
                    t = torch.tensor([j]*len(img_cont_A), device=self.args.device)

                    if self.args.map_net:
                        if self.args.map_time:
                            map_cont = self.model_map(img_cont_A, t)
                        else:
                            map_cont = self.model_map(img_cont_A)
                        xt_cont_B = xt_cont_B + self.args.lambda_map*map_cont

                    out = self.infer_samp_back.ddim_sample(self.diffae_B,
                                                           xt_cont_B,
                                                           t,
                                                           model_kwargs={'cond': {'ref':z_style_B, 'input':z_cont_A},
                                                                         'ref_cond_scale': [0, 1, 2, 3],
                                                                         'mix': True})     
                    xt_cont_B = out['sample'].detach().clone()
                
                save_dir = os.path.join(self.args.work_dir, style, Path(img_dir).stem)
                os.makedirs(save_dir, exist_ok=True)
                img_name = f'{Path(input_path).stem}_{self.args.attribute}.png'
                save_image(xt_cont_B/2+0.5, os.path.join(save_dir, img_name))

    # def infer_image_all(self):
    #     step2_dir = os.path.join(os.path.dirname(self.args.work_dir), 'step2')
    #     style_list = os.listdir(step2_dir)

    #     for style in tqdm(style_list):
    #         ckpt_path = os.path.join(step2_dir, style, 'ckpt', f'iter_{self.args.n_iter}.pt')
    #         ckpt = torch.load(ckpt_path, map_location='cpu')
    #         self.diffae_B.load_state_dict(ckpt['diffae_B'])
    #         self.diffae_B = self.diffae_B.to(self.args.device)

    #         if self.args.map_net:
    #             self.model_map.load_state_dict(ckpt['model_map'])
    #             self.model_map = self.model_map.to(self.args.device)

    #         # style image
    #         img_style_B = Image.open(os.path.join(self.args.style_domB_dir, f'{style}.png')).convert('RGB')
    #         img_style_B = self.transform_img(img_style_B).unsqueeze(0).to(self.args.device)
    #         z_style_B = self.diffae_A.encode(img_style_B)
    #         z_style_B = z_style_B.detach().clone()

    #         self.infer_image(style, self.args.infer_norm_dir, z_style_B)
    #         self.infer_image(style, self.args.infer_ood_dir, z_style_B)