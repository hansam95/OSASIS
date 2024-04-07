'''
image is load base on the dataloader
'''

import os
import glob
import copy
import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

from diffae.templates_latent import ffhq256_autoenc_latent
from diffae.experiment import LitModel
from utils.map_net import MappingNet, MappingNetTime


class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.imgs = glob.glob(os.path.join(img_dir, '*'))
        self.transform_img = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
            ])      

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_cont_A = Image.open(self.imgs[idx])
        img_cont_A = self.transform_img(img_cont_A)
        return img_cont_A, Path(self.imgs[idx]).name


class DiffFSTester:
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

        self.infer_samp_for = conf._make_diffusion_conf(self.args.T_infer_for).make_sampler()
        self.infer_samp_back = conf._make_diffusion_conf(self.args.T_infer_back).make_sampler()

        self.transform_img = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
            ])        

    def infer_image(self, img_dir, z_style_B):

        test_dataset = TestDataset(img_dir)
        test_loader = DataLoader(test_dataset, batch_size=64)

        for img_cont_A, file_name in test_loader:

            img_cont_A = img_cont_A.to(self.args.device)
            z_cont_A = self.diffae_A.encode(img_cont_A)            
            z_cont_A = z_cont_A.detach().clone()
            xt_cont_A = img_cont_A.clone()          

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
                
                save_dir = os.path.join(self.args.work_dir, 'imgs_test')
                os.makedirs(save_dir, exist_ok=True)
                # save_image(xt_cont_B/2+0.5, os.path.join(save_dir, Path(input_path).name))

                for i, img in enumerate(xt_cont_B):
                    save_image(img/2+0.5, os.path.join(save_dir, file_name[i]))
                    

    def infer_image_all(self):
        ckpt_path = os.path.join(self.args.work_dir, 'ckpt', f'iter_{self.args.n_iter}.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.diffae_B.load_state_dict(ckpt['diffae_B'])
        self.diffae_B = self.diffae_B.to(self.args.device)

        if self.args.map_net:
            self.model_map.load_state_dict(ckpt['model_map'])
            self.model_map = self.model_map.to(self.args.device)

        # style image
        img_style_B = Image.open(os.path.join(self.args.style_domB_dir, self.args.ref_img)).convert('RGB')
        img_style_B = self.transform_img(img_style_B).unsqueeze(0).to(self.args.device)
        z_style_B = self.diffae_A.encode(img_style_B)
        z_style_B = z_style_B.detach().clone()

        self.infer_image(self.args.infer_dir, z_style_B)