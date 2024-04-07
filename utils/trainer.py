import os
import copy
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import lpips
import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image

from diffae.templates_latent import ffhq256_autoenc_latent
from diffae.experiment import LitModel
from utils.map_net import MappingNet, MappingNetTime


class DiffFSTrainer:
    def __init__(self, args):
        self.args = args

        # set seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # load model
        self.model_clip, model_diffae, self.conf = self.load_model(diffae_ckpt=self.args.diffae_ckpt)

        # prepare train
        self.diffae_A, self.diffae_B, self.conds_std, self.conds_mean = self.prepare_train(model_diffae)
        if self.args.map_net:
            if self.args.map_time:
                self.model_map = MappingNetTime().to(args.device)
            else:
                self.model_map = MappingNet().to(args.device)

        # optimizer
        self.optim_diffae_B = optim.Adam(self.diffae_B.parameters(), lr=self.args.lr)
        if self.args.map_net:
            self.optim_map = optim.Adam(self.model_map.parameters(), lr=1e-4)

        # diffae sampler
        self.train_samp_for = self.conf._make_diffusion_conf(self.args.T_train_for).make_sampler()
        self.train_samp_back = self.conf._make_diffusion_conf(self.args.T_train_back).make_sampler()
        self.latent_samp = self.conf._make_latent_diffusion_conf(self.args.T_latent).make_sampler()    
        self.infer_samp_for = self.conf._make_diffusion_conf(self.args.T_infer_for).make_sampler()
        self.infer_samp_back = self.conf._make_diffusion_conf(self.args.T_infer_back).make_sampler()

        # loss
        self.l1_loss = nn.L1Loss()
        self.percept_loss = lpips.LPIPS(net='vgg').to(self.args.device)
        self.cosine_loss = nn.CosineSimilarity()

        # load style image        
        self.transform_img = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
        ])
        img_style_A = Image.open(os.path.join(self.args.style_domA_dir, self.args.ref_img)).convert('RGB')
        img_style_B = Image.open(os.path.join(self.args.style_domB_dir, self.args.ref_img)).convert('RGB')
        self.img_style_A = self.transform_img(img_style_A).unsqueeze(0).to(self.args.device)
        self.img_style_B = self.transform_img(img_style_B).unsqueeze(0).to(self.args.device)

    def load_model(self,
                   clip_encoder='ViT-B/32',
                   diffae_ckpt='diffae/checkpoints'):
        # load clip
        model_clip, _ = clip.load(clip_encoder, device=self.args.device)

        ## face
        ## load diffae
        conf = ffhq256_autoenc_latent()
        conf.pretrain.path = os.path.join(diffae_ckpt, 'ffhq256_autoenc/last.ckpt')
        conf.latent_infer_path = os.path.join(diffae_ckpt, 'ffhq256_autoenc/latent.pkl')

        model_diffae = LitModel(conf)
        state = torch.load(os.path.join(diffae_ckpt, f'{conf.name}/last.ckpt'), map_location='cpu')
        model_diffae.load_state_dict(state['state_dict'], strict=False)

        return model_clip, model_diffae, conf

    def prepare_train(self, model_diffae):
        # make diffae for domainA (photo) / freeze
        diffae_A = copy.deepcopy(model_diffae.ema_model)
        diffae_A = diffae_A.to(self.args.device)
        diffae_A.eval()
        diffae_A.requires_grad_(False)

        # make diffae for domainB (style) / train 
        diffae_B = copy.deepcopy(model_diffae.ema_model)
        diffae_B = diffae_B.to(self.args.device)
        diffae_B.train()
        diffae_B.requires_grad_(True)

        # latent ddim z statistics
        conds_std = model_diffae.conds_std
        conds_mean = model_diffae.conds_mean

        return diffae_A, diffae_B, conds_std, conds_mean

    def compute_loss(self, img_cont_A, x0_pred_cont_B, x0_pred_style_B):
        loss = 0
        loss_dict = {}

        # cross domain loss
        cross_dom_loss = self.compute_cross_dom_loss(img_cont_A, x0_pred_cont_B)
        loss += cross_dom_loss
        loss_dict['cross_dom'] = cross_dom_loss.item()       

        # in domain loss
        in_dom_loss = self.compute_in_dom_loss(img_cont_A, x0_pred_cont_B)
        loss += in_dom_loss
        loss_dict['in_dom'] = in_dom_loss.item()

        # clip embedding reconstruction loss
        recon_clip_loss = self.compute_recon_clip_loss(x0_pred_style_B)
        loss += recon_clip_loss
        loss_dict['recon_clip'] = recon_clip_loss.item()

        # ref image recon loss
        recon_l1_loss = self.compute_recon_l1_loss(x0_pred_style_B)
        loss += recon_l1_loss
        loss_dict['recon_l1'] = recon_l1_loss.item()

        # ref image lpips loss
        recon_lpips_loss = self.compute_recon_lpips_loss(x0_pred_style_B)
        loss += recon_lpips_loss
        loss_dict['recon_lpips'] = recon_lpips_loss.item()

        return loss, loss_dict

    # compute cross domain loss
    def compute_cross_dom_loss(self, img_cont_A, x0_pred_cont_B):
        style_A_clip = self.model_clip.encode_image(F.interpolate(self.img_style_A, (224,224)))
        style_B_clip = self.model_clip.encode_image(F.interpolate(self.img_style_B, (224,224)))
        style_dir = style_B_clip - style_A_clip
        style_dir /= style_dir.clone().norm(dim=-1, keepdim=True)
        style_dir = torch.cat([style_dir]*len(img_cont_A), dim=0)

        cont_A_clip = self.model_clip.encode_image(F.interpolate(img_cont_A, (224,224)))
        cont_B_clip = self.model_clip.encode_image(F.interpolate(x0_pred_cont_B, (224,224)))
        cont_dir = cont_B_clip - cont_A_clip
        cont_dir /= cont_dir.clone().norm(dim=-1, keepdim=True)

        cross_dom_loss = (1- self.cosine_loss(style_dir, cont_dir)).mean()
        cross_dom_loss = cross_dom_loss*self.args.cross_dom
        return cross_dom_loss            
       
    # compute in domain loss
    def compute_in_dom_loss(self, img_cont_A, x0_pred_cont_B):
        style_A_clip = self.model_clip.encode_image(F.interpolate(self.img_style_A, (224,224)))
        cont_A_clip = self.model_clip.encode_image(F.interpolate(img_cont_A, (224,224)))
        A_dir = cont_A_clip - style_A_clip
        A_dir /= A_dir.clone().norm(dim=-1, keepdim=True)

        style_B_clip = self.model_clip.encode_image(F.interpolate(self.img_style_B, (224,224)))
        cont_B_clip = self.model_clip.encode_image(F.interpolate(x0_pred_cont_B, (224,224)))
        B_dir = cont_B_clip - style_B_clip
        B_dir /= B_dir.clone().norm(dim=-1, keepdim=True)

        in_dom_loss = (1- self.cosine_loss(A_dir, B_dir)).mean()
        in_dom_loss = in_dom_loss*self.args.in_dom
        return in_dom_loss

    # compute style image clip embedding reconstruction loss
    def compute_recon_clip_loss(self, x0_pred_style_B):
        x0_pred_style_clip = self.model_clip.encode_image(F.interpolate(x0_pred_style_B, (224,224)))
        x0_pred_style_clip /= x0_pred_style_clip.clone().norm(dim=-1, keepdim=True)
        img_style_B_clip = self.model_clip.encode_image(F.interpolate(self.img_style_B, (224,224)))
        img_style_B_clip /= img_style_B_clip.clone().norm(dim=-1, keepdim=True)

        recon_clip_loss = (1- self.cosine_loss(x0_pred_style_clip, img_style_B_clip)).mean()
        recon_clip_loss = recon_clip_loss*self.args.recon_clip
        return recon_clip_loss

    # compute style image reconstruction loss (l1)
    def compute_recon_l1_loss(self, x0_pred_style_B):
        return self.l1_loss(x0_pred_style_B, self.img_style_B)*self.args.recon_l1

    # style image reconstruction loss (lpips)
    def compute_recon_lpips_loss(self, x0_pred_style_B):
        return self.percept_loss(x0_pred_style_B, self.img_style_B).mean()*self.args.recon_lpips

    def print_loss(self, i, loss, loss_dict):
        print('Iter {} | Loss {:5f} '.format(i+1, loss.item()), end='')
        for k in loss_dict:
            print(f'| {k} {loss_dict[k]:5f} ', end='')
        print('')

    def save_image(self, i, img_cont_A, xt_cont_B):
        imgs = torch.cat([img_cont_A, xt_cont_B], dim=0)
        # img_dir = os.path.join(self.args.work_dir, self.args.ref_img_name, 'imgs_train')
        img_dir = os.path.join(self.args.work_dir, 'imgs_train')
        os.makedirs(img_dir, exist_ok=True)
        save_image(imgs/2+0.5, os.path.join(img_dir, f'iter_{i+1}_contB.png'))

    def save_model(self, i):
        # ckpt_dir = os.path.join(self.args.work_dir, self.args.ref_img_name, 'ckpt')
        ckpt_dir = os.path.join(self.args.work_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        if self.args.map_net:
            content = {
                'iter': i+1,
                'diffae_B': self.diffae_B.state_dict(),
                'model_map': self.model_map.state_dict(),
                'optim_diffae_B': self.optim_diffae_B.state_dict(),
                'optim_map': self.optim_map.state_dict()
            }
        else:
            content = {
                'iter': i+1,
                'diffae_B': self.diffae_B.state_dict(),
                'optim_diffae_B': self.optim_diffae_B.state_dict()
            }

        torch.save(content, os.path.join(ckpt_dir, f'iter_{i+1}.pt'))
    
    def train(self):

        z_style_A = self.diffae_A.encode(self.img_style_A)    
        z_style_A = z_style_A.detach().clone()

        # training loop
        for i in tqdm(range(self.args.n_iter)):

            # generate image
            noise = torch.randn(self.args.batch_size,
                                3,
                                self.conf.img_size,
                                self.conf.img_size,
                                device=self.args.device)
            latent_noise = torch.randn(self.args.batch_size, self.conf.style_ch, device=self.args.device)
            cond = self.latent_samp.sample(
                model=self.diffae_A.latent_net,
                noise=latent_noise,
                clip_denoised=self.conf.latent_clip_sample,
            )

            if self.conf.latent_znormalize:
                cond = cond * self.conds_std.to(self.args.device) + self.conds_mean.to(self.args.device)

            img_cont_A = self.train_samp_back.sample(model=self.diffae_A, noise=noise, cond=cond)

            # encode image
            z_cont_A = self.diffae_A.encode(img_cont_A)
            z_cont_A = z_cont_A.detach().clone()

            xt_style_A = self.img_style_A.clone()
            xt_cont_A = img_cont_A.clone()

            # forward ddim
            with torch.no_grad():
                forward_indices = list(range(self.args.T_train_for))[:int(self.args.T_train_for*self.args.t0_ratio)]
                for j in forward_indices:
                    
                    # style image forward ddim
                    t_style = torch.tensor([j]*len(self.img_style_A), device=self.args.device)
                    out_style = self.train_samp_for.ddim_reverse_sample(self.diffae_A,
                                                                        xt_style_A,
                                                                        t_style,
                                                                        model_kwargs={'cond': z_style_A})
                    xt_style_A = out_style['sample']

                    # content image forward ddim
                    t_cont = torch.tensor([j]*len(img_cont_A), device=self.args.device)
                    out_cont = self.train_samp_for.ddim_reverse_sample(self.diffae_A,
                                                                       xt_cont_A,
                                                                       t_cont,
                                                                       model_kwargs={'cond': z_cont_A})
                    xt_cont_A = out_cont['sample']

            xt_style_B = xt_style_A.detach().clone()
            xt_cont_B = xt_cont_A.detach().clone()

            # backward ddim
            backward_indices = list(range(self.args.T_train_back))[::-1][int(self.args.T_train_back*(1-self.args.t0_ratio)):]
            for j in backward_indices:
                
                # style image backward ddim
                t_style = torch.tensor([j]*len(self.img_style_A), device=self.args.device)

                if self.args.map_net:
                    if self.args.map_time:
                        map_style = self.model_map(self.img_style_A, t_style)
                    else:
                        map_style = self.model_map(self.img_style_A)
                    xt_style_B = xt_style_B + map_style # original code
                    # xt_style_B = xt_style_B + self.args.lambda_map*map_style

                out_style = self.train_samp_back.ddim_sample(self.diffae_B,
                                                             xt_style_B,
                                                             t_style,
                                                             model_kwargs={'cond': z_style_A})
                x0_pred_style_B = out_style['pred_xstart'] 
                
                # content image backward ddim
                t_cont = torch.tensor([j]*len(img_cont_A), device=self.args.device)

                if self.args.map_net:
                    if self.args.map_time:
                        map_cont = self.model_map(img_cont_A, t_cont)
                    else:
                        map_cont = self.model_map(img_cont_A)
                    xt_cont_B = xt_cont_B + self.args.lambda_map*map_cont

                out_cont = self.train_samp_back.ddim_sample(self.diffae_B,
                                                            xt_cont_B,
                                                            t_cont,
                                                            model_kwargs={'cond': z_cont_A})
                x0_pred_cont_B = out_cont['pred_xstart']

                loss, loss_dict = self.compute_loss(img_cont_A, x0_pred_cont_B, x0_pred_style_B)

                self.optim_diffae_B.zero_grad()
                if self.args.map_net:
                    self.optim_map.zero_grad()
                loss.backward()
                self.optim_diffae_B.step()
                if self.args.map_net:
                    self.optim_map.step()

                # update xt
                xt_style_B = out_style['sample'].detach().clone()
                xt_cont_B = out_cont['sample'].detach().clone()

            # print loss
            if (i+1)%self.args.print_freq==0:
                self.print_loss(i, loss, loss_dict)

            # save images
            if (i+1)%self.args.train_img_freq==0:
                self.save_image(i, img_cont_A, xt_cont_B)
                        
            # save model
            if (i+1)%self.args.ckpt_freq==0:
                self.save_model(i)