import torch
import torch.nn as nn
from diffae.model.nn import timestep_embedding

class MappingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=1),

            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=1),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MappingNetTime(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_layer = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1),
        )

        self.block1 = MappingNetTimeBlock(
            in_ch = 128,
            out_ch = 256,
            t_emb_dim = 512
        )

        self.block2 = MappingNetTimeBlock(
            in_ch = 256,
            out_ch = 128,
            t_emb_dim = 512
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1)
        )
      
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, t):
        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_embed(t_emb)

        x = self.in_layer(x)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        x = self.out_layer(x)

        return x


class MappingNetTimeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()

        self.pre_layer = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GroupNorm(32, out_ch),
        )

        self.post_layer = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch*2)
        )
    
    def forward(self, x, t_emb, scale_bias:float=1):

        t_emb = self.emb_layer(t_emb)
        
        # match shape 
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]

        scale, shift = torch.chunk(t_emb, 2, dim=1)

        x = self.pre_layer(x)
        x = x * (scale_bias + scale)
        x = x + shift
        x = self.post_layer(x)

        return x