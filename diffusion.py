import torch
import torch.nn as nn
from utils import *
from unet import Unet

class diffusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.T = opt.T
        self.beta_t = torch.linspace(opt.beta_start, opt.beta_end, steps = self.T)
        self.Unet = Unet(dim = 64, dim_mults = (1, 2, 4, 8))
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids))
        self.Unet.to(self.device)
    
    def forward(self, xt, t):
        epsilon_est = self.Unet(xt, t)
        return epsilon_est
    
        
        
        
    
