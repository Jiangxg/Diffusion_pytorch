import torch
import argparse
from diffusion import diffusion
from tqdm import tqdm
from utils import *
import random
import math
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F

random.seed(10)

parser = argparse.ArgumentParser()
# Add Argument here!
parser.add_argument('-beta_start', type=float, default=1e-4, help="the lower bound of hyper parameter beta_t")
parser.add_argument('-beta_end', type=float, default=2e-2, help="the upper bound of hyper parameter beta_t")
parser.add_argument('-T', type=int, default=1000, help="The number of steps")
parser.add_argument('-gpu_ids', type=int, default=0, help="index of GPU")
parser.add_argument('-output_dir', type=str, default="output", help="output dirs")


opt = parser.parse_args()


def main():
    # figure out the device
    device = torch.device('cuda:{}'.format(opt.gpu_ids))
    
    # instantiation model
    model = diffusion(opt)
    
    save_path = os.path.join(os.getcwd(), "ddpm_cifar10_100.pth")
    model = torch.load(save_path, map_location=device)
    model.eval()
    
    
    # hyper parameters
    beta_t = torch.linspace(opt.beta_start, opt.beta_end, steps = opt.T).to(device)
    a_t =  1. - beta_t
    # 推理过程中使用到了a_t_bar[t - 2], 所以这里需要特殊处理，与训练过程中略有不同
    a_t_bar = F.pad(torch.cumprod(a_t, dim=0), (0, 1), value = 1.)


    # sample xt
    xt = torch.randn(1, 3, 32, 32).to(device)

    inference_list = []
    with torch.no_grad():
        for i in tqdm(range(int(opt.T))):
            t = torch.tensor([opt.T - i], device=device).long()    
            epsilon_est = model(xt, t)
            
            # 逆向推理公式1
            #xt = (1. / math.sqrt(a_t[t - 1])) * (xt - epsilon_est * beta_t[t - 1] / (math.sqrt(1 - a_t_bar[t - 1])))
            
            
            # 逆向推理公式2，此公式能方便我们限定x0的边界。优势不明
            x0 = (1. / torch.sqrt(a_t_bar[t - 1])) * (xt - torch.sqrt(1 - a_t_bar[t - 1])[:, None, None, None] * epsilon_est)
            x0 = torch.clamp(x0, min=-1., max=1.)
            #TODO：因为索引t-2的存在，所以需要重新处理一下a_t_bar向量，防止在 t = 1的时候出现逻辑错误
            xt = (math.sqrt(a_t[t - 1]) * (1 - a_t_bar[t - 2]) * xt + math.sqrt(a_t_bar[t - 2]) * beta_t[t - 1] * x0) / (1 - a_t_bar[t - 1]) 
            
            
            # 这部分必不可少！
            # 必须加上一个随机噪声才能出现好结果，为什么？
            z = torch.randn_like(xt)
            xt += math.sqrt(beta_t[t - 1]) * z
            
            
            if (i + 1) % 25 == 0:
                #print("save image after {} iterations".format(i + 1))
                img = normalize_to_zero_to_one(xt[0].cpu())
                filename = "image_{}.png".format(i + 1)
                save_image(filename, img, opt.output_dir)
                inference_list.append(filename)
                
        video_gen(opt.output_dir, inference_list)
                
        
if __name__ == '__main__':
    main()
