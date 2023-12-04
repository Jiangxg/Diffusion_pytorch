import torch
import argparse
from diffusion import diffusion
from tqdm import tqdm
from utils import *
import random
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.optim import Adam
import tensorboardX

random.seed(10)

parser = argparse.ArgumentParser()
# Add Argument here!
parser.add_argument('-example', type=int, default=100, help="it is an example of an argument")
parser.add_argument('-beta_start', type=float, default=1e-4, help="the lower bound of hyper parameter beta_t")
parser.add_argument('-beta_end', type=float, default=2e-2, help="the upper bound of hyper parameter beta_t")
parser.add_argument('-T', type=int, default=1000, help="The number of steps")
parser.add_argument('-gpu_ids', type=int, default=0, help="index of GPU")
parser.add_argument('-batch_size', type=int, default=128, help="batch size")
parser.add_argument('-data_path', type=str, default="", help="path of data")
parser.add_argument('-train_lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('-train_epoch', type=int, default=100, help="training steps")

opt = parser.parse_args()


def main():
    
    # log the training info
    writer = tensorboardX.SummaryWriter()
    
    # figure out the device
    device = torch.device('cuda:{}'.format(opt.gpu_ids))
    
    # instantiation model
    model = diffusion(opt)


    # load general datasets
    #dataset = cifar10_dataset(opt)
    
    
    # Specifically: load CIFAR-10 dataset    
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    train_data = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download=True)
    #test_data =datasets.CIFAR10(root=os.getcwd(),train=False,transform=transform_test,download=False)

    
    # construct dataloader
    dataloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=opt.batch_size,
                                            shuffle=True)
    
    
    # hyper parameters
    beta_t = torch.linspace(opt.beta_start, opt.beta_end, steps = opt.T).to(device)
    a_t =  1. - beta_t
    a_t_bar = torch.cumprod(a_t, dim=0)


    # construct optimizer
    optimizer = Adam(model.parameters(), lr = opt.train_lr, betas = (0.9, 0.99))
    
    # visualize diffused sample
    data = next(iter(dataloader))
    x0,_ = data
    x0 = normalize_to_neg_one_to_one(x0).to(device)
    epsilon_gt = torch.randn_like(x0)
    xt = math.sqrt(a_t_bar[1000 - 1]) * x0 + math.sqrt(1 - a_t_bar[1000 - 1]) * epsilon_gt
    
    
    img = normalize_to_zero_to_one(x0[0].detach().cpu())
    path = "x0_sample.png"
    save_image(path, img)
    
    
    img = normalize_to_zero_to_one(xt[0].detach().cpu())
    path = "xT_sample.png"
    save_image(path, img)
    
    # train the model
    #for i in tqdm(range(opt.epoch)):
    Iter = 0
    for epoch in tqdm(range(int(opt.train_epoch))):
        for j, data in enumerate(dataloader):
            Iter += 1
            optimizer.zero_grad()
            
            # construct xt from x0
            x0, target = data
            x0 = normalize_to_neg_one_to_one(x0).to(device)
            #print(x0)
            
            epsilon_gt = torch.randn_like(x0)
            
            # generate a random tensor, with a size of 1
            batch_size = x0.size()[0]
            t = torch.randint(1, opt.T + 1, (batch_size,), device=device).long()
            
            xt = torch.sqrt(a_t_bar[t - 1])[:, None, None, None] * x0 + torch.sqrt(1 - a_t_bar[t - 1])[:, None, None, None] * epsilon_gt

            # output estimated noise and calculate loss
            epsilon_est = model(xt, t)
            loss = torch.nn.MSELoss().to(device)(epsilon_est, epsilon_gt)
            
            # log the training info
            writer.add_scalar("Loss/Iterations", loss, Iter)
            
            # backpropagation and update parameters
            loss.backward()
            optimizer.step()

        
        print("loss:{}".format(loss))
        save_path = os.path.join(os.getcwd(), "ddpm_cifar10_{}.pth".format(epoch + 1))
        torch.save(model, save_path)
        
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
