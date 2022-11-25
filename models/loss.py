import torch
import numpy as np
from torch.nn.functional import kl_div
from torch.distributions.gamma import Gamma

class Lognorm_KL_Loss():
    def __init__(self, bin=20) -> None:
        self.bin = bin
        self.x_borders = torch.linspace(0,1,bin+1)
    def __call__(self, pred, mean):
        # if isinstance(self.)
        batch_size = pred.__len__()
        var = pred.var()+1e-7
        miu = torch.log((mean**2)/(torch.sqrt(var+mean**2)))
        sigma = torch.sqrt(torch.log(1+(var/mean**2)))
        target = torch.randn(batch_size).log_normal_(miu.item(),sigma.item())

        prob_density = torch.zeros(self.bin).to(pred.device)
        target_prob_density = torch.zeros(self.bin).to(pred.device)

        for idx in range(batch_size):
            for i in range(self.x_borders.__len__()-1):
                if self.x_borders[i] <= pred[idx] < self.x_borders[i+1]:
                    prob_density[i] += 1
                if self.x_borders[i] <= target[idx] < self.x_borders[i+1]:
                    target_prob_density[i] += 1
        
        prob_density /= batch_size
        target_prob_density /= batch_size

        return kl_div((prob_density+1e-5).log(), target_prob_density)

class Gamma_KL_Loss():
    def __init__(self, bin=20) -> None:
        self.bin = bin
        self.x_borders = torch.linspace(0,1,bin+1)
    def __call__(self, pred, mean):
        # if isinstance(self.)
        batch_size = pred.__len__()
        var = pred.var()

        g = Gamma(mean.item()**2/var, mean.item()/var)

        target = g.sample_n(batch_size)

        prob_density = torch.zeros(self.bin).to(pred.device)
        target_prob_density = torch.zeros(self.bin).to(pred.device)

        for idx in range(batch_size):
            for i in range(self.x_borders.__len__()-1):
                if self.x_borders[i] <= pred[idx] < self.x_borders[i+1]:
                    prob_density[i] += 1
                if self.x_borders[i] <= target[idx] < self.x_borders[i+1]:
                    target_prob_density[i] += 1
        
        prob_density /= batch_size
        target_prob_density /= batch_size

        return kl_div((prob_density+1e-5).log(), target_prob_density)
