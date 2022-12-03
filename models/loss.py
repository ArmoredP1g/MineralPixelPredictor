import torch
import numpy as np
from torch.nn.functional import kl_div
from torch.distributions.gamma import Gamma


class Multi_Lognorm_KL_Loss():
    def __init__(self, bin=20) -> None:
        self.bin = bin
        self.x_borders = torch.linspace(0,1,bin+1)
    def __call__(self, pred, var, mean):
        # if isinstance(self.)
        batch_size = pred.__len__()
        label_count = mean.__len__()        
        sub_batch_size = batch_size//label_count

        miu = torch.log((mean**2)/(torch.sqrt(var+mean**2)))
        sigma = torch.sqrt(torch.log(1+(var/mean**2)))

        target = []
        for i in range(label_count):
            target.append(torch.randn(sub_batch_size).log_normal_(miu[i].item(),sigma[i].item()))
        target = torch.cat(target, dim=0)

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


class BatchLognorm_KL_Loss():
    def __init__(self, bin=20) -> None:
        self.bin = bin
        self.x_borders = torch.linspace(0,1,bin+1)
    def __call__(self, pred, mean):
        # pred: [samples, tasks]
        # mean: [samples, tasks]
        samples, tasks = pred.shape
        var = pred.var()+1e-7
        miu = torch.log((mean**2)/(torch.sqrt(var+mean**2)))
        sigma = torch.sqrt(torch.log(1+(var/mean**2)))
        target = torch.cat([torch.randn(samples).log_normal_(miu[i].item(),sigma[i].item()).unsqueeze(0) for i in range(tasks)], dim=0).to(pred.device)  # tasks, samples

        prob_density = torch.zeros(self.bin, tasks).to(pred.device) # 计算样本各个区间的概率
        target_prob_density = torch.zeros(self.bin, tasks).to(pred.device) # 对应方差与真实标签对应的目标分布-对数正态分布

        # 遍历所有随机采样样本
        for idx in range(samples):
            # 遍历bins
            for i in range(self.x_borders.__len__()-1):
                prob_density[i] += (self.x_borders[i] <= pred[idx]) & (pred[idx] < self.x_borders[i+1])
                target_prob_density[i] += (self.x_borders[i].to(pred.device) <= target[:,idx]) & (target[:,idx] < self.x_borders[i+1].to(pred.device))
        
        prob_density /= samples
        target_prob_density /= samples

        return kl_div((prob_density+1e-5).log(), target_prob_density)

