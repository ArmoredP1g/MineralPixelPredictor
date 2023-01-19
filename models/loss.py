import torch
import numpy as np
from torch.nn.functional import kl_div
from torch.distributions.gamma import Gamma


class Tail_Lifter():
    # 标签应符合长尾分布时，防止模型过于倾向于输出均值
    def __init__(self, bin=20, top_k=30) -> None:
        self.bin = bin
        self.top_k = top_k
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

        # 获取对数正态分布下、预测结果的top-k个输出值的平均，以此限制模型倾向输出均值
        target = target.topk(self.top_k, sorted=False)[0].mean()   
        pred_ = pred.topk(self.top_k, sorted=False)[0].mean()

        return torch.clamp(target - pred, min=0)

class Lognorm_Loss():

    def __init__(self, bin_size, order=2, expansion=0) -> None:
        self.bin_size = bin_size
        self.order = order
        self.expansion = expansion
    def __call__(self, pred, var, mean):
        # if isinstance(self.)
        var = var * (1+self.expansion)
        batch_size = pred.__len__()
        label_count = mean.__len__()        
        sub_batch_size = batch_size//label_count

        miu = torch.log((mean**2)/(torch.sqrt(var+mean**2)))
        sigma = torch.sqrt(torch.log(1+(var/mean**2)))

        target = []
        for i in range(label_count):
            target.append(torch.randn(sub_batch_size).log_normal_(miu[i].item(),sigma[i].item()))
        target = torch.cat(target, dim=0)

        pred = torch.flatten(pred)
        pred = torch.sort(pred)[0]
        target = torch.sort(target)[0]

        idx = 0
        loss = torch.Tensor([0.]).to(pred.device)
        while idx+self.bin_size < batch_size:
            loss += (pred[idx:idx+self.bin_size].mean() - target[idx:idx+self.bin_size].mean())**self.order
            idx += 1
        loss += (pred[idx:batch_size].mean() - target[idx:batch_size].mean())**self.order
        return loss



#   无法反向传播
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




# 对极差的惩罚，看看这玩意能反向传播不
class Range_Loss:
    def __init__(self, bins=20) -> None:
        self.bins = bins
        self.x_borders = torch.linspace(0,1,bins+1)
        
    def __call__(self, x):
        x = torch.flatten(x)
        samples = x.shape[0]
        prob_density = torch.zeros(self.bins).to(x.device) # 计算样本各个区间的概率

        # 遍历所有随机采样样本
        for idx in range(samples):
            # 遍历bins
            for i in range(self.x_borders.__len__()-1):
                prob_density[i] += (self.x_borders[i] <= x[idx]) & (x[idx] < self.x_borders[i+1])
        
        prob_density /= samples

        return prob_density.max() - prob_density.min()


