import torch
import numpy as np
from torch.nn.functional import kl_div

def Lognorm_KL_loss(pred, mean):
    pred = pred.squeeze(1)
    device = pred.device
    std = torch.std(pred).item()
    size = pred.shape[0]
    pred = pred.sort()[0]

    miu = np.log((mean**2)/(np.sqrt(std+mean**2)))
    sigma = np.sqrt(2*np.log(np.sqrt(std+mean)/mean))

    target = torch.zeros(size).log_normal_(miu, sigma).sort()[0].to(device)
    return kl_div(pred.log(), target)


class Lognorm_KL_Loss():
    def __init__(self, bin=20) -> None:
        self.bin = bin
        self.x_borders = torch.linspace(0,1,bin+1)
    def __call__(self, pred, mean):
        # if isinstance(self.)
        batch_size = pred.__len__()
        std = pred.std()
        miu = torch.log((mean**2)/(torch.sqrt(std+mean**2)))
        sigma = torch.sqrt(torch.log(1+(std/mean**2)))
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

    # def log_norm_prob(self,x,mean,std):
    #     miu = torch.log((mean**2)/(torch.sqrt(std+mean**2)))
    #     sigma = torch.sqrt(torch.log(1+(std/mean**2)))
    #     return (1/(x*sigma*torch.sqrt(2*torch.tensor(torch.pi))))*torch.exp(-1*((torch.log(x)-miu)**2/2*sigma**2))

