import torch.nn as nn
import torch

# 暴力全连接
class fc_single_variable(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm
        self.linear = nn.Sequential(
                    nn.Linear(256,128),
                    nn.ELU(),
                    nn.Linear(128,64),
                    nn.ELU(),
                    nn.Linear(64,32),
                    nn.ELU(),
                    nn.Linear(32,20),
                    nn.ELU()
                )

    
    def forward(self, x):
        if self.norm:
            x /= (x.max()-x.min())
        return self.linear(x)


