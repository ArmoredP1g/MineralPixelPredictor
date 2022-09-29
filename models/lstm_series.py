import torch.nn as nn
import torch

class lstm_single_variable(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=16, 
                            batch_first=True, 
                            num_layers=2,
                            dropout=0.1,
                            bidirectional=True)
        self.linear = nn.Sequential(
                    nn.Linear(32,25),
                    nn.ELU(),
                    nn.Linear(25,20),
                    nn.ELU()
                )

        self.weight_init()


    def weight_init(self):
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, x):
        if self.norm:
            x /= (x.max()-x.min())
        o, (h,c) = self.lstm(x.unsqueeze(2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))


class lstm_multi_variable(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=16, 
                            batch_first=True, 
                            num_layers=2,
                            dropout=0.1,
                            bidirectional=True)
        self.linear = nn.Sequential(
                    nn.ELU(),
                    nn.Linear(32,25),
                    nn.ELU(),
                    nn.Linear(25,20),
                    nn.ELU()
                )

        self.weight_init()


    def weight_init(self):
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, x):
        o, (h,c) = self.lstm(x.unsqueeze(2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))