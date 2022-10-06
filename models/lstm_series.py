import torch.nn as nn
import torch
from torch.nn.functional import pad
from configs.training_cfg import device

class multi_scale_smoothing(nn.Module):
    def __init__(self, scale=8):
        super().__init__()
        self.scale = scale
        self.front = scale-1
    
    def forward(self, x):
        b, l = x.shape
        front = self.scale-1
        x = pad(x, (front,front), mode='replicate')
        result = torch.zeros(b,l,self.scale).to(device)
        for i in range(l):
            for j in range(self.scale):
                result[:,i,j] = torch.mean(x[:,i+front-j:i+front+j+1], dim=1)
        return result

class lstm_single_variable(nn.Module):
    def __init__(self, norm=False, diff=False):
        super().__init__()
        self.diff = diff
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
        if self.diff:
            x = x.diff()
        o, (h,c) = self.lstm(x.unsqueeze(2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))

class lstm_single_variable_combined(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm
        self.lstm = nn.LSTM(input_size=2, 
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
        o, (h,c) = self.lstm(torch.cat((x[:,1:].unsqueeze(2), x.diff().unsqueeze(2)), dim=2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))

class lstm_smooth_combined(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm
        self.smooth = multi_scale_smoothing(9).to(device)
        self.lstm = nn.LSTM(input_size=10, 
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
        o, (h,c) = self.lstm(torch.cat((x[:,1:].unsqueeze(2), self.smooth(x.diff())), dim=2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))

class lstm_multi_variable(nn.Module):
    def __init__(self, diff=False):
        super().__init__()
        self.diff = diff
        self.smooth = multi_scale_smoothing(8).to(device)
        self.lstm = nn.LSTM(input_size=8, 
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
        if self.diff:
            x = x.diff()
        x = self.smooth(x)
        o, (h,c) = self.lstm(x)
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))



class lstm_multiscale_conv(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm
        self.smooth = multi_scale_smoothing(9).to(device)
        self.lstm = nn.LSTM(input_size=10, 
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
        o, (h,c) = self.lstm(torch.cat((x[:,1:].unsqueeze(2), self.smooth(x.diff())), dim=2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))

class lstm_singlescale_conv(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm
        self.smooth = multi_scale_smoothing(9).to(device)
        self.lstm = nn.LSTM(input_size=10, 
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
        o, (h,c) = self.lstm(torch.cat((x[:,1:].unsqueeze(2), self.smooth(x.diff())), dim=2))
        return self.linear(torch.cat((h[-1,:,:], h[-2,:,:]), 1))