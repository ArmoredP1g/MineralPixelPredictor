import torch.nn as nn
import torch
from math import sqrt, log, ceil

class SoftPool_1d(nn.Module):
    def __init__(self, kernel_size, dim, hidden, stride=None, padding=0):
        super().__init__()

        # default_args
        self.kernel_size = kernel_size
        if stride != None:
            self.stride = stride
        else:
            self.stride = 1

        self.expand = nn.Linear(dim, hidden)
        self.restore = nn.Linear(hidden, 1)
        
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):    # input: [batch, len, dim]
        b,l,d = x.shape
        steps = (l - self.kernel_size +2*0) // self.stride + 1

        output = torch.zeros(b,steps,d).to(x.device)
        for i in range(steps):
            base = i*self.stride
            data = x[:,base:base+self.stride,:]
            weight = torch.tanh(self.expand(data.reshape(b*self.stride,d)))
            weight = self.soft(torch.sigmoid(self.restore(weight).reshape(b,self.stride)))   # [b,stride]
            data = (data*weight.unsqueeze(2)).sum(dim=1) #[b,d]
            output[:,i,:] = data

        return output

class SoftPool_2d(nn.Module):
    def __init__(self, kernel_size, dim, hidden, stride=None, padding=0):
        super().__init__()

        # default_args
        self.kernel_size = kernel_size
        if stride != None:
            self.stride = stride
        else:
            self.stride = 1

        self.expand = nn.Linear(dim, hidden)
        self.restore = nn.Linear(hidden, 1)
        
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):    # input: [batch, row, col, dim]
        b,r,c,d = x.shape
        steps_row = (r - self.kernel_size +2*0) // self.stride + 1
        steps_col = (c - self.kernel_size +2*0) // self.stride + 1

        output = torch.zeros(b,steps_row,steps_col,d).to(x.device)
        for row in range(steps_row):
            for col in range(steps_col):
                base_row = row*self.stride
                base_col = col*self.stride
                data = x[:,base_row:base_row+self.stride,base_col:base_col+self.stride,:]   # [b,stride,stride,d]
                weight = torch.tanh(self.expand(data.reshape(b*self.stride*self.stride,d)))
                weight = self.soft(torch.sigmoid(self.restore(weight).reshape(b,self.stride*self.stride)))
                data = (data*weight.reshape(b,self.stride,self.stride).unsqueeze(3)).sum(dim=(1,2))
                output[:,row,col,:] = data

        return output
