import torch.nn as nn
import torch
from math import sqrt, log, ceil
from torch.nn import AvgPool1d, AvgPool2d

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


class AttnPool_2d(nn.Module):
    def __init__(self, kernel_size, dim_in, dim_out, hidden, head, qk_dim, stride=None, padding=0):
        super().__init__()

        # default_args
        self.kernel_size = kernel_size
        self.qk_dim = qk_dim
        self.dim_out = dim_out
        self.head = head
        if stride != None:
            self.stride = stride
        else:
            self.stride = 1

        self.expand = nn.Linear(dim_out, hidden)
        self.restore = nn.Linear(hidden, dim_out)
        self.WQ = nn.Linear(dim_in, head*qk_dim, bias=False)
        self.WK = nn.Linear(dim_in, head*qk_dim, bias=False)
        self.WV = nn.Linear(dim_in, head*qk_dim, bias=False)
        self.WZ = nn.Linear(head*qk_dim, dim_out)
        
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):    # input: [batch, row, col, dim]
        b,r,c,d = x.shape
        steps_row = (r - self.kernel_size +2*0) // self.stride + 1
        steps_col = (c - self.kernel_size +2*0) // self.stride + 1

        output = torch.zeros(b,steps_row,steps_col,self.dim_out).to(x.device)
        for row in range(steps_row):
            for col in range(steps_col):
                base_row = row*self.stride
                base_col = col*self.stride
                data = x[:,base_row:base_row+self.stride,base_col:base_col+self.stride,:]   # [b,stride,stride,d]
                data = data.reshape(b, self.stride**2, d)   
                q = self.WQ(data.mean(dim=1)).reshape(b, self.head, self.qk_dim).unsqueeze(0).unsqueeze(0)  # [1, 1, b, h, qk_dim]
                k = self.WK(data.reshape(b*self.stride**2,d)).reshape(b, self.stride**2, self.head, self.qk_dim).transpose(0,1)   # [l,b,h,qk_dim]
                v = self.WV(data).reshape(b,self.stride**2,self.head,self.qk_dim)  # [b,l,h,qk_dim]
                qk = self.soft((q*k).sum(dim=4) / sqrt(self.qk_dim))    # [1, len(softmax), batch, heads] 
                z = torch.einsum('lsbh,lsbhd->blhd', qk, v.permute(1,0,2,3).unsqueeze(0)).squeeze(1).reshape(b,self.head*self.qk_dim)   # [batch, heads*input_dim]
                z = self.WZ(z)   # [b, dim_out]
                h = torch.relu(self.expand(z))
                data = torch.relu(self.restore(h))
                output[:,row,col,:] = data

        return output