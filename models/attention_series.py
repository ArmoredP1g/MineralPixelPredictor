import torch.nn as nn
import torch
from torch.nn.functional import pad
from configs.training_cfg import device

class FeaturePicker(nn.Module):
    def __init__(self,input_size,pool_size,head,emb_size) -> None:
        '''
        args:
            input_size:光谱波长采样点
            pool_size:池化尺寸
            head:获取特征图组数，作为隐藏层
            emb_size:经全脸结后输出嵌入向量维度
        '''
        super().__init__()
        self.feature_map_size = input_size//pool_size
        self.pool = nn.AvgPool2d(pool_size, pool_size)
        self.weight_map = nn.Parameter(torch.randn(head,self.feature_map_size,self.feature_map_size))
        self.bias = nn.Parameter(torch.randn(head))
        self.fc = nn.Linear(head, emb_size)
        self.elu = nn.ELU()

    def forward(self, input):
        batch_size = input.size()[0]
        input = self.pool(input).unsqueeze(1)
        input = (input * self.weight_map.repeat(batch_size,1,1,1)).view(batch_size,-1,self.feature_map_size ** 2)
        input = input.sum(dim=2) + self.bias    # b,head,feature_map_size^2 -> b,head
        input = self.elu(input)
        if torch.isnan(input).any():
            print("")
        return self.elu(self.fc(input))   # b,emb



class feature_analysis(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.fp_RI = FeaturePicker(256,4,64,32).to(device)
        self.fp_DI = FeaturePicker(256,4,64,32).to(device)
        self.fp_NDI = FeaturePicker(256,4,64,32).to(device)
        self.weight = nn.Parameter(torch.randn(3))
        self.fc = nn.Linear(32, 20)
        
    
    def forward(self, x):
        shape = x.size()[0]
        RI = (x.unsqueeze(1)/(x.unsqueeze(2)+1e-5))
        DI = (x.unsqueeze(1)-x.unsqueeze(2))
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        
        RI = self.fp_RI(RI).unsqueeze(1)
        DI = self.fp_RI(DI).unsqueeze(1)
        NDI = self.fp_RI(NDI).unsqueeze(1)

        emb = torch.cat((RI,DI,NDI), dim=1) * self.weight.repeat(shape,1).unsqueeze(2)
        emb = emb.mean(dim=1)
        if torch.isnan(emb).any():
            print("")
        return self.fc(emb)

