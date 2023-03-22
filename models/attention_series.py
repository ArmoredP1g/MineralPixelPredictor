import torch.nn as nn
import torch
from torch.nn.functional import pad
from torch.nn.functional import leaky_relu
import numpy as np
from models.ProbSparse_Self_Attention import ProbSparse_Self_Attention_Block, Self_Attention_Decoder
from models.PositionEmbedding import positionalencoding1d
from configs.training_cfg import device
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9 
# plt.rcParams['figure.figsize'] = 20,3

class MSI_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(12,12)

        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(96)
        self.ln3 = nn.LayerNorm(64)

        self.FC_0 = nn.Linear(588,512)
        self.FC_1 = nn.Linear(512,384)
        self.FC_2 = nn.Linear(384,256)

    def forward(self, x):
        b,_ = x.shape
        DI = x.unsqueeze(1)-x.unsqueeze(2) 
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        RI = x.unsqueeze(1)/(x.unsqueeze(2)+x.unsqueeze(1)+1e-5)      # The division operation results in 
                                                                      # a wide range of RI values, +self in the denominator to 
                                                                      # stabilizes the model performance

        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3), RI.unsqueeze(3)], dim=3)  
        del DI, NDI, RI

        x = self.pool(x.permute(0,3,1,2)).permute(0,2,3,1).reshape(b,-1)

        x = torch.relu(self.FC_0(x))
        x = torch.relu(self.FC_1(x))
        x = torch.relu(self.FC_2(x))
        return x

class TSI_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_sampling_0 = nn.Conv2d(588,512,1,1,bias=True)
        self.down_sampling_1 = nn.Conv2d(512,384,1,1,bias=True)
        self.down_sampling_2 = nn.Conv2d(384,256,1,1,bias=True)

        self.soft = nn.Softmax(dim=2)
        self.weight_mask = nn.Parameter(torch.ones(16,1,12,12,1)/1e6)   # multiplied with: head(16), batch, row, col, dim

    def forward(self, x):   # [b, 168]
        x[:,-1] = 0
        x_norm=((x-x.min(dim=1)[0].unsqueeze(1))/(x.max(dim=1)[0]-x.min(dim=1)[0]+1e-5).unsqueeze(1))
        DI = x.unsqueeze(1)-x.unsqueeze(2) 
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        RI = x.unsqueeze(1)/(x.unsqueeze(2)+x.unsqueeze(1)+1e-5)      # The division operation results in 
                                                                      # a wide range of RI values, +self in the denominator to 
                                                                      # stabilizes the model performance

        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3), RI.unsqueeze(3)], dim=3)  
        del DI, NDI, RI
        b,r,c,d = x.shape

        # tensor split
        new_tensor = []
        for row in range(14):
            for col in range(14):
                new_tensor.append(x[:,row*12:(row+1)*12,col*12:(col+1)*12,:])

        x = torch.cat(new_tensor, dim=3) # [batch, 12, 12, 784]

        # branch A
        x = torch.relu(self.down_sampling_0(x.permute(0,3,1,2)))
        x = torch.relu(self.down_sampling_1(x))
        x = torch.relu(self.down_sampling_2(x))
        x = x.permute(0,2,3,1)  # [b, 12, 12, 256]


        x = torch.cat(list(x.unsqueeze(0).split(16,4)), dim=0)     # [16, b, 12, 12, 16]
        x = x*self.soft(self.weight_mask.reshape(16,1,12*12,1)).reshape(16,1,12,12,1)  # [16, b, 12, 12, 16]
        x = x.sum(dim=(2,3))   # [16,b,16]
        x = x.transpose(0,1).reshape(b,256) # [batch, 256]
        return torch.relu(x)

class Spec_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain_mode = False
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(96)
        self.ln3 = nn.LayerNorm(64)
        self.fc0 = nn.Linear(256,128)
        self.fc1 = nn.Linear(128,96)
        self.fc2 = nn.Linear(96,64)
        self.fc_out = nn.Linear(64,1)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = self.ln1(x)
        x = torch.relu(self.fc1(x))
        x = self.ln2(x)
        x = torch.relu(self.fc2(x))
        x = self.ln3(x)
        x = self.fc_out(x)
        result = torch.clamp(x, max=1, min=0)
        if self.training:
            transboundary_loss = ((x>1)*(x-1) + (x<0)*(-x)).mean()
            return result, transboundary_loss
        else:
            return result

class Grade_regressor(nn.Module):
    def __init__(self, encoder='TSI', tasks=1):
        super().__init__()
        self.vis_flag = (encoder=='TSI')
        if encoder == 'TSI':
            self.encoder = TSI_Encoder()
        else:
            self.encoder = MSI_Encoder()
        self.decoders = Spec_Decoder()
        self.decoders = nn.ModuleList([Spec_Decoder() for i in range(tasks)])

    def forward(self, x):
        x = self.encoder(x)

        if self.training:
            result = []
            transboundary_loss = 0
            for m in self.decoders:
                p, loss = m(x)  # p: batch,168
                result.append(p)
                transboundary_loss += loss
            return torch.cat(result, dim=1), transboundary_loss       # result: batch, 168, tasks
        else:
            result = []
            for m in self.decoders:
                result.append(m(x))
            return torch.cat(result, dim=1)

    
    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        if self.vis_flag:
            for i in range(16):
                axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
                # axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(1).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
                sum_writer.add_figure('heatmap_attn/head{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下