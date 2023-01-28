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

class Spec_Encoder_Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_sampling_0 = nn.Conv2d(1176,768,1,1,bias=True)
        self.down_sampling_1 = nn.Conv2d(768,512,1,1,bias=True)
        self.down_sampling_2 = nn.Conv2d(512,256,1,1,bias=True)

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
        DI_NORM = x_norm.unsqueeze(1)-x_norm.unsqueeze(2)  
        NDI_NORM = (x_norm.unsqueeze(1)-x_norm.unsqueeze(2))/(x_norm.unsqueeze(1)+x_norm.unsqueeze(2)+1e-5)
        RI_NORM = x_norm.unsqueeze(1)/(x_norm.unsqueeze(2)+x_norm.unsqueeze(1)+1e-5)

        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3), RI.unsqueeze(3), DI_NORM.unsqueeze(3), NDI_NORM.unsqueeze(3), RI_NORM.unsqueeze(3)], dim=3)  
        del DI, NDI, RI, DI_NORM, NDI_NORM, RI_NORM
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
        self.dropout = nn.Dropout(p=0.2)
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

        # return (torch.tanh(self.fc_out(x)) + 1)*0.5
        return torch.clamp(self.fc_out(x), max=1, min=0)



class Grade_regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Spec_Encoder_Linear()
        self.decoder1 = Spec_Decoder()
        # self.decoder2 = Spec_Decoder()
        # self.decoder3 = Spec_Decoder()
        # self.decoder4 = Spec_Decoder()
        # self.decoder5 = Spec_Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = torch.cat([
            self.decoder1(x)
            # self.decoder2(x),
            # self.decoder3(x),
            # self.decoder4(x),
            # self.decoder5(x),
        ], dim=1)

        return x
    
    # def weight_init(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight,mean=0,std=0.15)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight,mean=0,std=0.15)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    
    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(16):
            axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            # axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(1).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn/head{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下