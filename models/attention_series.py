import torch.nn as nn
import torch
from torch.nn.functional import pad
from torch.nn.functional import leaky_relu
import numpy as np
from models.SoftPool import SoftPool_1d, SoftPool_2d, AttnPool_2d
from models.ProbSparse_Self_Attention import ProbSparse_Self_Attention_Block
from models.PositionEmbedding import positionalencoding1d, positionalencoding2d
from configs.training_cfg import device
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9 

def smooth(x, scale=9):
    b, l = x.shape
    x = pad(x, (scale,scale), mode='replicate')
    result = torch.zeros(b,l).to(device)
    for i in range(l):
        result[:,i] = torch.mean(x[:,i:i+2*scale+1],dim=1)
    return result

def plt2arr(fig, draw=True):
    if draw:
        fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr


class Spec_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_sampling_1 = nn.Conv2d(512,256,1,1)
        self.down_sampling_2 = nn.Conv2d(256,128,1,1)
        self.masking_1 = nn.Conv2d(128,64,1,1)
        self.masking_2 = nn.Conv2d(64,4,1,1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0)    # [batch, 304]
        DI = x.unsqueeze(1)-x.unsqueeze(2)  # [batch, 304, 304]
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3)], dim=3)   # [batch, 304, 304, 2]
        del DI, NDI
        b,r,c,d = x.shape

        # tensor split
        new_tensor = []
        for row in range(16):
            for col in range(16):
                new_tensor.append(x[:,row*19:(row+1)*19,col*19:(col+1)*19,:])
        x = torch.cat(new_tensor, dim=3) # [batch, 19, 19, 512]
        x = leaky_relu(self.down_sampling_1(x.permute(0,3,1,2)))   # [batch, 19, 19, 256]
        x = leaky_relu(self.down_sampling_2(x)).permute(0,2,3,1)   # [batch, 19, 19, 128]

        mask = x + positionalencoding2d(128,19,19).permute(1,2,0).unsqueeze(0).repeat(b,1,1,1).to(x.device)
        mask = torch.relu(self.masking_1(x.permute(0,3,1,2)))
        mask = torch.relu(self.masking_2(mask).permute(0,2,3,1)).permute(3,0,1,2).unsqueeze(4) # [4, batch, 19, 19, 1]
        mask = self.soft(mask.reshape(4,b,19*19,1)).reshape(4,b,19,19,1)

        # 可视化
        self.weight_mask = mask[:,0,:,:,:].unsqueeze(1)

        x = torch.cat(list(x.unsqueeze(0).split(32,4)), dim=0)     # [4, b, 19, 19, 32]
        x = x*mask
        x = x.sum(dim=(2,3))   # [8,b,32]
        x = x.transpose(0,1).reshape(b,128) # [batch, 128]
        return x

class Spec_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128,96)
        self.fc2 = nn.Linear(96,64)
        self.fc_out = nn.Linear(64,1)

    def forward(self, x):
        x = self.dropout(x)
        x = leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = leaky_relu(self.fc2(x))
        # x = torch.sigmoid(self.fc_out(x))
        x = self.fc_out(x)
        x = torch.clamp(x, min=0, max=1)
        return x


class Grade_regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Spec_Encoder()
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
        for i in range(4): 
            axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn/head{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下