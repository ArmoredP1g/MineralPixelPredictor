import torch.nn as nn
import torch
from torch.nn.functional import pad
import numpy as np
from configs.training_cfg import device
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9 


def plt2arr(fig, draw=True):
    if draw:
        fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr

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
        self.fp_NDI = FeaturePicker(256,4,64,32).to(device)
        self.fp_O23 = FeaturePicker(256,4,64,32).to(device)
        self.fp_E = FeaturePicker(256,4,64,32).to(device)
        self.fp_LN = FeaturePicker(256,4,64,32).to(device)
        self.weight = nn.Parameter(torch.randn(4))
        self.fc = nn.Linear(32, 20)
        
    
    def forward(self, x):
        shape = x.size()[0]
        # RI = (x.unsqueeze(1)/(x.unsqueeze(2)+1e-5))
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        O23 = (x.unsqueeze(1)**2 - x.unsqueeze(2)**3)
        E = (torch.exp(x.unsqueeze(1))-torch.exp(x.unsqueeze(2)))
        LN = (torch.log((x.unsqueeze(1)+1)/(x.unsqueeze(2)+1)))
        
        NDI = self.fp_NDI(NDI).unsqueeze(1)
        O23 = self.fp_O23(O23).unsqueeze(1)
        E = self.fp_E(E).unsqueeze(1)
        LN = self.fp_LN(LN).unsqueeze(1)

        emb = torch.cat((NDI,O23,E,LN), dim=1) * self.weight.repeat(shape,1).unsqueeze(2)
        emb = emb.mean(dim=1)
        if torch.isnan(emb).any():
            print("")
        return self.fc(emb)

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(64):
            axexSub_NDI = sns.heatmap(torch.abs(self.fp_NDI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_NDI/channel{}'.format(i), axexSub_NDI.figure, scale)# 得.figure转一下
            axexSub_O23 = sns.heatmap(torch.abs(self.fp_O23.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_O23/channel{}'.format(i), axexSub_O23.figure, scale)# 得.figure转一下
            axexSub_E = sns.heatmap(torch.abs(self.fp_E.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_E/channel{}'.format(i), axexSub_E.figure, scale)# 得.figure转一下
            axexSub_LN = sns.heatmap(torch.abs(self.fp_LN.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_LN/channel{}'.format(i), axexSub_LN.figure, scale)# 得.figure转一下
