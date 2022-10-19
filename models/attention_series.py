import torch.nn as nn
import torch
from torch.nn.functional import pad
import numpy as np
from configs.training_cfg import device
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
        self.fp_DI = FeaturePicker(256,4,64,48).to(device)
        self.fp_NDI = FeaturePicker(256,4,64,48).to(device)
        self.fp_DI_DIFF = FeaturePicker(256,4,64,48).to(device)
        self.fp_NDI_DIFF = FeaturePicker(256,4,64,48).to(device)
        self.weight = nn.Parameter(torch.randn(4))
        self.fc1 = nn.Linear(48, 32)
        self.fc2 = nn.Linear(32, 20)
        self.elu = nn.ELU()
        
    
    def forward(self, x):
        shape = x.size()[0]
        diff = smooth(x.diff(prepend=x[:,0].unsqueeze(1)))
        
        DI = x.unsqueeze(1)-x.unsqueeze(2)
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        DI_DIFF = diff.unsqueeze(1)-diff.unsqueeze(2)
        NDI_DIFF = (diff.unsqueeze(1)-diff.unsqueeze(2))/(diff.unsqueeze(1)+diff.unsqueeze(2)+1e-5)
        
        DI = self.fp_DI(DI).unsqueeze(1)
        NDI = self.fp_NDI(NDI).unsqueeze(1)
        DI_DIFF = self.fp_DI_DIFF(DI_DIFF).unsqueeze(1)
        NDI_DIFF = self.fp_NDI_DIFF(NDI_DIFF).unsqueeze(1)

        emb = torch.cat((DI,NDI,DI_DIFF,NDI_DIFF), dim=1) * self.weight.repeat(shape,1).unsqueeze(2)
        emb = emb.mean(dim=1)
        if torch.isnan(emb).any():
            print("")
        
        emb = self.elu(self.fc1(emb))
        return self.fc2(emb)

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(64):
            axexSub_DI = sns.heatmap(torch.abs(self.fp_DI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_DI/channel{}'.format(i), axexSub_DI.figure, scale)# 得.figure转一下
            axexSub_NDI = sns.heatmap(torch.abs(self.fp_NDI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_NDI/channel{}'.format(i), axexSub_NDI.figure, scale)# 得.figure转一下
            axexSub_DI_DIFF = sns.heatmap(torch.abs(self.fp_DI_DIFF.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_DI_DIFF/channel{}'.format(i), axexSub_DI_DIFF.figure, scale)# 得.figure转一下
            axexSub_NDI_DIFF = sns.heatmap(torch.abs(self.fp_NDI_DIFF.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_NDI_DIFF/channel{}'.format(i), axexSub_NDI_DIFF.figure, scale)# 得.figure转一下


class feature_conbined(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.fp_DI = FeaturePicker(256,4,64,48).to(device)
        self.fp_NDI = FeaturePicker(256,4,64,48).to(device)
        self.fp_DI_DIFF = FeaturePicker(256,4,64,48).to(device)
        self.fc1 = nn.Linear(48*3, 48)
        self.fc2 = nn.Linear(48, 20)
        self.elu = nn.ELU()
        
    
    def forward(self, x):
        diff = smooth(x,scale=2).diff(prepend=x[:,0].unsqueeze(1))
        
        DI = x.unsqueeze(1)-x.unsqueeze(2)
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        DI_DIFF = diff.unsqueeze(1)-diff.unsqueeze(2)
        
        DI = self.fp_DI(DI)
        NDI = self.fp_NDI(NDI)
        DI_DIFF = self.fp_DI_DIFF(DI_DIFF)

        emb = torch.cat((DI,NDI,DI_DIFF), dim=1)

        if torch.isnan(emb).any():
            print("")

        emb = self.elu(self.fc1(emb))
        return self.fc2(emb)

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(64):
            axexSub_DI = sns.heatmap(torch.abs(self.fp_DI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_DI/channel{}'.format(i), axexSub_DI.figure, scale)# 得.figure转一下
            axexSub_NDI = sns.heatmap(torch.abs(self.fp_NDI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_NDI/channel{}'.format(i), axexSub_NDI.figure, scale)# 得.figure转一下
            axexSub_DI_DIFF = sns.heatmap(torch.abs(self.fp_DI_DIFF.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_DI_DIFF/channel{}'.format(i), axexSub_DI_DIFF.figure, scale)# 得.figure转一下


class feature_conbined_regression(nn.Module):
    def __init__(self, pretrained=False, freeze=True):
        super().__init__()
        self.fp_DI = FeaturePicker(296,4,64,48).to(device)
        self.fp_NDI = FeaturePicker(296,4,64,48).to(device)
        self.fp_DI_DIFF = FeaturePicker(296,4,64,48).to(device)
        self.fc1 = nn.Linear(48*3, 48)
        # 加载预训练参数
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = torch.load(pretrained)
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        # 冻结上边部分的参数
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # 重新学习的参数
        self.learnable_1 = nn.Linear(48, 20)
        self.learnable_2 = nn.Linear(20, 1)
        self.elu = nn.ELU()
        
    
    def forward(self, x):
        diff = x.diff(prepend=x[:,0].unsqueeze(1))#就别平滑了，
        
        DI = x.unsqueeze(1)-x.unsqueeze(2)
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        DI_DIFF = diff.unsqueeze(1)-diff.unsqueeze(2)
        
        DI = self.fp_DI(DI)
        NDI = self.fp_NDI(NDI)
        DI_DIFF = self.fp_DI_DIFF(DI_DIFF)

        emb = torch.cat((DI,NDI,DI_DIFF), dim=1)

        if torch.isnan(emb).any():
            print("")

        emb = self.elu(self.fc1(emb))
        emb = self.elu(self.learnable_1(emb))
        return torch.sigmoid(self.learnable_2(emb))

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(64):
            axexSub_DI = sns.heatmap(torch.abs(self.fp_DI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_DI/channel{}'.format(i), axexSub_DI.figure, scale)# 得.figure转一下
            axexSub_NDI = sns.heatmap(torch.abs(self.fp_NDI.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_NDI/channel{}'.format(i), axexSub_NDI.figure, scale)# 得.figure转一下
            axexSub_DI_DIFF = sns.heatmap(torch.abs(self.fp_DI_DIFF.weight_map).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_DI_DIFF/channel{}'.format(i), axexSub_DI_DIFF.figure, scale)# 得.figure转一下