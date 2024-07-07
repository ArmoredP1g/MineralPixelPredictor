import torch.nn as nn
import torch
from torch.nn.functional import pad
from scipy.signal import savgol_filter
import numpy as np
# from models.ProbSparse_Self_Attention import ProbSparse_Self_Attention_Block, Self_Attention_Decoder
# from models.PositionEmbedding import positionalencoding1d
from configs.training_cfg import *
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
plt.rcParams['font.size'] = 9 
# plt.rcParams['figure.figsize'] = 20,3


import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个Z分数归一化的函数
def z_score_normalization(wave):
    # 计算均值和标准差
    mean_value = torch.mean(wave)
    std_value = torch.std(wave)
    # 按照公式进行归一化
    wave_normalized = (wave - mean_value) / std_value
    # 返回归一化后的张量
    return wave_normalized

# 定义一个最大最小归一化的函数
def min_max_normalization(wave):
    # 计算最大值和最小值
    max_value = torch.max(wave)
    min_value = torch.min(wave)
    # 按照公式进行归一化
    wave_normalized = (wave - min_value) / (max_value - min_value)
    # 返回归一化后的张量
    return wave_normalized

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None

    def save_gradient(self, module, input, grad):
        self.gradient = grad

    def save_feature(self, module, input, output):
        self.feature = output.detach()

    def __call__(self, x):
        x.requires_grad_()
        self.feature = None
        self.gradient = None

        # 注册hook
        h = self.model.conv_layer4.register_forward_hook(self.save_feature)
        h1 = self.model.conv_layer4.register_backward_hook(self.save_gradient)

        # 前向传播
        output = self.model(x)

        print(output)

        # 反向传播
        output.backward(torch.ones_like(output))

        # 移除hook
        h.remove()
        h1.remove()

        # 计算权重
        weights = F.adaptive_avg_pool1d(self.gradient[0], 1)

        # 计算CAM
        cam = torch.mul(self.feature, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[1],), mode='linear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam, float(output)
    

class MultiScale_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        chn_left = out_channels
        chn_per = out_channels//4
        self.conv2 = nn.Conv1d(in_channels,chn_per,kernel_size+4,1,bias=True, padding=((kernel_size+4)//2), padding_mode='reflect')
        chn_left -= chn_per
        self.conv3 = nn.Conv1d(in_channels,chn_per,kernel_size+8,1,bias=True, padding=((kernel_size+8)//2), padding_mode='reflect')
        chn_left -= chn_per
        self.conv4 = nn.Conv1d(in_channels,chn_per,kernel_size+12,1,bias=True, padding=((kernel_size+12)//2), padding_mode='reflect')
        chn_left -= chn_per
        self.conv1 = nn.Conv1d(in_channels,chn_left,kernel_size,1,bias=True, padding=(kernel_size//2), padding_mode='reflect')

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat([x1,x2,x3,x4], dim=1)


# class ConvPredictor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.pool = nn.AvgPool1d(2,2)
#         # self.conv_layer1 = MultiScale_Conv(3,30,5)
#         self.conv_layer1 = nn.Conv1d(3,30,5,1,bias=True, padding=2, padding_mode='reflect')
#         self.conv_layer2 = nn.Conv1d(30,20,3,1,bias=True, padding=1, padding_mode='reflect')
#         self.conv_layer3 = nn.Conv1d(20,10,3,1,bias=True, padding=1, padding_mode='reflect')
#         self.conv_layer4 = nn.Conv1d(10,3,3,1,bias=True, padding=1, padding_mode='reflect')

#         self.output_layer_1 = nn.Linear(63, 42)
#         self.output_layer_2 = nn.Linear(42, 21)
#         self.output_layer_3 = nn.Linear(21, 1)

#         self.ln1 = nn.LayerNorm(168)
#         self.ln2 = nn.LayerNorm(84)
#         self.ln3 = nn.LayerNorm(42)

#     def forward(self, x):
#         # x 此时的shape为[batch, 168]
#         # 我需要对x进行一阶和二阶差分操作，并将结果和x拼接在通道维度拼接起来，新的x的shape为[batch, 3, 168]
#         # 注意差分后的长度要和原始的x一样是168，要在头部补0

#         x = self.ln1(x)
#         # x_savgol = torch.Tensor(savgol_filter(x.detach().cpu().numpy(), 4, 2)).to(x.device)
#         x_diff1 = torch.cat([torch.zeros(x.shape[0],1).to(x.device), x[:,1:]-x[:,:-1]], dim=1)
#         # x_diff2 = torch.cat([torch.zeros(x.shape[0],1).to(x.device), x_diff1[:,1:]-x_diff1[:,:-1]], dim=1)
#         # x = torch.stack([x, x_diff1, x_diff2], dim=1)
#         x = torch.stack([x, x, x_diff1], dim=1)
#         x = torch.relu(self.conv_layer1(x)) # [batch, 30, 168]
#         x = self.ln1(x)
#         x = torch.relu(self.conv_layer2(x))
#         x = self.pool(x)                    # [batch, 20, 84]
#         x = self.ln2(x)
#         x = torch.relu(self.conv_layer3(x))
#         x = self.pool(x)                    # [batch, 10, 42]
#         x = self.ln3(x)
#         x = torch.relu(self.conv_layer4(x)) # [batch, 3, 42]
#         x = self.pool(x)                    # [batch, 3, 21]
#         x = x.view(x.shape[0], -1)          # [batch, 63]
#         x = torch.relu(self.output_layer_1(x)) # [batch, 42]
#         x = torch.relu(self.output_layer_2(x)) # [batch, 21]
#         x = self.output_layer_3(x)              # [batch, 1]
#         result = torch.clamp(x, max=1, min=0)  # [batch, 1]
#         if self.training:
#             transboundary_loss = ((x>1)*(x-1) + (x<0)*(-x)).mean()
#             return result, transboundary_loss
#         else:
#             return result
        

class ConvPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(2,2)
        self.conv_layer1 = MultiScale_Conv(3,30,5)
        self.conv_layer2 = nn.Conv1d(30,20,3,1,bias=True, padding=1, padding_mode='reflect')
        self.conv_layer3 = nn.Conv1d(20,10,3,1,bias=True, padding=1, padding_mode='reflect')
        self.conv_layer4 = nn.Conv1d(10,3,3,1,bias=True, padding=1, padding_mode='reflect')

        self.output_layer_1 = nn.Linear(63, 42)
        self.output_layer_2 = nn.Linear(42, 21)
        self.output_layer_3 = nn.Linear(21, 1)

        self.ln1 = nn.LayerNorm(168)
        self.ln2 = nn.LayerNorm(84)
        self.ln3 = nn.LayerNorm(42)

    def forward(self, x):
        # x 此时的shape为[batch, 168]
        # 我需要对x进行一阶和二阶差分操作，并将结果和x拼接在通道维度拼接起来，新的x的shape为[batch, 3, 168]
        # 注意差分后的长度要和原始的x一样是168，要在头部补0

        x = self.ln1(x)
        # x_savgol = torch.Tensor(savgol_filter(x.detach().cpu().numpy(), 4, 2)).to(x.device)
        x_diff1 = torch.cat([torch.zeros(x.shape[0],1).to(x.device), x[:,1:]-x[:,:-1]], dim=1)
        # x_diff2 = torch.cat([torch.zeros(x.shape[0],1).to(x.device), x_diff1[:,1:]-x_diff1[:,:-1]], dim=1)
        # x = torch.stack([x, x_diff1, x_diff2], dim=1)
        x = torch.stack([x, x, x_diff1], dim=1)
        x = torch.relu(self.conv_layer1(x)) # [batch, 63, 168]
        x = self.ln1(x)
        x = torch.relu(self.conv_layer2(x))
        x = self.pool(x)                    # [batch, 20, 84]
        x = self.ln2(x)
        x = torch.relu(self.conv_layer3(x))
        x = self.pool(x)                    # [batch, 10, 42]
        x = self.ln3(x)
        x = torch.relu(self.conv_layer4(x)) # [batch, 3, 42]
        x = self.pool(x)                    # [batch, 3, 21]
        x = x.view(x.shape[0], -1)          # [batch, 63]
        x = torch.relu(self.output_layer_1(x))
        x = torch.relu(self.output_layer_2(x))
        x = self.output_layer_3(x)
        result = torch.clamp(x, max=1, min=0)
        if self.training:
            transboundary_loss = ((x>1)*(x-1) + (x<0)*(-x)).mean()
            return result, transboundary_loss
        else:
            return result



    
