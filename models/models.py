import torch.nn as nn
import torch
from torch.nn.functional import pad
from scipy.signal import savgol_filter
import numpy as np
# from models.ProbSparse_Self_Attention import ProbSparse_Self_Attention_Block, Self_Attention_Decoder
# from models.PositionEmbedding import positionalencoding1d
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



class AE_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(2,2)
        self.conv_layer1 = nn.Conv1d(1,30,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer2 = nn.Conv1d(30,20,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer3 = nn.Conv1d(20,10,3,1,bias=True, padding=1, padding_mode='reflect')
        self.conv_layer4 = nn.Conv1d(10,3,3,1,bias=True, padding=1, padding_mode='reflect')
        self.dense = nn.Linear(63, 21)

        self.ln1 = nn.LayerNorm(168)
        self.ln2 = nn.LayerNorm(84)
        self.ln3 = nn.LayerNorm(42)

    def forward(self, x):
        # x: [batch, 168]
        x = self.ln1(x)
        x = x.unsqueeze(1)                  # [batch, 1, 168]
        x = torch.relu(self.conv_layer1(x)) # [batch, 30, 168]
        x = torch.relu(self.conv_layer2(x)) # [batch, 20, 168]
        x = self.pool(x)                       # [batch, 20, 84]
        x = self.ln2(x)
        x = torch.relu(self.conv_layer3(x)) # [batch, 10, 84]
        x = self.pool(x)                       # [batch, 10, 42]
        x = self.ln3(x)
        x = torch.relu(self.conv_layer4(x)) # [batch, 3, 42]
        x = self.pool(x)                    # [batch, 3, 21]
        x = x.view(x.shape[0], -1)          # [batch, 63]
        x = self.dense(x)
        return x


class AE_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(21, 63)
        
        self.deconv_layer1 = nn.ConvTranspose1d(3, 10, 3, 1, padding=1)
        self.deconv_layer2 = nn.ConvTranspose1d(10, 20, 3, 1, padding=1)
        self.deconv_layer3 = nn.ConvTranspose1d(20, 30, 5, 1, padding=2)
        self.final_conv = nn.Conv1d(30, 1, 5, 1, padding=2, padding_mode='reflect')

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.ln1 = nn.LayerNorm(42)
        self.ln2 = nn.LayerNorm(84)
        self.ln3 = nn.LayerNorm(168)

    def forward(self, x):
        # x: [batch, 21]
        x = self.dense(x)                   # [batch, 63]
        x = x.view(x.shape[0], 3, 21)       # [batch, 3, 21]
        
        x = self.upsample(x)                # [batch, 3, 42]
        x = torch.relu(self.deconv_layer1(x)) # [batch, 10, 42]
        x = self.ln1(x)
        
        x = self.upsample(x)                # [batch, 10, 84]
        x = torch.relu(self.deconv_layer2(x)) # [batch, 20, 84]
        x = self.ln2(x)

        x = self.upsample(x)                # [batch, 20, 168]
        x = torch.relu(self.deconv_layer3(x)) # [batch, 30, 168]
        
        x = self.final_conv(x)              # [batch, 1, 168]
        x = x.squeeze(1)                    # [batch, 168]
        x = self.ln3(x)
        
        return x


class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Linear(21, 21)
        self.dense2 = nn.Linear(21, 10)
        self.dense3 = nn.Linear(10, 1)

    def forward(self, x):
        # x: [batch, 21]
        x = torch.relu(self.dense1(x))     # [batch, 10]
        x = torch.relu(self.dense2(x))     # [batch, 10]
        x = self.dense3(x)                   # [batch, 1]
        return torch.sigmoid(x)

