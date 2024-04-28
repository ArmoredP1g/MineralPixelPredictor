import torch.nn as nn
import torch
from torch.nn.functional import pad
from torch.nn.functional import leaky_relu
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
        h = self.model.conv_layer4_s5.register_forward_hook(self.save_feature)
        h1 = self.model.conv_layer4_s5.register_backward_hook(self.save_gradient)

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


class Conv_Raw(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(2,2)
        self.conv_layer1_s5 = nn.Conv1d(1,10,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer1_s9 = nn.Conv1d(1,10,9,1,bias=True, padding=4, padding_mode='reflect')
        self.conv_layer1_s13 = nn.Conv1d(1,10,13,1,bias=True, padding=6, padding_mode='reflect')

        self.conv_layer2_s5 = nn.Conv1d(30,20,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer3_s5 = nn.Conv1d(20,10,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer4_s5 = nn.Conv1d(10,1,5,1,bias=True, padding=2, padding_mode='reflect')

        # layer_norm
        self.ln1 = nn.LayerNorm(10)
        

        self.output_layer = nn.Linear(21,1)

    def forward(self, x):
        if norm_type == 'z_score':
            x = z_score_normalization(x)
        elif norm_type == 'min_max':
            x = min_max_normalization(x)
        x1 = torch.relu(self.conv_layer1_s5(x))
        x2 = torch.relu(self.conv_layer1_s9(x))
        x3 = torch.relu(self.conv_layer1_s13(x))
        x = torch.cat([x1,x2,x3], dim=1)    # [batch, 30, 168]
        x = torch.relu(self.conv_layer2_s5(x))
        x = self.pool(x)                    # [batch, 20, 84]
        x = torch.relu(self.conv_layer3_s5(x))
        x = self.pool(x)                    # [batch, 10, 42]
        x = torch.relu(self.conv_layer4_s5(x))
        x = self.pool(x)                    # [batch, 1, 21]
        x = x.squeeze(1)
        x = self.output_layer(x)
        result = torch.clamp(x, max=1, min=0)
        if self.training:
            transboundary_loss = ((x>1)*(x-1) + (x<0)*(-x)).mean()
            return result, transboundary_loss
        else:
            return result
    
class Conv_Diff(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(2,2)
        self.conv_layer1_s5 = nn.Conv1d(3,10,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer1_s9 = nn.Conv1d(3,10,9,1,bias=True, padding=4, padding_mode='reflect')
        self.conv_layer1_s13 = nn.Conv1d(3,10,13,1,bias=True, padding=6, padding_mode='reflect')

        self.conv_layer2_s5 = nn.Conv1d(30,20,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer3_s5 = nn.Conv1d(20,10,5,1,bias=True, padding=2, padding_mode='reflect')
        self.conv_layer4_s5 = nn.Conv1d(10,3,5,1,bias=True, padding=2, padding_mode='reflect')

        self.output_layer_1 = nn.Linear(63, 42)
        self.output_layer_2 = nn.Linear(42, 21)
        self.output_layer_3 = nn.Linear(21, 1)

        self.ln1 = nn.LayerNorm(168)
        self.ln2 = nn.LayerNorm(84)
        self.ln3 = nn.LayerNorm(42)
        self.ln4 = nn.LayerNorm(21)

    def forward(self, x):


        # x 此时的shape为[batch, 168]
        # 我需要对x进行一阶和二阶差分操作，并将结果和x拼接在通道维度拼接起来，新的x的shape为[batch, 3, 168]
        # 注意差分后的长度要和原始的x一样是168，要在头部补0
        x = self.ln1(x)

        x_diff1 = torch.cat([torch.zeros(x.shape[0],1).to(x.device), x[:,1:]-x[:,:-1]], dim=1)
        x_diff2 = torch.cat([torch.zeros(x.shape[0],1).to(x.device), x_diff1[:,1:]-x_diff1[:,:-1]], dim=1)
        x = torch.stack([x, x_diff1, x_diff2], dim=1)

        x1 = torch.relu(self.conv_layer1_s5(x))
        x2 = torch.relu(self.conv_layer1_s9(x))
        x3 = torch.relu(self.conv_layer1_s13(x))
        x = torch.cat([x1,x2,x3], dim=1)    # [batch, 30, 168]
        x = self.ln1(x)
        x = torch.relu(self.conv_layer2_s5(x))
        x = self.pool(x)                    # [batch, 20, 84]
        x = self.ln2(x)
        x = torch.relu(self.conv_layer3_s5(x))
        x = self.pool(x)                    # [batch, 10, 42]
        x = self.ln3(x)
        x = torch.relu(self.conv_layer4_s5(x))
        x = self.pool(x)                    # [batch, 3, 21]
        x = self.ln4(x)     
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
        DI = x.unsqueeze(1)-x.unsqueeze(2)*0
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)*0
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
        # x[:,-1] = 0
        # x_norm=((x-x.min(dim=1)[0].unsqueeze(1))/(x.max(dim=1)[0]-x.min(dim=1)[0]+1e-5).unsqueeze(1))
        if norm_type == 'z_score':
            x = z_score_normalization(x)
        elif norm_type == 'min_max':
            x = min_max_normalization(x)
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