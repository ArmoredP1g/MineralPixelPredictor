import torch.nn as nn
import torch
from torch.nn.functional import pad
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

class Infomer_Based_softpool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool1d(2,2)
        self.pe = nn.Linear(4,8)
        self.attn_1 = ProbSparse_Self_Attention_Block(input_dim=8,dim_feedforward=12,sparse=True)
        self.soft_pool_1 = SoftPool_1d(2,8,12,2)
        self.attn_2 = ProbSparse_Self_Attention_Block(input_dim=8,dim_feedforward=12,sparse=True)
        self.soft_pool_2 = SoftPool_1d(2,8,12,2)
        self.attn_3 = ProbSparse_Self_Attention_Block(input_dim=8,dim_feedforward=12,sparse=True)
        self.dim_reduce = nn.Conv1d(8,4,1,1)
        self.fc1 = nn.Linear(37*4,37*2)
        self.fc2 = nn.Linear(37*2,37)
        self.fc_out = nn.Linear(37,1)

    def forward(self, input):
        raw = input # [batch, len]
        normed = input/(input.max()-input.min()+1e-6)
        input = torch.cat([raw.unsqueeze(2), normed.unsqueeze(2)],dim=2)    # [batch, len(296), 2]
        input = self.avgpool(input.transpose(1,2)).transpose(1,2)    # [batch, len(148), 2]
        b,l,d = input.shape
        input = torch.cat([input,positionalencoding1d(2,l).to(input.device).unsqueeze(0).repeat(b,1,1)],dim=2) # [batch, len(148), 4]
        _,l,d = input.shape
        input = self.pe(input.reshape(b*l,d)).reshape(b,l,-1)
        _,_,d = input.shape # [batch, len(148), 8]

        input = self.attn_1(input)
        input = self.soft_pool_1(input)  # [batch, len(74), 8]
        input = self.attn_2(input)
        input = self.soft_pool_2(input)  # [batch, len(37), 8]
        input = self.attn_3(input)

        input = self.dim_reduce(input.transpose(1,2)).transpose(1,2)
        _,l,d = input.shape
        input = torch.relu(self.fc1(input.reshape(b,l*d)))
        input = torch.relu(self.fc2(input))
        return torch.sigmoid(self.fc_out(input))

class Infomer_Based_1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool1d(2,2)
        self.pe = nn.Linear(4,8)
        self.attn_1 = ProbSparse_Self_Attention_Block(input_dim=8,dim_feedforward=12,sparse=True)
        self.attn_2 = ProbSparse_Self_Attention_Block(input_dim=8,dim_feedforward=12,sparse=True)
        self.attn_3 = ProbSparse_Self_Attention_Block(input_dim=8,dim_feedforward=12,sparse=True)
        self.dim_reduce = nn.Conv1d(8,4,1,1)
        self.fc1 = nn.Linear(37*4,37*2)
        self.fc2 = nn.Linear(37*2,37)
        self.fc_out = nn.Linear(37,1)

    def forward(self, input):
        raw = input # [batch, len]
        normed = input/(input.max()-input.min()+1e-6)
        input = torch.cat([raw.unsqueeze(2), normed.unsqueeze(2)],dim=2)    # [batch, len(296), 2]
        input = self.avgpool(input.transpose(1,2)).transpose(1,2)    # [batch, len(148), 2]
        b,l,d = input.shape
        input = torch.cat([input,positionalencoding1d(2,l).to(input.device).unsqueeze(0).repeat(b,1,1)],dim=2) # [batch, len(148), 4]
        _,l,d = input.shape
        input = self.pe(input.reshape(b*l,d)).reshape(b,l,-1)
        _,_,d = input.shape # [batch, len(148), 8]

        input = self.attn_1(input)
        input = self.avgpool(input.transpose(1,2)).transpose(1,2)  # [batch, len(74), 8]
        input = self.attn_2(input)
        input = self.avgpool(input.transpose(1,2)).transpose(1,2)  # [batch, len(37), 8]
        input = self.attn_3(input)

        input = self.dim_reduce(input.transpose(1,2)).transpose(1,2)
        _,l,d = input.shape
        input = torch.relu(self.fc1(input.reshape(b,l*d)))
        input = torch.relu(self.fc2(input))
        return torch.sigmoid(self.fc_out(input))


    def forward(self, x):   # x:[batch, row, col, dim]
        b,r,c,d = x.shape
        step_r = r // self.sub_win_size
        step_c = c // self.sub_win_size

        # Calculate the attention for each subwindows separately
        for sr in range(step_r):
            for sc in range(step_c):
                x[:,sr*self.sub_win_size:(sr+1)*self.sub_win_size,sc*self.sub_win_size:(sc+1)*self.sub_win_size,:] = \
                    self.attn(x[:,sr*self.sub_win_size:(sr+1)*self.sub_win_size,sc*self.sub_win_size:(sc+1)*self.sub_win_size,:])
        
        return x

class real_attn_block(nn.Module):
    def __init__(self, input_dim, qk_dim, dim_feedforward, sparse=False, sub_win_size=4) -> None:
        super().__init__()
        self.sub_win_size = sub_win_size
        self.attn = ProbSparse_Self_Attention_Block(input_dim=input_dim,qk_dim=qk_dim,dim_feedforward=dim_feedforward,sparse=sparse)

    def forward(self, x):   # x:[batch, row, col, dim]
        b,r,c,d = x.shape
        step_r = r // self.sub_win_size
        step_c = c // self.sub_win_size

        # Calculate the attention for each subwindows separately
        for sr in range(step_r):
            for sc in range(step_c):
                x[:,sr*self.sub_win_size:(sr+1)*self.sub_win_size,sc*self.sub_win_size:(sc+1)*self.sub_win_size,:] = \
                    self.attn(x[:,sr*self.sub_win_size:(sr+1)*self.sub_win_size,sc*self.sub_win_size:(sc+1)*self.sub_win_size,:])
        
        return x


class Infomer_Based_2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #   attn_pool_layer 1
        self.attn_pool_1 = AttnPool_2d(4,2,6,6,4,4,4)
        #   attn_layer
        self.attn_1 = ProbSparse_Self_Attention_Block(input_dim=8,qk_dim=8,dim_feedforward=13,sparse=True,sf_q=10)
        # self.attn_2 = ProbSparse_Self_Attention_Block(input_dim=8,qk_dim=8,dim_feedforward=8,sparse=True)

        self.weight_mask = nn.Parameter(torch.randn(6,1,19,19,1))   
        self.avg_pool_1 = nn.AvgPool2d(4,4)
        self.avg_pool_2 = nn.AvgPool2d(16,16)
        self.soft = nn.Softmax(dim=2)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(48,32)
        self.fc2 = nn.Linear(32,24)
        self.fc_out = nn.Linear(24,1)

    def forward(self, x):   # [batch, 300]
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0)    # [batch, 304]
        DI = x.unsqueeze(1)-x.unsqueeze(2)  # [batch, 304, 304]
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3)], dim=3)   # [batch, 304, 304, 2]
        del DI, NDI
        b,r,c,d = x.shape

        x1 = self.avg_pool_1(x.permute(0,3,1,2)).permute(0,2,3,1)     # [batch, 76, 76, 2]
        x2 = self.avg_pool_2(x.permute(0,3,1,2)).permute(0,2,3,1)     # [batch, 19, 19, 2]

        x1 = self.attn_pool_1(x1)    # [batch, 76, 76, 6]
        x = torch.cat([x1,x2], dim=3)     # [batch, 19, 19, 8]
        x += positionalencoding2d(8,19,19).permute(1,2,0).unsqueeze(0).repeat(b,1,1,1).to(x.device)
        # x = self.attn_1(x)
        x = self.attn_1(x)  # [batch, 19, 19, 8]
        x = x.unsqueeze(0)*self.soft(self.weight_mask.reshape(6,1,19*19,1)).reshape(6,1,19,19,1)  # [6, b, 19, 19, 8]
        x = x.sum(dim=(2,3))   # [6,b,8]
        x = x.transpose(0,1).reshape(b,48)
        x = torch.relu(self.fc1(x))   # [batch, 32]
        x = torch.relu(self.fc2(x))   # [batch, 24]
        x = torch.sigmoid(self.fc_out(x))   # [batch, 1]
        return x

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(6):  # 6,1,19,19,1
            axexSub_attn = sns.heatmap(torch.abs(self.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下

class MCV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avg_pool_1 = nn.AvgPool2d(4,4)
        self.attn_pool_1 = AttnPool_2d(4,2,4,6,4,4,4)
        self.weight_mask = nn.Parameter(torch.randn(6,1,19,19,1))   
        self.soft = nn.Softmax(dim=2)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(24,18)
        self.fc_out = nn.Linear(18,1)

    def forward(self, x):
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0)    # [batch, 304]
        DI = x.unsqueeze(1)-x.unsqueeze(2)  # [batch, 304, 304]
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3)], dim=3)   # [batch, 304, 304, 2]
        del DI, NDI
        b,r,c,d = x.shape
        x = self.avg_pool_1(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.attn_pool_1(x)
        x = x.unsqueeze(0)*self.soft(self.weight_mask.reshape(6,1,19*19,1)).reshape(6,1,19,19,1)  # [6, b, 19, 19, 4]
        x = x.sum(dim=(2,3))   # [6,b,4]
        x = x.transpose(0,1).reshape(b,24)
        x = torch.relu(self.fc1(x))   # [batch, 32]
        x = torch.sigmoid(self.fc_out(x))   # [batch, 1]
        return x

class MCV_conv(nn.Module):
    def __init__(self, attn=False, sample_factor=15) -> None:
        super().__init__()
        self.attn_flag = attn
        self.pool = nn.AvgPool2d(2,2)
        self.conv_1 = nn.Conv2d(2,4,5,1,2)  # 152
        self.conv_2 = nn.Conv2d(4,6,5,1,2)  # 76
        self.conv_3 = nn.Conv2d(6,8,5,1,2)  # 38
        self.conv_4 = nn.Conv2d(8,12,5,1,2)  # 19

        if attn:
            self.attn_1 = ProbSparse_Self_Attention_Block(input_dim=8,qk_dim=8,dim_feedforward=13,sparse=True,sf_q=sample_factor)

        self.weight_mask = nn.Parameter(torch.randn(6,1,19,19,1))   
        self.soft = nn.Softmax(dim=2)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(72,48)
        self.fc2 = nn.Linear(48,36)
        self.fc_out = nn.Linear(36,1)

    def forward(self, x):
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0)    # [batch, 304]
        DI = x.unsqueeze(1)-x.unsqueeze(2)  # [batch, 304, 304]
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3)], dim=3)   # [batch, 304, 304, 2]
        del DI, NDI
        b,r,c,d = x.shape
        x = x.permute(0,3,1,2)  # [b,d,r,c]

        x = self.elu(self.conv_1(x))
        x = self.pool(x)
        x = self.elu(self.conv_2(x))
        x = self.pool(x)
        x = self.elu(self.conv_3(x))
        x = self.pool(x)
        x = self.elu(self.conv_4(x))
        x = self.pool(x)
        x = x.permute(0,2,3,1)
        if self.attn_flag:
            x = self.attn_1(x)
        x = x.unsqueeze(0)*self.soft(self.weight_mask.reshape(6,1,19*19,1)).reshape(6,1,19,19,1)  # [6, b, 19, 19, 4]
        x = x.sum(dim=(2,3))   # [6,b,12]
        x = x.transpose(0,1).reshape(b,72)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))
        return x

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(6):  # 6,1,19,19,1
            axexSub_attn = sns.heatmap(torch.abs(self.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下

class MCV_Split_Attn(nn.Module):
    def __init__(self, attn=False, sample_factor=5) -> None:
        super().__init__()
        self.attn_flag = attn
        if attn:
            self.attn_1 = ProbSparse_Self_Attention_Block(input_dim=32,qk_dim=16,dim_feedforward=48,sparse=True,sf_q=sample_factor)
            self.attn_2 = ProbSparse_Self_Attention_Block(input_dim=32,qk_dim=16,dim_feedforward=48,sparse=True,sf_q=sample_factor)

        self.weight_mask = nn.Parameter(torch.randn(4,1,19,19,1))   # multiplied with: head(8), batch, row, col, dim
        self.down_sampling_1 = nn.Conv2d(512,256,1,1)
        self.down_sampling_2 = nn.Conv2d(256,128,1,1)
        self.soft = nn.Softmax(dim=2)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(128,96)
        self.fc2 = nn.Linear(96,64)
        # self.fc3 = nn.Linear(144,96)
        # self.fc4 = nn.Linear(96,64)
        self.fc_out = nn.Linear(64,1)

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
        x = self.elu(self.down_sampling_1(x.permute(0,3,1,2)))   # [batch, 19, 19, 384]
        x = self.elu(self.down_sampling_2(x)).permute(0,2,3,1)   # [batch, 19, 19, 256]

        if self.attn_flag:
            x2 = x
            x2[:,:,:,96:] = self.attn_1(positionalencoding2d(32,19,19).permute(1,2,0).unsqueeze(0).repeat(b,1,1,1).to(x.device) + x[:,:,:,96:])
            x = x2
            del x2
            
        x = torch.cat(list(x.unsqueeze(0).split(32,4)), dim=0)     # [4, b, 19, 19, 32]
        x = x*torch.tanh(self.weight_mask.reshape(4,1,19*19,1)).reshape(4,1,19,19,1)  # [4, b, 19, 19, 32]
        x = x.sum(dim=(2,3))   # [8,b,32]
        x = x.transpose(0,1).reshape(b,128)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight,mean=0,std=0.15)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight,mean=0,std=0.15)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(4): 
            axexSub_attn = sns.heatmap(torch.abs(self.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn/head{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下


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


class feature_conbined_regression(nn.Module):
    def __init__(self, pretrained=False, freeze=True):
        super().__init__()
        self.fp_DI = FeaturePicker(296,4,64,48).to(device)
        self.fp_NDI = FeaturePicker(296,4,64,48).to(device)
        self.fc1 = nn.Linear(48*2, 48)
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
        x = x[:,:-4]
        DI = x.unsqueeze(1)-x.unsqueeze(2)
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        
        DI = self.fp_DI(DI)
        NDI = self.fp_NDI(NDI)

        emb = torch.cat((DI,NDI), dim=1)

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