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

# 对照模型1


# 对照模型2
class Conv_BiLSTM_regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(1,4,5,1, padding=2, padding_mode='reflect', bias=False)
        self.conv_2 = nn.Conv1d(4,16,5,1, padding=2, padding_mode='reflect', bias=False)
        self.conv_3 = nn.Conv1d(16,64,5,1, padding=2, padding_mode='reflect', bias=False)
        self.avgpool = nn.AvgPool1d(2,2)
        self.bi_lstm = nn.LSTM(input_size=64, hidden_size=48, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True, bias=False)
        self.fc_1 = nn.Linear(96,64)
        self.fc_2 = nn.Linear(64,1)
        self.bn_1 = nn.BatchNorm1d(16)
        self.bn_2 = nn.BatchNorm1d(64)

    def forward(self,x):
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0).unsqueeze(1)    # [batch, 1, 304]
        x = torch.relu(self.conv_1(x))  # [batch, 4, 304]
        x = self.avgpool(x)             # [batch, 4, 152]
        x = torch.relu(self.conv_2(x))  # [batch, 16, 152]
        x = self.bn_1(x)
        x = self.avgpool(x)
        x = torch.relu(self.conv_3(x))  # [batch, 64, 76]
        x = self.bn_2(x)
        x = self.avgpool(x)             # [batch, 64, 38]
        _,(h,_) = self.bi_lstm(x.transpose(1,2))         # [2*2, batch, 48]
        h_forward = h[-2,:,:]
        h_backward = h[-1,:,:]
        h = torch.cat([h_forward, h_backward], dim=1)   # [batch, 96]
        x = torch.relu(self.fc_1(h))
        x = torch.relu(self.fc_2(x))
        return x
        

# 对照模型3
class Spec_attn_regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_1 = ProbSparse_Self_Attention_Block()
        self.attn_2 = ProbSparse_Self_Attention_Block()
        self.attn_3 = ProbSparse_Self_Attention_Block()
        self.attn_decoder = Self_Attention_Decoder()
        self.pool = nn.AvgPool1d(2,2)
        self.MLP1 = nn.Linear(32,24)
        self.MLP2 = nn.Linear(24,1)


        self.conv_1 = nn.Conv1d(1,4,5,1, padding=2, padding_mode='reflect', bias=False)
        self.conv_2 = nn.Conv1d(4,16,5,1, padding=2, padding_mode='reflect', bias=False)
        self.conv_3 = nn.Conv1d(16,32,5,1, padding=2, padding_mode='reflect', bias=False)
    
    def forward(self, x):
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0).unsqueeze(1)    # [batch, 1, 304]
        x = leaky_relu(self.conv_1(x))  # [batch, 4, 304]
        x = self.pool(x)             # [batch, 4, 152]
        x = leaky_relu(self.conv_2(x))  # [batch, 16, 152]
        x = self.pool(x)
        x = leaky_relu(self.conv_3(x))  # [batch, 32, 76]
        x = self.pool(x)             # [batch, 32, 38]
        x += positionalencoding1d(32, 38).transpose(0,1).unsqueeze(0).repeat(b,1,1).to(x.device)
        x = self.attn_1(x.permute(0,2,1)).permute(0,2,1)
        x = self.attn_2(x.permute(0,2,1)).permute(0,2,1)
        x = self.attn_3(x.permute(0,2,1))
        x = torch.relu(self.attn_decoder(x))
        x = torch.relu(self.MLP1(x))
        x = torch.relu(self.MLP2(x))
        x = torch.clamp(x, min=0, max=1)
        return x

        x = x.squeeze(0)
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0).unsqueeze(1)    # [batch, 1, 304]
        x = torch.cat([
                torch.relu(self.conv_9(x)),
                torch.relu(self.conv_13(x)),
                torch.relu(self.conv_17(x)),
                torch.relu(self.conv_25(x)),
                torch.relu(self.conv_33(x)),
                torch.relu(self.conv_41(x))
            ], dim=1)   # [batch, 24, 304]
        x += positionalencoding1d(24, 304).transpose(0,1).unsqueeze(0).repeat(b,1,1).to(x.device)
        x = self.pool(x)   # [batch, 24, 152]
        x = self.pool(self.attn_1(x.permute(0,2,1)).permute(0,2,1))   # [batch, 24, 76]
        x = self.pool(self.attn_2(x.permute(0,2,1)).permute(0,2,1))   # [batch, 24, 38]
        x = self.attn_3(x.permute(0,2,1))   
        x = self.attn_decoder(x)
        x = torch.tanh(self.MLP1(x))
        x = self.MLP2(x)
        x = torch.clamp(x, min=0, max=1)
        return x

    def visualization(self, sum_writer, scale):
        '''
        朝tensorboard输出可视化内容
        '''
        for i in range(8): #  [b, 1, l(softmax), head]
            axexSub_attn = sns.heatmap(self.attn_decoder.attn_map[0,:,:,i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn/head{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下

class Spec_Encoder_Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(1,4,5,1, padding=2, padding_mode='reflect', bias=False)
        self.conv_2 = nn.Conv1d(4,16,5,1, padding=2, padding_mode='reflect', bias=False)
        self.conv_3 = nn.Conv1d(16,64,5,1, padding=2, padding_mode='reflect', bias=False)
        # self.bn_4 = nn.BatchNorm2d(256)
        self.avgpool = nn.AvgPool1d(2,2)
        self.up_sampling = nn.Conv2d(128,256,1,1)
        self.soft = nn.Softmax(dim=3)
        self.weight_mask = nn.Parameter(torch.zeros(8,1,1,38,38)/1e6)   # multiplied with: head(8), batch, row, col, dim

    def forward(self, x):
        x=((x-x.min(dim=1)[0].unsqueeze(1))/(x.max(dim=1)[0]-x.min(dim=1)[0]+1e-5).unsqueeze(1))
        b,_ = x.shape
        x = pad(x, (2,2), "constant", 0).unsqueeze(1)    # [batch, 1, 304]
        x = torch.relu(self.conv_1(x))  # [batch, 4, 304]
        x = self.avgpool(x)             # [batch, 4, 152]
        x = torch.relu(self.conv_2(x))  # [batch, 16, 152]
        x = self.avgpool(x)
        x = torch.relu(self.conv_3(x))  # [batch, 64, 76]
        x = self.avgpool(x)             # [batch, 64, 38]
        DI = x.unsqueeze(2)-x.unsqueeze(3)  # [batch, 64, 38, 38]
        NDI = (x.unsqueeze(2)-x.unsqueeze(3))/(x.unsqueeze(2)+x.unsqueeze(3)+1e-5)
        x = torch.cat([DI, NDI], dim=1)   # [batch, 128, 38, 38]
        x = leaky_relu(self.up_sampling(x)) # [batch, 256, 38, 38]
        del DI, NDI
        b,d,r,c = x.shape
        x = torch.cat(list(x.unsqueeze(0).split(32,2)), dim=0)     # [8, b, 32, 38, 38]
        x = x*self.soft(self.weight_mask.reshape(8,1,1,38*38)).reshape(8,1,1,38,38)  # [8, b, 32, 38, 38]
        x = x.sum(dim=(3,4))   # [8,b,32]
        x = x.transpose(0,1).reshape(b,256) # [batch, 256]
        return torch.relu(x)

class Spec_Encoder_Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_sampling_0 = nn.Conv2d(1024,512,1,1,bias=True)
        self.down_sampling_1 = nn.Conv2d(512,384,1,1,bias=True)
        self.down_sampling_2 = nn.Conv2d(384,256,1,1,bias=True)

        self.soft = nn.Softmax(dim=2)
        self.weight_mask = nn.Parameter(torch.ones(8,1,19,19,1)/1e6)   # multiplied with: head(8), batch, row, col, dim

    def forward(self, x):
        x_norm=((x-x.min(dim=1)[0].unsqueeze(1))/(x.max(dim=1)[0]-x.min(dim=1)[0]+1e-5).unsqueeze(1))

        x = pad(x, (2,2), "constant", 0)    # [batch, 304]
        x_norm = pad(x_norm, (2,2), "constant", 0)

        DI = x.unsqueeze(1)-x.unsqueeze(2)  # [batch, 304, 304]
        NDI = (x.unsqueeze(1)-x.unsqueeze(2))/(x.unsqueeze(1)+x.unsqueeze(2)+1e-5)
        DI_NORM = x_norm.unsqueeze(1)-x_norm.unsqueeze(2)  # [batch, 304, 304]
        NDI_NORM = (x_norm.unsqueeze(1)-x_norm.unsqueeze(2))/(x_norm.unsqueeze(1)+x_norm.unsqueeze(2)+1e-5)

        x = torch.cat([DI.unsqueeze(3), NDI.unsqueeze(3), DI_NORM.unsqueeze(3), NDI_NORM.unsqueeze(3)], dim=3)   # [batch, 304, 304, 4]
        del DI, NDI, DI_NORM, NDI_NORM
        b,r,c,d = x.shape

        # tensor split
        new_tensor = []
        for row in range(16):
            for col in range(16):
                new_tensor.append(x[:,row*19:(row+1)*19,col*19:(col+1)*19,:])

        x = torch.cat(new_tensor, dim=3) # [batch, 19, 19, 1024]

        # branch A
        x = torch.relu(self.down_sampling_0(x.permute(0,3,1,2)))
        x = torch.relu(self.down_sampling_1(x))
        x = torch.relu(self.down_sampling_2(x))
        x = x.permute(0,2,3,1)  # [b, 19, 19, 256]


        x = torch.cat(list(x.unsqueeze(0).split(32,4)), dim=0)     # [8, b, 19, 19, 32]
        x = x*self.soft(self.weight_mask.reshape(8,1,19*19,1)).reshape(8,1,19,19,1)  # [8, b, 19, 19, 32]
        x = x.sum(dim=(2,3))   # [8,b,32]
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
        x = self.ln1(self.fc0(x))
        x = torch.tanh(x)
        x = self.ln2(self.fc1(x))
        x = torch.tanh(x)
        x = self.ln3(self.fc2(x))
        x = torch.tanh(x)
        # x = (torch.tanh(self.fc_out(x)) + 1)*0.5
        x = self.fc_out(x)
        x = torch.clamp(x, min=0, max=1)
        return x


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
        for i in range(8):
            axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(4).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            # axexSub_attn = sns.heatmap(torch.abs(self.encoder.weight_mask.squeeze(1).squeeze(1)).cpu().detach().numpy()[i], cmap="viridis", xticklabels=False, yticklabels=False)
            sum_writer.add_figure('heatmap_attn/head{}'.format(i), axexSub_attn.figure, scale)# 得.figure转一下