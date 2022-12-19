import torch.nn as nn
import torch
from math import sqrt, log, ceil

class ProbSparse_Self_Attention_Block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # default_args
        self.args = {
            'input_dim': 32,
            'qk_dim': 24,
            'heads': 8,
            'dim_feedforward': 24,
            'sparse': False,
            'sf_q': 5,   # sampling factor, sf_q*log(len) q_vecs will be selected.
            'sf_k': 5,   # sampling factor for k
            'res': True
        }

        self.args.update(kwargs)

        self.res = self.args['res']
        
        self.WQ = nn.Linear(self.args['input_dim'], self.args['heads']*self.args['qk_dim'], bias=False)
        self.WK = nn.Linear(self.args['input_dim'], self.args['heads']*self.args['qk_dim'], bias=False)
        self.WV = nn.Linear(self.args['input_dim'], self.args['heads']*self.args['qk_dim'], bias=False)
        self.WZ = nn.Linear(self.args['heads']*self.args['qk_dim'], self.args['input_dim'])
        

        self.MLP_1 = nn.Linear(self.args['input_dim'], self.args['dim_feedforward'])
        self.MLP_2 = nn.Linear(self.args['dim_feedforward'], self.args['input_dim'])
        self.layer_norm = nn.LayerNorm(self.args['input_dim'])
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):    # input: [batch, len, dim] / [batch, row, col, dim]
        shape = x.shape
        reshape_flag = False
        b,l,d,r,c = 0,0,0,0,0

        if shape.__len__() == 3:  # 1d input
            b,l,d = shape
        elif shape.__len__() == 4:  # 2d input
            reshape_flag = True
            b,r,c,d = shape
            l = r*c
            x = x.reshape(b,l,d)
            
            
        h = self.args['heads']
        input = x
        sample_q = ceil(self.args['sf_q']*log(l))
        sample_k = ceil(self.args['sf_k']*log(l))
        x = x.reshape(b*l,d)


        # attn 
        if self.args['sparse'] == False:    # classic self-attn
            q = self.WQ(x).reshape(b,l,h,self.args['qk_dim']).permute(1,0,2,3).unsqueeze(1)   # [batch, len, heads, qkv_dim] -> [len, 1, batch, heads, qkv_dim]
            k = self.WK(x).reshape(b,l,h,self.args['qk_dim']).permute(1,0,2,3)   # [batch, len, heads, qkv_dim] -> [len, batch, heads, qkv_dim]
            v = self.WV(x).reshape(b,l,h,self.args['qk_dim'])   # [batch, len, heads, qkv_dim]
            qk = self.soft((q*k).sum(dim=4) / sqrt(self.args['qk_dim']))  # [len, len(softmax), batch, heads] 
            z = torch.einsum('lsbh,lsbhd->blhd', qk, v.permute(1,0,2,3).unsqueeze(0).repeat(l,1,1,1,1)).reshape(b,l,h*self.args['qk_dim'])   # [batch, len, heads*input_dim]
            z = self.WZ(z.reshape(b*l,-1)).reshape(b,l,-1) + input  # residual
            z = self.layer_norm(z)  # 对单个时间步处理
        elif self.args['sparse'] == True:   # sparse self-attn
            q = self.WQ(x).reshape(b,l,h,self.args['qk_dim']).permute(1,0,2,3).unsqueeze(1)   # [batch, len, heads, qkv_dim] -> [len, 1, batch, heads, qkv_dim]
            k = self.WK(x).reshape(b,l,h,self.args['qk_dim']).permute(1,0,2,3)   # [batch, len, heads, qkv_dim] -> [len, batch, heads, qkv_dim]
            v = self.WV(x).reshape(b,l,h,self.args['qk_dim'])   # [batch, len, heads, qkv_dim]
        
            # randomly select sf_k*ln(len) keys to calculate M
            random_keys = k.unsqueeze(1).repeat(1,l,1,1,1)[torch.arange(l).unsqueeze(1), torch.randint(l,(l,sample_k)),:,:,:] # [l,sample_k,b,h,d] 

            # calculate sparsity, pick top sample_q Qs
            sparsity = torch.max((q*random_keys).sum(dim=4), dim=1)[0] - (q*random_keys).sum(dim=4).mean(dim=1)  # [len, batch, heads]
            quiries_idx = sparsity.topk(sample_q,0)[1]  # [sample_q, batch, heads]
            q = q.squeeze(1)[quiries_idx,torch.arange(b)[None,:,None],torch.arange(h)[None,None,:],:].unsqueeze(1)  # [len, 1, batch, heads, qkv_dim] -> [sample_q, 1, batch, heads, qkv_dim]

            # calculate qk,z with selected Qs
            qk = self.soft((q*k).sum(dim=4) / sqrt(self.args['qk_dim']))    # [sample_q, len(softmax), batch, heads]
            z_activate = torch.einsum('slbh,slbhd->bshd', qk, v.permute(1,0,2,3).unsqueeze(0).repeat(sample_q,1,1,1,1)).reshape(b,sample_q,h,d)   # [batch, sample_q, heads, input_dim]

            # unselected part replaced by mean(V)
            z = v.mean(dim=1).unsqueeze(1).repeat(1,l,1,1)  # all assigned with mean(V) [batch, len, heads, input_dim]  # The inactive parts replaced by mean(V)
            z[torch.arange(b)[:,None,None],quiries_idx.permute(1,0,2), torch.arange(h)[None,None,:],:] = z_activate    # replaced by the active part
            z = self.WZ(z.reshape(b*l,-1)).reshape(b,l,-1) + input  # residual
            z = self.layer_norm(z)  # 对单个时间步处理

        # mlp
        h = torch.relu(self.MLP_1(z.reshape(b*l,d)))
        output = self.MLP_2(h).reshape(b,l,-1)
        output = self.layer_norm(output + z)    # residual

        if reshape_flag:
            output = output.reshape(b,r,c,-1)
        return output


class Self_Attention_Decoder(nn.Module):
    '''
    ......
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # default_args
        self.args = {
            'input_dim': 32,
            'output_dim': 32,
            'qk_dim': 24,
            'heads': 8,
            'dim_feedforward': 24
        }

        self.WK = nn.Linear(self.args['input_dim'], self.args['heads']*self.args['qk_dim'], bias=False)
        self.WV = nn.Linear(self.args['input_dim'], self.args['heads']*self.args['qk_dim'], bias=False)
        self.WZ = nn.Linear(self.args['heads']*self.args['qk_dim'], self.args['output_dim'])

        self.quest_token = nn.Parameter(torch.ones(self.args['qk_dim'])/1e5)

        self.MLP_1 = nn.Linear(self.args['input_dim'], self.args['dim_feedforward'])
        self.MLP_2 = nn.Linear(self.args['dim_feedforward'], self.args['input_dim'])
        self.layer_norm = nn.LayerNorm(self.args['input_dim'])
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):   # input: [batch, len, dim]
        input = x
        b,l,d = x.shape
        h = self.args['heads']
        x = x.reshape(b*l,d)
        k = self.WK(x).reshape(b,l,h,self.args['qk_dim']) #  [b, l, head, qk_dim]
        v = self.WV(x).reshape(b,l,h,self.args['qk_dim']) #  [b, l, head, qk_dim]
        qk = self.soft((self.quest_token * k).sum(dim=3)) / sqrt(self.args['qk_dim'])#  [b, l(softmax), head]
        # for visualizition
        self.attn_map = qk.unsqueeze(1).cpu().detach().numpy()

        z = torch.einsum('blh,blhd->blhd', qk, v)   # [b, l, h, d]
        z = z.sum(dim=1).reshape(b, h*self.args['qk_dim']) #  [b, h*qkdim]
        z = self.WZ(z)  # [b, output_dim]
        return z