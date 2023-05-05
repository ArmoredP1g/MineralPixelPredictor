import pandas as pd
import torch
import h5py
import spectral
from torch.utils.data.dataset import Dataset
from configs.training_cfg import device
import random
import itertools

class dataset_iron_balanced_mixed(Dataset):
    '''
        按照样本标签值出现的频率做概率加权随机采样，缓解样本不平衡的问题
    '''
    def __init__(self, path_csv, path_hdf5, samples=100000, sample_list=False, samplepoint=500, balance=True):
        '''
        args:
            path_csv: ...
            path_hdf5: ...
            sample_list: 可手动选择作为测试集的样本，不指定则会从csv中获取所有的样本
            samplepoint: 每次在单个样品上的随机采样点数
        '''
        super().__init__()
        self.len = samples
        self.hdf5 = None
        self.path_hdf5 = path_hdf5
        self.df = pd.read_csv(path_csv)
        if sample_list == False:
            self.sample_list = self.df['sample_id'].unique().tolist()
        else:
            # 去重
            self.sample_list = list(set(sample_list))

        # 获取sample_list的标签值
        self.label_list = []
        interval_list = [] # 记录标签落在哪个区间范围内

        # 统计标签出现在各个区间段上的频率
        freq = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        for id in self.sample_list:
            label = float(self.df[self.df['sample_id']==id].sample()['gt_TFe']) 
            for i in range(20):
                if i*5 < label <= (i+1)*5:
                    freq[i] += 1
                    interval_list.append(i)
                    break
            self.label_list.append(label)

        freq /= freq.sum()
        # freq = (torch.sqrt(freq / 0.001) + 1) * 0.001 / freq
        self.freq = freq[interval_list] # 采样概率权重

        if balance == False:
            self.freq /= self.freq
        else:
            self.freq = (1 / self.freq**balance)

        self.samplepoint = samplepoint
        self.training_list = torch.multinomial(self.freq, self.len, replacement=True)
    def __getitem__(self, index):
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.path_hdf5, 'r')

        # id_list = torch.multinomial(self.freq, 1)
        gt = self.label_list[self.training_list[index]]/100

        result = []

        dataid_from = self.df[self.df['sample_id']==self.sample_list[self.training_list[index]]].dataid_from.to_list()[0]
        dataid_to = self.df[self.df['sample_id']==self.sample_list[self.training_list[index]]].dataid_to.to_list()[0]
        id_list = random.sample(range(dataid_from,dataid_to),self.samplepoint)

        for id in id_list:
            result.append(torch.Tensor(self.hdf5[str(id)]).unsqueeze(0))
        return torch.cat(result,dim=0), gt

    def __len__(self):
        return self.len


class dataset_multi_labels(Dataset):
    '''
        按照样本标签值出现的频率做概率加权随机采样，缓解样本不平衡的问题
    '''
    def __init__(self, path_csv, path_hdf5, samples=100000, sample_list=False, samplepoint=500):
        '''
        args:
            path_csv: ...
            path_hdf5: ...
            sample_list: 可手动选择作为测试集的样本，不指定则会从csv中获取所有的样本
            samplepoint: 每次在单个样品上的随机采样点数
        '''
        super().__init__()
        self.len = samples
        self.hdf5 = None
        self.path_hdf5 = path_hdf5
        self.df = pd.read_csv(path_csv)
        if sample_list == False:
            self.sample_list = self.df['sample_id'].unique().tolist()
        else:
            # 去重
            self.sample_list = list(set(sample_list))

        # 获取sample_list的标签值
        self.label_list = []

        for id in self.sample_list:
            label_TFe = float(self.df[self.df['sample_id']==id].sample()['gt_TFe'])/100
            label_Fe3 = float(self.df[self.df['sample_id']==id].sample()['gt_Fe3_plus'])/100
            label_SiO2 = float(self.df[self.df['sample_id']==id].sample()['gt_SiO2'])/100
            self.label_list.append({
                "TFe": label_TFe,
                "Fe3": label_Fe3,
                "SiO2": label_SiO2
            })

        self.samplepoint = samplepoint
        self.training_list = torch.multinomial(torch.ones(self.sample_list.__len__()), self.len, replacement=True)

    def __getitem__(self, index):
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.path_hdf5, 'r')

        # id_list = torch.multinomial(self.freq, 1)
        gt = self.label_list[self.training_list[index]]

        result = []

        dataid_from = self.df[self.df['sample_id']==self.sample_list[self.training_list[index]]].dataid_from.to_list()[0]
        dataid_to = self.df[self.df['sample_id']==self.sample_list[self.training_list[index]]].dataid_to.to_list()[0]
        id_list = random.sample(range(dataid_from,dataid_to),self.samplepoint)

        for id in id_list:
            result.append(torch.Tensor(self.hdf5[str(id)]).unsqueeze(0))
        return torch.cat(result,dim=0), gt

    def __len__(self):
        return self.len