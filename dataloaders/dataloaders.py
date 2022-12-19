import pandas as pd
import torch
import h5py
import spectral
from torch.utils.data.dataset import Dataset
from configs.training_cfg import device, hdf5_path
import itertools

class dataset_iron_mixed(Dataset):
    '''
        将所有样本做排列组合
    '''
    def __init__(self, path_csv, path_hdf5, sample_list=False, samplepoint=500):
        '''
        args:
            path_csv: ...
            path_hdf5: ...
            sample_list: 可手动选择作为测试集的样本，不指定则会从csv中获取所有的样本
            samplepoint: 每次在单个样品上的随机采样点数
        '''
        super().__init__()
        # self.hdf5 = h5py.File(path_hdf5, 'r')
        self.path_hdf5 = path_hdf5
        self.df = pd.read_csv(path_csv)
        if sample_list == False:
            self.sample_list = self.df['sample_id'].unique().tolist()
        else:
            self.sample_list = sample_list
        self.samplepoint = samplepoint

        # 所有的排列组合
        self.sample_list = list(itertools.combinations(self.sample_list,5))

    
    def __getitem__(self, index):
        hdf5 = h5py.File(self.path_hdf5, 'r')['data']
        gt = {
            'gt_TFe': [],
            'gt_FeO': [],
            'gt_SiO2': [],
            'gt_F3O4': [],
            'gt_F2O3': []
        }
        id_list = list(self.sample_list[index])
        for sample_id in id_list:
            sample = self.df[self.df['sample_id']==sample_id].sample()
            gt['gt_TFe'].append(float(sample.gt_TFe)/100)
            gt['gt_FeO'].append(float(sample.gt_FeO)/100)
            gt['gt_SiO2'].append(float(sample.gt_SiO2)/100)
            gt['gt_F3O4'].append(float(sample.gt_F3O4)/100)
            gt['gt_F2O3'].append(float(sample.gt_F2O3)/100)

        result = []
        for sample_id in list(id_list):
            id_list = self.df[self.df['sample_id']==sample_id].sample(self.samplepoint).data_id.to_list() 
            for id in id_list:
                result.append(torch.Tensor(hdf5[id]).unsqueeze(0)) # 做数据集时犯了低级错误，需要再去掉两个元素
        return torch.cat(result,dim=0), gt

    def __len__(self):
        return self.sample_list.__len__()


class dataset_iron_balanced_mixed(Dataset):
    '''
        按照样本标签值出现的频率做概率加权随机采样，缓解样本不平衡的问题
    '''
    def __init__(self, path_pickle, path_hdf5, sample_list=False, samplepoint=500, balance=True):
        '''
        args:
            path_pickle: ...
            path_hdf5: ...
            sample_list: 可手动选择作为测试集的样本，不指定则会从csv中获取所有的样本
            samplepoint: 每次在单个样品上的随机采样点数
        '''
        super().__init__()
        self.hdf5 = None
        self.path_hdf5 = path_hdf5
        self.df = pd.read_pickle(path_pickle)
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
            label = float(self.df[self.df['sample_id']==id].sample().gt_TFe)
            for i in range(20):
                if i*5 < label <= (i+1)*5:
                    freq[i] += 1
                    interval_list.append(i)
                    break
            self.label_list.append(label)

        freq /= freq.sum()
        freq = (torch.sqrt(freq / 0.001) + 1) * 0.001 / freq
        self.freq = freq[interval_list] # 采样概率权重

        if balance == False:
            self.freq /= self.freq
        self.samplepoint = samplepoint
    
    def __getitem__(self, index):
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.path_hdf5, 'r')['data']
        gt = {
            'gt_TFe': []
        }

        id_list = torch.multinomial(self.freq, 1)

        for sample_idx in id_list:
            gt['gt_TFe'].append(self.label_list[sample_idx]/100)

        result = []
        for sample_idx in list(id_list):
            id_list = self.df[self.df['sample_id']==self.sample_list[sample_idx]].sample(self.samplepoint).data_id.to_list() 
            for id in id_list:
                result.append(torch.Tensor(self.hdf5[id]).unsqueeze(0)) # 做数据集时犯了低级错误，需要再去掉两个元素
        return torch.cat(result,dim=0), gt

    def __len__(self):
        return int(1e6)
