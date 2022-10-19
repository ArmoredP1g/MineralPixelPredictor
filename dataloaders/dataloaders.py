import pandas as pd
import torch
import h5py
import spectral
from torch.utils.data.dataset import Dataset
from configs.training_cfg import device, hdf5_path

hdf5 = h5py.File(hdf5_path, "r")

class dataset_xiongan(Dataset):
    '''
        雄安新区
        须先生成像素信息pixel_info.csv
    '''
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path+"/pixel_info.csv")
        self.hdr = spectral.envi.open(path+"/XiongAn.hdr")
    
    def __getitem__(self, index):
        x, y, label, name = self.df.loc[index]
        return torch.Tensor(self.hdr[x,y]/10000)[0,0].to(device), label

    def __len__(self):
        return self.df.__len__()


class dataset_iron(Dataset):
    '''
        每次从样本中随机选择若干个点做回归
    '''
    def __init__(self, path_csv, samplepoint):
        '''
        args:
            samplepoint: 每次在单个样品上的随机采样点数
        '''
        super().__init__()
        self.df = pd.read_csv(path_csv)
        self.sample_list = self.df['sample_id'].unique().tolist()
        self.samplepoint = samplepoint
    
    def __getitem__(self, index):
        sample = self.df[self.df['sample_id']==self.sample_list[index]].sample()
        gt = {
            'gt_TFe': float(sample.gt_TFe)/100,
            'gt_SiO2': float(sample.gt_SiO2)/100
        }
        del sample

        id_list = self.df[self.df['sample_id']==self.sample_list[index]].sample(self.samplepoint).data_id.to_list()
        result = []
        for id in id_list:
            result.append(torch.Tensor(hdf5[str(id)][:-2]).unsqueeze(0)) # 做数据集时犯了低级错误，需要再去掉两个元素
        return torch.cat(result,dim=0), gt

    def __len__(self):
        return self.sample_list.__len__()