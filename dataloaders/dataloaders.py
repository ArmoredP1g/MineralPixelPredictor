import pandas as pd
import torch
import spectral
from torch.utils.data.dataset import Dataset
from configs.training_cfg import device

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
        