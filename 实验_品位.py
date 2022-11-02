from unicodedata import name
from models.attention_series import Infomer_Based
import h5py
from dataloaders.dataloaders import dataset_iron
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_regression
from configs.training_cfg import device
import torch

if __name__ == "__main__":
    train_list = ['11_A', '11_B', '11_C', '12_A', 
    '12_B', '12_C', '13_A', '13_B', '13_C', '14_A', '14_B', '14_C', 
    '15_A', '15_B', '15_C', '16_A', '16_B', '16_C', '17_A', '17_B', 
    '17_C', '18_A', '18_B', '18_C', '19_A', '19_B', '19_C', '20_A', 
    '20_B', '20_C', '21_A', '21_B', '22_A', '22_B', '23_A', '23_B', 
    '23_C', '24_A', '24_B', '24_C', '25_A', '25_B', '25_C', '26_A', 
    '26_B', '26_C']

    test_list = ['10_A', '10_B', '10_C', '9_A', 
    '9_B', '9_C']

    train_set = dataset_iron("E:\\成像光谱\\spectral_data_winsize9.csv", "E:\\成像光谱\\spectral_data_winsize9.hdf5", train_list, 500)
    # test_set = dataset_iron("E:\\成像光谱\\spectral_data_winsize5.csv", "E:\\成像光谱\\spectral_data_winsize5.hdf5", train_list, 2000)
    # train_set, test_set = random_split(dataset, [dataset.__len__()-2, 2], generator=torch.Generator().manual_seed(42))
    # del dataset

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    
    model = Infomer_Based().to(device)
    train_regression(train_loader, model, 1000000, lr=0.0005, tag="informer", vis=None)


