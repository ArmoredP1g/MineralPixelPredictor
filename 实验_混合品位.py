from unicodedata import name
from models.attention_series import Grade_regressor
import h5py
from dataloaders.dataloaders import dataset_iron_balanced_mixed
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_regression_mix
from configs.training_cfg import device
import torch

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    # # 全部样本
    train_list = [
        '1_A','1_B','1_C','1_D','12_A','12_B','12_C','12_D','13_A','13_B','13_C','13_D','14_A',
        '14_B','14_C','14_D','15_A','15_B','15_C','15_D','16_A','16_B','16_C','16_D','17_A','17_B','17_C','17_D',
        '18_A','18_B','18_C','18_D','19_A','19_B','19_C','19_D','2_A','2_B','2_C','2_D','20_A','20_B','20_C',
        '20_D','21_A','21_B','21_C','21_D','22_A','22_B','22_C','22_D','23_A','23_B','23_C','23_D','24_A','24_B','24_C',
        '24_D','25_A','25_B','25_C','25_D','26_A','26_B','26_C','26_D','27_A','27_B','3_A','3_B','3_C','3_D',
        '4_A','4_B','4_C','4_D','5_A','5_B','5_C','5_D','6_A','6_B','6_C','6_D','7_A','7_B','7_C','7_D',
        '8_A','8_B','8_C','8_D','9_A','9_B','9_C','9_D']        # '10_A','10_B','10_C','10_D','11_A','11_B','11_C','11_D',

    #for test: 9,14,17



    train_set = dataset_iron_balanced_mixed("D:\\可见光粉末\\spectral_data_winsize9.csv", 
            "D:\\可见光粉末\\spectral_data_winsize9.hdf5", 100000, train_list, 96, balance=True) 

    # test_set = dataset_iron("E:\\成像光谱\\spectral_data_winsize5.csv", "E:\\成像光谱\\spectral_data_winsize5.hdf5", train_list, 2000)
    # train_set, test_set = random_split(dataset, [dataset.__len__()-2, 2], generator=torch.Generator().manual_seed(42)) 

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=7, drop_last=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    
    model = Grade_regressor().to(device)
    # model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(1预训练)精选 MSE 混合 lr8e-7 b250 dropout0.2_2000.pt"))
    model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/粉末预训练_7500.pt")) 
    # model.weight_init() 
    train_regression_mix(train_loader, model, 1000000, lr=1e-5*0.94**75, tag="粉末预训练", lr_decay=0.94, lr_decay_step=1000, lr_lower_bound=1e-7, step=7501, vis=model.visualization)

