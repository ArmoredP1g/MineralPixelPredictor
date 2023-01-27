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
    train_list = ['10_A','10_B','10_C','12_A','12_B','12_C',
                  '13_A','13_B','13_C','15_A','15_B','15_C','16_A','16_B','16_C','17_A','17_B','17_C',
                  '18_A','18_B','18_C','19_A','19_B','19_C','20_A','20_B','20_C',
                  '21_B','22_B','23_A','23_B','23_C','24_A','24_B','24_C',
                  '25_A','25_B','25_C','26_A','26_B','26_C','27_A','27_B','27_C','28_A','28_B','29_A','29_B','29_C',
                  '30_A','30_B','30_C','31_A','31_B','31_C','32_A','32_B','32_C','33_A','33_B','33_C',
                  '34_A','34_B','34_C','35_A','35_B','35_C','36_A','36_B','36_C','37_A','37_B','37_C',
                  '38_A','38_B','38_C','39_A','39_B','39_C','40_A','40_B','40_C','41_A','41_B','41_C',
                  '42_A','42_B','42_C','43_A','43_B','43_C','44_A','44_B','44_C','45_A','45_B','45_C',
                  '46_A','46_B','46_C','47_A','47_B','47_C','48_A','48_B','48_C','49_A','49_B','49_C',
                  '50_A','50_B','50_C','51_A','51_B','51_C','52_A','52_B','52_C','53_A','53_B','53_C','54_A','54_B','54_C',
                  '55_A','55_B','55_C','56_A','56_B','56_C','57_A','57_B','57_C',
                  '58_A','58_B','58_C','59_A','59_B','59_C','60_A','60_B','60_C','61_A','61_B','61_C',
                  '62_A','62_B','62_C','63_A','63_B','63_C','64_A']

    #for test: 9,14,11



    train_set = dataset_iron_balanced_mixed("E:\\d盘备份\\可见光部分\\spectral_data_winsize9.csv", 
            "E:\\d盘备份\\可见光部分\\spectral_data_winsize9.hdf5", 100000, train_list, 96, balance=True) 

    # test_set = dataset_iron("E:\\成像光谱\\spectral_data_winsize5.csv", "E:\\成像光谱\\spectral_data_winsize5.hdf5", train_list, 2000)
    # train_set, test_set = random_split(dataset, [dataset.__len__()-2, 2], generator=torch.Generator().manual_seed(42)) 

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=8, drop_last=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    
    model = Grade_regressor().to(device)
    # model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(1预训练)精选 MSE 混合 lr8e-7 b250 dropout0.2_2000.pt"))
    # model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/RELU+ALLINDEX_TANH_4000.pt")) 
    # model.weight_init() 
    train_regression_mix(train_loader, model, 1000000, lr=1e-5, tag="(VIS)RELU+ALLINDEX_LINEAR", lr_decay=0.93, lr_decay_step=1000, lr_lower_bound=1e-7, step=1, vis=model.visualization)

