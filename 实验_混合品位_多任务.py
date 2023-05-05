from unicodedata import name
from models.attention_series import Grade_regressor
import h5py
import ast
import numpy as np
import spectral
from spectral import imshow
from PIL import Image
spectral.settings.envi_support_nonlowercase_params = True
from dataloaders.dataloaders import dataset_multi_labels
import torch
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_regression_multitask
from configs.training_cfg import device
import torch

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    dataset_path = "E:\\d盘备份\\近红外部分"

    train_list = ['13_A','11_A','12_A','12_C','25_A','42_B','55_C','15_A',
                '56_B','4_B','42_A','57_A','14_B','36_B','43_C','26_A',
                '9_C','43_A','53_A','3_B','30_C','27_A','22_B','27_C','31_C',
                '53_B','32_A','6_B','52_B','8_B','41_B','31_A','34_A',
                '7_B','53_C','54_C','29_B','16_B','47_A','49_B','10_C',
                '21_C','31_B','50_A','18_A','22_C','52_C','38_A','17_A',
                '59_A','4_A','57_B','33_C','7_A','49_C','58_B','4_C',
                '52_A','17_C','23_A','7_C','46_B','30_B','46_A','18_C',
                '24_A','55_A','40_A','55_B','6_A','59_B','3_C','27_B',
                '18_B','5_A','29_A','25_B','49_A','32_C','45_C','12_B',
                '20_A','9_A','28_C','29_C','5_C','46_C','14_C','19_A',
                '23_B','9_B','40_B','35_C','13_C','50_B','35_B','15_B',
                '44_C','45_A','23_C','1_C','1_B','35_A','32_B','6_C',
                '51_B','28_B','2_B','58_C','38_C','2_A','26_B','2_C',
                '16_C','43_B','24_C','54_B','15_C','42_C','36_A','37_A',
                '41_C','44_B','19_C','51_C','1_A','39_B','28_A','58_A',
                '39_C','30_A','39_A','54_A','61_C','61_A','37_B','48_C',
                '21_A','22_A','48_B','48_A']
    
    test_list = ['14_A',
                '11_C','16_A','13_B','21_B','36_C','31_A','34_B','56_A',
                '47_C','8_A','19_B','3_A','62_A','33_B','10_A','24_B','60_B',
                '11_B','59_C','10_B','60_C','45_B','8_C','41_A','38_B',
                '57_C','50_C','61_B','37_C']

    #for test: 14 19 54
    mask_rgb_values = [[255,242,0],[34,177,76],[255,0,88]]
    pool = torch.nn.AvgPool2d(3,3)
    test_data = []
    for id in test_list:
        pixel_list = []
        imgid, sampleid = id.split('_')
        sampleid = ord(sampleid) - 65
        img_data = spectral.envi.open(dataset_path+"\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(imgid))
        gt_TFe = ast.literal_eval(img_data.metadata['gt_TFe'])
        gt_Fe3 = ast.literal_eval(img_data.metadata['gt_Fe3_plus'])
        gt_SiO2 = ast.literal_eval(img_data.metadata['gt_SiO2'])
        img_data = torch.Tensor(img_data.asarray()/6000)[:,:,:]
        img_data = pool(img_data.permute(2,0,1)).permute(1,2,0)
        mask = np.array(Image.open(dataset_path+"\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png".format(imgid)))
        row, col, _ = img_data.shape
        for r in range(row):
            for c in range(col):
                if mask[r*3+1,c*3+1].tolist() == mask_rgb_values[sampleid]:
                    pixel_list.append(img_data[r,c].unsqueeze(0))

        pixel_list = torch.cat(pixel_list, dim=0)
        test_data.append({
            "tensor": pixel_list.to(device),
            "gt": torch.Tensor([gt_TFe[sampleid], gt_Fe3[sampleid], gt_SiO2[sampleid]]).to(device)
        })


    train_set = dataset_multi_labels("E:\\d盘备份\\近红外部分\\spectral_data_IR_winsize3.csv",
            "E:\\d盘备份\\近红外部分\\spectral_data_IR_winsize3.hdf5", 100000, train_list, 96)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=7, drop_last=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    model = Grade_regressor(encoder='TSI',tasks=3).to(device)
    # model.weight_init() 
    train_regression_multitask(train_loader, model, 1, lr=1e-6, tag="(NIR-TSI)Mutitask0.93", pretrain_step=-1, lr_decay=0.93, lr_decay_step=1000, lr_lower_bound=1e-8, step=0, test_data=test_data, vis=model.visualization)

    # train_set = dataset_multi_labels("E:\\d盘备份\\近红外部分\\spectral_data_IR_winsize3.csv",
    #         "E:\\d盘备份\\近红外部分\\spectral_data_IR_winsize3.hdf5", 100000, train_list, 96)

    # train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=7, drop_last=True)
    # # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    # model = Grade_regressor(encoder='ASI',tasks=3).to(device)
    # model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(NIR-ASI)Mutitask0.98_100000.pt")) 
    # # model.weight_init() 
    # train_regression_multitask(train_loader, model, 1, lr=1e-6*0.98**100, tag="(NIR-ASI)Mutitask0.98", pretrain_step=-1, lr_decay=0.98, lr_decay_step=1000, lr_lower_bound=1e-8, step=100001, test_data=test_data, vis=model.visualization)



