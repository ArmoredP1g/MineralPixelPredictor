from unicodedata import name
from models.attention_series import Grade_regressor
import h5py
import ast
import numpy as np
import spectral
from spectral import imshow
from PIL import Image
spectral.settings.envi_support_nonlowercase_params = True
from dataloaders.dataloaders import dataset_iron_balanced_mixed
import torch
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_regression_mix
from configs.training_cfg import device
import torch

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    dataset_path = "E:\\d盘备份\\近红外部分"

    # # 全部样本
    train_list = ['13_A','14_A','11_A','12_A','12_C','25_A',
                    '42_B','55_C','15_A','56_B','4_B','42_A',
                    '57_A','14_B','36_B','43_C','26_A','13_B',
                    '43_A','53_A','3_B','30_C','21_B','27_A',
                    '27_C','53_B','32_A','36_C','6_B','52_B',
                    '41_B','31_A','31_C','34_A','7_B','53_C',
                    '29_B','34_B','16_B','47_A','49_B','10_C',
                    '17_A','21_C','31_B','50_A','18_A','22_C',
                    '52_C','38_A','59_A','4_A','57_B','8_A',
                    '7_A','49_C','58_B','4_C','19_B','52_A',
                    '23_A','7_C','46_B','3_A','30_B','46_A',
                    '24_A','55_A','62_A','40_A','55_B','6_A',
                    '3_C','33_B','27_B','18_B','5_A','29_A',
                    '10_A','49_A','32_C','45_C','12_B','20_A',
                    '9_A','28_C','29_C','5_C','46_C','11_B',
                    '19_A','23_B','9_B','40_B','59_C','35_C',
                    '10_B','50_B','35_B','60_C','15_B','44_C',
                    '23_C','1_C','45_B','1_B','35_A','32_B',
                    '51_B','8_C','28_B','2_B','58_C','38_C',
                    '41_A','26_B','2_C','16_C','43_B','24_C',
                    '54_B','15_C','42_C','36_A','37_A','57_C',
                    '44_B','19_C','51_C','1_A','50_C','39_B',
                    '58_A','39_C','60_B','48_C','30_A','39_A',
                    '61_C','61_A','61_B','37_B','21_A','22_A',
                    '48_A','37_C']

    test_list = ['11_C','16_A','9_C','22_B','8_B','54_C','56_A','47_C','33_C','17_C','18_C',
                '59_B','25_B','24_B','14_C','13_C','45_A','6_C','2_A','38_B','41_C','28_A','54_A','48_B']

    #for test: 14 19 54
    mask_rgb_values = [[255,242,0],[34,177,76],[255,0,88]]
    pool = torch.nn.AvgPool2d(3,3)
    test_data = []
    for id in test_list:
        pixel_list = []
        imgid, sampleid = id.split('_')
        sampleid = ord(sampleid) - 65
        img_data = spectral.envi.open(dataset_path+"\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(imgid))
        gt = ast.literal_eval(img_data.metadata['gt_TFe'])
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
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })



    train_set = dataset_iron_balanced_mixed("E:\\d盘备份\\近红外部分\\spectral_data_IR_winsize3.csv",
            "E:\\d盘备份\\近红外部分\\spectral_data_IR_winsize3.hdf5", 100000, train_list, 96, balance=True)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=7, drop_last=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    
    model = Grade_regressor().to(device)
    model.decoder1.pretrain_on()
    # model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(1预训练)精选 MSE 混合 lr8e-7 b250 dropout0.2_2000.pt"))
#     model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(NIR)RELU+ALLINDEX_LINEAR_1500.pt")) 
    # model.weight_init() 
    train_regression_mix(train_loader, model, 1000000, lr=1e-5, tag="NIR_Reselected", pretrain_step=2000, lr_decay=0.93, lr_decay_step=1000, lr_lower_bound=1e-7, step=0, test_data=test_data, vis=model.visualization)

