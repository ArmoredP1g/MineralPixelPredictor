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
    dataset_path = "E:\\d盘备份\\可见光部分"

    # # 全部样本
    train_list = ['14_B','22_B','25_A','13_B','46_C','61_B','5_B','18_C','45_C','60_A',
                '20_C','44_C','26_A','19_C','43_C','55_A','17_C','33_A','3_C','4_C','60_C','34_C','27_C',
                '28_A','12_B','48_A','55_B','31_B','34_A','51_B','58_A','32_A','52_A',
                '6_C','8_C','60_B','13_A','23_C','32_B','14_A','24_C','44_A','49_B',
                '29_C','18_B','57_A','8_A','37_C','7_A','10_A','3_B','50_B','19_A','25_B','47_B','17_B','39_C','46_A','20_A','56_B',
                '57_B','46_B','51_C','2_A','9_A','30_C','34_B','16_A','1_A','52_B',
                '25_C','8_B','56_A','35_B','14_C','16_B','24_B','7_B','33_B','11_A',
                '45_B','13_C','15_B','20_B','9_C','47_A','50_C','35_C','19_B','10_C',
                '54_A','37_B','61_C','40_A','45_A','23_A','2_C','3_A','1_B','32_C',
                '36_A','49_A','10_B','2_B','50_A','4_B','29_A','18_A','26_C','4_A','6_A','41_B','24_A','64_A',
                '58_B','5_A','43_B','35_A','59_A','40_C','41_A','54_B',
                '17_A','38_C','37_A','59_B','30_B','61_A','42_A','63_A',
                '36_C','30_A','58_C','39_A','62_C','42_C','62_B','42_B']

    test_list = ['15_C','16_C','21_B','6_B','7_C','29_B','56_C','11_B','47_C','51_A','53_B',
                '52_C','9_B','26_B','53_C','39_B','44_B','33_C','23_B','53_A','15_A','5_C',
                '43_A','55_C','41_C','31_A','36_B','57_C','63_B','63_C','62_A','38_B']

    #for test: 14 19 54
    mask_rgb_values = [[255,242,0],[34,177,76],[255,0,88]]
    pool = torch.nn.AvgPool2d(9,9)
    test_data = []
    for id in test_list:
        print("装载测试数据{}".format(id))
        pixel_list = []
        imgid, sampleid = id.split('_')
        sampleid = ord(sampleid) - 65
        img_data = spectral.envi.open(dataset_path+"\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(imgid))
        gt = ast.literal_eval(img_data.metadata['gt_TFe'])
        img_data = torch.Tensor(img_data.asarray()/6000)[:,:,:]
        img_data = pool(img_data.permute(2,0,1)).permute(1,2,0)
        mask = np.array(Image.open(dataset_path+"\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png".format(imgid)))
        if mask.shape[2] == 4:
            mask = mask[:,:,:-1]
        row, col, _ = img_data.shape
        for r in range(row):
            for c in range(col):
                if mask[r*9+4,c*9+4].tolist() == mask_rgb_values[sampleid]:
                    pixel_list.append(img_data[r,c].unsqueeze(0))

        pixel_list = torch.cat(pixel_list, dim=0)
        test_data.append({
            "tensor": pixel_list.to(device),
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })



    train_set = dataset_iron_balanced_mixed("E:\\d盘备份\\可见光部分\\spectral_data_winsize9.csv",
            "E:\\d盘备份\\可见光部分\\spectral_data_winsize9.hdf5", 100000, train_list, 96, balance=True)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=7, drop_last=True)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    
    model = Grade_regressor().to(device)
    model.decoder1.pretrain_on()
    # model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(1预训练)精选 MSE 混合 lr8e-7 b250 dropout0.2_2000.pt"))
#     model.load_state_dict(torch.load("D:/source/repos/Pixel-wise-hyperspectral-feature-classification-experiment/ckpt/(NIR)RELU+ALLINDEX_LINEAR_1500.pt")) 
    # model.weight_init() 
    train_regression_mix(train_loader, model, 1000000, lr=1e-5, tag="VIS_30T_all_修改平衡采样概率", pretrain_step=2500, lr_decay=0.93, lr_decay_step=1000, lr_lower_bound=1e-7, step=0, test_data=test_data, vis=model.visualization)


