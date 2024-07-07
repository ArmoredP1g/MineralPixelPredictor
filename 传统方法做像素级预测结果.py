import sys
sys.path.append('../')
from unicodedata import name
import ast
import numpy as np
import spectral
from PIL import Image
spectral.settings.envi_support_nonlowercase_params = True
import torch
from configs.training_cfg import *
import torch
import random
import os
import openpyxl
from time import sleep
from multiprocessing import Manager, Process
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor

torch.autograd.set_detect_anomaly(True)
pool = torch.nn.AvgPool2d(3,3)
mask_rgb_values = [[255,242,0],[34,177,76],[255,0,88]]
torch.autograd.set_detect_anomaly(True)

def snv(data):
    return (data - torch.mean(data, dim=0)) / torch.std(data, dim=0)

def test_data_avg(list, pool_size=3):
    pool = torch.nn.AvgPool2d(pool_size, pool_size)
    test_data_list = []
    for id in list:
        pixel_list = []
        imgid, sampleid = id.split('_')
        sampleid = ord(sampleid) - 65
        img_data = spectral.envi.open(dataset_path+"/spectral_data/{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(imgid))
        gt = ast.literal_eval(img_data.metadata['gt_TFe'])
        img_data = torch.Tensor(img_data.asarray()/6000)[:,:,:]
        img_data = pool(img_data.permute(2,0,1)).permute(1,2,0)
        mask = np.array(Image.open(dataset_path+"/spectral_data/{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png".format(imgid)))
        mask = pool(torch.tensor(mask, dtype=float).permute(2,0,1)).permute(1,2,0)
        print(img_data.shape)
        row, col, _ = img_data.shape
        for r in range(row):
            for c in range(col):
                if mask[r,c].tolist() == mask_rgb_values[sampleid]:
                    pixel_list.append(img_data[r,c].unsqueeze(0))

        pixel_list = torch.cat(pixel_list, dim=0)
        test_data_list.append({
            "tensor": torch.mean(pixel_list, dim=0).to(device),
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })
    return test_data_list

def test_data_pix(list, pool_size=3):
    pool = torch.nn.AvgPool2d(pool_size, pool_size)
    test_data_list = []
    for id in list:
        pixel_list = []
        imgid, sampleid = id.split('_')
        sampleid = ord(sampleid) - 65
        img_data = spectral.envi.open(dataset_path+"/spectral_data/{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(imgid))
        gt = ast.literal_eval(img_data.metadata['gt_TFe'])
        img_data = torch.Tensor(img_data.asarray()/6000)[:,:,:]
        img_data = pool(img_data.permute(2,0,1)).permute(1,2,0)
        mask = np.array(Image.open(dataset_path+"/spectral_data/{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png".format(imgid)))
        mask = pool(torch.tensor(mask, dtype=float).permute(2,0,1)).permute(1,2,0)
        print(img_data.shape)
        row, col, _ = img_data.shape
        for r in range(row):
            for c in range(col):
                if mask[r,c].tolist() == mask_rgb_values[sampleid]:
                    pixel_list.append(img_data[r,c].unsqueeze(0))

        pixel_list = torch.cat(pixel_list, dim=0)
        test_data_list.append({
            "tensor": pixel_list.reshape(-1, 168).to(device),
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })
    return test_data_list


baseline_cfg = {
    'SG': False,
    'SG_win_len': 4,
    'SG_poly': 1,
    'dim_reduction_method': 'KPCA',
    'KPCA_kernel': 'rbf',
    'ISOMAP_n_neighbors': 5,
    'ISOMAP_p': 1,
    'SVR_C': 9.307534342501125,
    'SVR_epsilon': 4.497825696984441,
    'SVR_kernel': 'poly',
    'SVR_degree': 3,
    'XGB_eta': 0.12466381919021673,
    'XGB_n_estimators': 63,
    'XGB_max_depth': 8,
    'XGB_subsample': 0.501430483369999,
    'PLSR_max_iter': 320,
    'n_components': 17
}


if __name__ == "__main__":
    model = 'SVR'
    # processor = 6


    all_samples = ['13_A','11_A','12_A','12_C','25_A','42_B','55_C','15_A',
                '56_B','4_B','42_A','57_A','14_B','36_B','43_C','26_A',
                '9_C','43_A','53_A','3_B','30_C','27_A','22_B','27_C','31_C',
                '53_B','32_A','6_B','52_B','8_B','41_B','31_A','34_A',
                '7_B','53_C','54_C','29_B','16_B','47_A','49_B','10_C',
                '21_C','50_A','18_A','22_C','52_C','38_A','17_A',
                '59_A','4_A','57_B','33_C','7_A','49_C','58_B','4_C',
                '52_A','17_C','23_A','7_C','46_B','30_B','46_A','18_C',
                '24_A','55_A','40_A','55_B','6_A','59_B','3_C','27_B',
                '18_B','5_A','29_A','25_B','49_A','32_C','45_C','12_B',
                '20_A','9_A','28_C','29_C','5_C','46_C','14_C','19_A',
                '23_B','9_B','40_B','35_C','13_C','50_B','35_B','15_B',
                '45_A','23_C','1_C','1_B','35_A','32_B','6_C',
                '51_B','28_B','2_B','58_C','38_C','2_A','26_B','2_C',
                '16_C','43_B','24_C','54_B','15_C','42_C','36_A','37_A',
                '41_C','44_B','19_C','51_C','1_A','39_B','28_A',
                '39_C','30_A','39_A','54_A','61_C','61_A','37_B','48_C',
                '21_A','22_A','48_B','48_A'] + \
                ['14_A',
                '11_C','16_A','13_B','21_B','36_C','34_B','56_A',
                '47_C','8_A','19_B','3_A','62_A','33_B','10_A','24_B','60_B',
                '11_B','59_C','10_B','8_C','41_A','38_B',
                '57_C','61_B','37_C']

    data = []
    for id in all_samples:
        imgid, sampleid = id.split('_')
        sampleid = ord(sampleid) - 65
        metadata = spectral.envi.open(dataset_path+"/spectral_data/{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(imgid)).metadata
        gt = ast.literal_eval(metadata['gt_TFe'])
        data.append({
            "id": id,
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })

    # 排序数据集
    data = sorted(data, key=lambda x: x["gt"])

    # split data into 5 folds, make each fold has the same distribution
    folds = [[],[],[],[],[]]
    for i in range(len(data)):
        folds[i%5].append(data[i]["id"])

    training_list = []
    test_list = []
    i = 0
    for j in range(5):
        if j != i:
            training_list += folds[j]
        else:
            test_list += folds[j]

    training_tdl = test_data_avg(training_list)
    test_tdl = test_data_pix(test_list)

    training_data = torch.cat([snv(x["tensor"]).unsqueeze(0) for x in training_tdl], dim=0).cpu().numpy()
    training_gt = torch.cat([x["gt"] for x in training_tdl], dim=0).cpu().numpy()

    test_data = [torch.cat([snv(t['tensor'][s]).unsqueeze(0) for s in range(0, t['tensor'].shape[0])], dim=0) for t in test_tdl]
    test_gt = torch.cat([x["gt"] for x in test_tdl], dim=0).cpu().numpy()

    # evaluate the performance of the model

    if baseline_cfg["SG"] == True:
        training_data = savgol_filter(training_data, baseline_cfg["SG_win_len"], baseline_cfg["SG_poly"], axis=1)
        for i in range(len(test_data)):
            test_data[i] = savgol_filter(test_data[i].cpu().numpy(), baseline_cfg["SG_win_len"], baseline_cfg["SG_poly"], axis=1)

    # PLSR不需要降维
    if model != 'PLSR':
        if baseline_cfg["dim_reduction_method"] == 'PCA':
            pca = PCA(n_components=15)
            training_data = pca.fit_transform(training_data)
            for i in range(len(test_data)):
                test_data[i] = pca.transform(test_data[i].cpu().numpy())
        elif baseline_cfg["dim_reduction_method"] == 'KPCA':
            kpca = KernelPCA(n_components=15, kernel=baseline_cfg["KPCA_kernel"])
            training_data = kpca.fit_transform(training_data)
            for i in range(len(test_data)):
                test_data[i] = kpca.transform(test_data[i].cpu().numpy())
        elif baseline_cfg["dim_reduction_method"] == 'ISOMAP':
            iso = Isomap(n_neighbors=baseline_cfg["ISOMAP_n_neighbors"], n_components=15, p=baseline_cfg["ISOMAP_p"])
            training_data = iso.fit_transform(training_data)
            for i in range(len(test_data)):
                test_data[i] = iso.transform(test_data[i].cpu().numpy())


    if model == 'SVR':
        svr = SVR(C=baseline_cfg["SVR_C"], epsilon=baseline_cfg["SVR_epsilon"], kernel=baseline_cfg["SVR_kernel"], degree=baseline_cfg["SVR_degree"])
        svr.fit(training_data, training_gt)
        pred = np.array([np.mean(svr.predict(data)) for data in test_data])
        
    elif model == 'XGB':
        xgb = XGBRegressor(eta=baseline_cfg["XGB_eta"], n_estimators=baseline_cfg["XGB_n_estimators"], max_depth=baseline_cfg["XGB_max_depth"], subsample=baseline_cfg["XGB_subsample"])
        xgb.fit(training_data, training_gt)
        pred = np.array([np.mean(xgb.predict(data)) for data in test_data])

    elif model == 'PLSR':
        plsr = PLSRegression(n_components=15, max_iter=baseline_cfg["PLSR_max_iter"])
        plsr.fit(training_data, training_gt)
        pred = np.array([np.mean(plsr.predict(data)) for data in test_data])

    rmse = np.sqrt(np.mean((pred - test_gt)**2))
    nrmse = np.sqrt(np.mean((pred - test_gt)**2)) / (np.max(test_gt) - np.min(test_gt))
    r2 = 1 - np.sum((pred - test_gt)**2) / np.sum((test_gt - np.mean(test_gt))**2)

    # 打印结果
    print("RMSE: ", rmse)
    print("NRMSE: ", nrmse)

    print("R2: ", r2)

