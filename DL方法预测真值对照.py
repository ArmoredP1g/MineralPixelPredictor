import sys
sys.path.append('../')
from unicodedata import name
from models.models import ConvPredictor
import json
import h5py
import ast
import numpy as np
import spectral
from spectral import imshow
from PIL import Image
spectral.settings.envi_support_nonlowercase_params = True
from dataloaders.dataloaders import dataset_iron_balanced_mixed
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_regression
from copy import deepcopy
from configs.training_cfg import *
import torch
import random
import os
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



def plot_balanced_vs_unbalanced(actual_balanced, prediction_balanced, actual_unbalanced, prediction_unbalanced, file_name):
    # 创建一个新的图像
    fig = plt.figure()

    # 画出平衡方法的预测值与真实值的散点图，用蓝色表示
    plt.scatter(actual_balanced, prediction_balanced, color='blue', label='Balanced')

    # 画出不平衡方法的预测值与真实值的散点图，用红色表示
    plt.scatter(actual_unbalanced, prediction_unbalanced, color='red', label='Unbalanced')

    # 计算所有值的最大值和最小值，用于画y=x的直线
    max_value = max(max(actual_balanced), max(prediction_balanced), max(actual_unbalanced), max(prediction_unbalanced))
    min_value = min(min(actual_balanced), min(prediction_balanced), min(actual_unbalanced), min(prediction_unbalanced))
    plt.plot([min_value, max_value], [min_value, max_value], 'g--')  # 使用绿色虚线表示y=x

    # 设置图像的标题和坐标轴标签
    plt.title('Balanced vs Unbalanced Prediction vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # 添加图例
    plt.legend()

    # 保存图像到文件
    plt.savefig(file_name)

    # 返回图像，方便在tensorboard中或其他地方显示
    return fig

def snv(data):
    return (data - torch.mean(data, dim=0)) / torch.std(data, dim=0)

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
        row, col, _ = img_data.shape
        for r in range(row):
            for c in range(col):
                if mask[r*pool_size+1,c*pool_size+1].tolist() == mask_rgb_values[sampleid]:
                    pixel_list.append(img_data[r,c].unsqueeze(0))

        pixel_list = torch.cat(pixel_list, dim=0)
        test_data_list.append({
            "tensor": pixel_list,
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })
    return test_data_list


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
        row, col, _ = img_data.shape
        for r in range(row):
            for c in range(col):
                if mask[r*pool_size+1,c*pool_size+1].tolist() == mask_rgb_values[sampleid]:
                    pixel_list.append(img_data[r,c].unsqueeze(0))

        pixel_list = torch.cat(pixel_list, dim=0)
        test_data_list.append({
            "tensor": torch.mean(pixel_list, dim=0).to(device),
            "gt": torch.Tensor([gt[sampleid]]).to(device)
        })
    return test_data_list


if __name__ == "__main__":

    models_balanced = [ConvPredictor().to(device).eval() for i in range(5)]
    # models_unbalanced = [ConvPredictor().to(device).eval() for i in range(5)]

    # models_unbalanced[0].load_state_dict(torch.load("ckpt/(0)unbalanced/fold0_step100000.pt"))
    # models_unbalanced[1].load_state_dict(torch.load("ckpt/(1)unbalanced/fold0_step100000.pt"))
    # models_unbalanced[2].load_state_dict(torch.load("ckpt/(2)unbalanced/fold0_step100000.pt"))
    # models_unbalanced[3].load_state_dict(torch.load("ckpt/(3)unbalanced/fold0_step100000.pt"))
    # models_unbalanced[4].load_state_dict(torch.load("ckpt/(4)unbalanced/fold0_step100000.pt"))

    models_balanced[0].load_state_dict(torch.load("ckpt/(0)2x_1diff1/fold0_step100000.pt"))
    models_balanced[1].load_state_dict(torch.load("ckpt/(1)2x_1diff1/fold0_step100000.pt"))
    models_balanced[2].load_state_dict(torch.load("ckpt/(2)2x_1diff1/fold0_step100000.pt"))
    models_balanced[3].load_state_dict(torch.load("ckpt/(3)2x_1diff1/fold0_step100000.pt"))
    models_balanced[4].load_state_dict(torch.load("ckpt/(4)2x_1diff1/fold0_step100000.pt"))


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

    test_tdl = test_data_pix(test_list)
    test_tdl_avg = test_data_avg(test_list)

    gt_all = []
    for d in test_tdl:
        gt_all.append(d['gt'].cpu())

    avg_prediction_unbalanced = [0 for i in range(len(test_tdl))]
    avg_prediction_balanced = [0 for i in range(len(test_tdl))]

    for model in models_balanced:
        model.eval()
        for idx, d in enumerate(test_tdl):
            data = d['tensor'].to(device)
            gt = d['gt']
            with torch.no_grad():
                len = data.shape[0]
                cur = 0
                pixelwise_prediction = []
                while cur + 100 <= len: # 每次评估100个像素
                    pixelwise_prediction.append(model(data[cur:cur+100]).to("cpu"))
                    cur += 100
                    torch.cuda.empty_cache()

                if cur != len:
                    pixelwise_prediction.append(model(data[cur:len]).to("cpu"))

                pixelwise_prediction = torch.cat(pixelwise_prediction, dim=0)
            prediction = pixelwise_prediction.mean()*(100/5)
            avg_prediction_balanced[idx] += prediction


    DL_prediction = [i for i in avg_prediction_balanced]

    # svr pixelwise

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


    training_tdl = test_data_avg(training_list)

    training_data = torch.cat([torch.cat((snv(x["tensor"]), (x["tensor"][1:]-x["tensor"][:-1]))).unsqueeze(0) for x in training_tdl], dim=0).cpu().numpy()
    training_gt = torch.cat([x["gt"] for x in training_tdl], dim=0).cpu().numpy()

    test_data = [torch.cat([torch.cat((snv(t['tensor'][s]), (t['tensor'][s][1:]-t['tensor'][s][:-1]))).unsqueeze(0) for s in range(0, t['tensor'].shape[0])], dim=0) for t in test_tdl] #n, 168
    
    test_data_a = torch.cat([torch.cat((snv(x["tensor"]), (x["tensor"][1:]-x["tensor"][:-1]))).unsqueeze(0) for x in test_tdl_avg], dim=0).cpu().numpy()
    
    test_gt = torch.cat([x["gt"] for x in test_tdl], dim=0).cpu().numpy()

    # PLSR不需要降维
    if model != 'PLSR':
        if baseline_cfg["dim_reduction_method"] == 'PCA':
            pca = PCA(n_components=15)
            training_data = pca.fit_transform(training_data)
            for i in range(test_data.__len__()):
                test_data[i] = pca.transform(test_data[i].cpu().numpy())
            test_data_a = pca.transform(test_data_a)
            
        elif baseline_cfg["dim_reduction_method"] == 'KPCA':
            kpca = KernelPCA(n_components=15, kernel=baseline_cfg["KPCA_kernel"])
            training_data = kpca.fit_transform(training_data)
            for i in range(test_data.__len__()):
                test_data[i] = kpca.transform(test_data[i].cpu().numpy())
            test_data_a = kpca.transform(test_data_a)
        elif baseline_cfg["dim_reduction_method"] == 'ISOMAP':
            iso = Isomap(n_neighbors=baseline_cfg["ISOMAP_n_neighbors"], n_components=15, p=baseline_cfg["ISOMAP_p"])
            training_data = iso.fit_transform(training_data)
            for i in range(test_data.__len__()):
                test_data[i] = iso.transform(test_data[i].cpu().numpy())
            test_data_a = iso.transform(test_data_a)
    svr = SVR(C=baseline_cfg["SVR_C"], epsilon=baseline_cfg["SVR_epsilon"], kernel=baseline_cfg["SVR_kernel"], degree=baseline_cfg["SVR_degree"])
    svr.fit(training_data, training_gt)    ##我草泥马啊
    pred = np.array([np.mean(svr.predict(data)) for data in test_data])
    pred_svr_pixelwise = deepcopy(pred)
    pred = svr.predict(test_data_a)
    pred_svr_avg = deepcopy(pred)



    # xgb pixelwise


    #  new baseline cfg:{'SG': True, 'SG_win_len': 9, 'SG_poly': 1, 'dim_reduction_method': 'KPCA', 'KPCA_kernel': 'poly', 'ISOMAP_n_neighbors': 10, 'ISOMAP_p': 2, 'SVR_C': 5.638334322963087, 'SVR_epsilon': 9.280636384767028, 'SVR_kernel': 'sigmoid', 'SVR_degree': 4, 'XGB_eta': 0.10140214609758606, 'XGB_n_estimators': 72, 'XGB_max_depth': 8, 'XGB_subsample': 0.7191052612654845, 'PLSR_max_iter': 384, 'n_components': 19}
    baseline_cfg = {

        'SG': True,
        'SG_win_len': 9,
        'SG_poly': 1,
        'dim_reduction_method': 'KPCA',
        'KPCA_kernel': 'poly',
        'ISOMAP_n_neighbors': 10,
        'ISOMAP_p': 2,
        'SVR_C': 5.638334322963087,
        'SVR_epsilon': 9.280636384767028,
        'SVR_kernel': 'sigmoid',
        'SVR_degree': 4,
        'XGB_eta': 0.10140214609758606,
        'XGB_n_estimators': 72,
        'XGB_max_depth': 8,
        'XGB_subsample': 0.7191052612654845,
        'PLSR_max_iter': 384,
        'n_components': 19
    }


    training_tdl = test_data_avg(training_list)

    training_data = torch.cat([torch.cat((snv(x["tensor"]), (x["tensor"][1:]-x["tensor"][:-1]))).unsqueeze(0) for x in training_tdl], dim=0).cpu().numpy()
    training_gt = torch.cat([x["gt"] for x in training_tdl], dim=0).cpu().numpy()

    test_data = [torch.cat([torch.cat((snv(t['tensor'][s]), (t['tensor'][s][1:]-t['tensor'][s][:-1]))).unsqueeze(0) for s in range(0, t['tensor'].shape[0])], dim=0) for t in test_tdl] #n, 168
    
    test_data_a = torch.cat([torch.cat((snv(x["tensor"]), (x["tensor"][1:]-x["tensor"][:-1]))).unsqueeze(0) for x in test_tdl_avg], dim=0).cpu().numpy()
    
    test_gt = torch.cat([x["gt"] for x in test_tdl], dim=0).cpu().numpy()

    # PLSR不需要降维
    if model != 'PLSR':
        if baseline_cfg["dim_reduction_method"] == 'PCA':
            pca = PCA(n_components=15)
            training_data = pca.fit_transform(training_data)
            for i in range(test_data.__len__()):
                test_data[i] = pca.transform(test_data[i].cpu().numpy())
            test_data_a = pca.transform(test_data_a)
        elif baseline_cfg["dim_reduction_method"] == 'KPCA':
            kpca = KernelPCA(n_components=15, kernel=baseline_cfg["KPCA_kernel"])
            training_data = kpca.fit_transform(training_data)
            for i in range(test_data.__len__()):
                test_data[i] = kpca.transform(test_data[i].cpu().numpy())
            test_data_a = kpca.transform(test_data_a)
        elif baseline_cfg["dim_reduction_method"] == 'ISOMAP':
            iso = Isomap(n_neighbors=baseline_cfg["ISOMAP_n_neighbors"], n_components=15, p=baseline_cfg["ISOMAP_p"])
            training_data = iso.fit_transform(training_data)
            for i in range(test_data.__len__()):
                test_data[i] = iso.transform(test_data[i].cpu().numpy())
            test_data_a = iso.transform(test_data_a)

        
    xgb = XGBRegressor(eta=baseline_cfg["XGB_eta"], n_estimators=baseline_cfg["XGB_n_estimators"], max_depth=baseline_cfg["XGB_max_depth"], subsample=baseline_cfg["XGB_subsample"])
    xgb.fit(training_data, training_gt)
    pred = np.array([np.mean(xgb.predict(data)) for data in test_data])
    pred_xgb_pixelwise = deepcopy(pred)
    pred = xgb.predict(test_data_a)
    pred_xgb_avg = deepcopy(pred)


    # 请绘制一个图像
    # 图像要矢量图格式的，比如pdf，svg, eps等
    # 涉及的数据包括：gt_all, DL_prediction, pred_svr_pixelwise, pred_svr_avg, pred_xgb_pixelwise, pred_xgb_avg
    # 图像包括一行三个回归散点图，分别是SVR，XGB，DL，其中svr和xgb分别有两个散点图，分别是pixelwise和avg（用不同颜色的散点画在一张图上，用图例表示哪个是像素级哪个是平均），
    # DL只有一个散点图
    # 横坐标是真实值，纵坐标是预测值
    # 图像标题是"Prediction vs Actual"
    # 横坐标标题是"Groud Truth"
    # 纵坐标标题是"Predicted Value"
    # 图例是"Pixelwise", "Avg"
    # 请保存图像到文件"prediction_vs_actual.pdf"
    # 请返回图像对象

fig = plt.figure()
fig = plt.figure(figsize=(20, 6))  # Adjust the figure size to make subplots square
plt.subplot(1,3,1)
plt.scatter(gt_all, pred_svr_pixelwise, color='blue', label='Pixelwise')
plt.scatter(gt_all, pred_svr_avg, color='red', label='Avg')
plt.plot([0, 100], [0, 100], 'g--')
plt.title('SVR Prediction vs Actual')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted Value')
plt.legend()

plt.subplot(1,3,2)
plt.scatter(gt_all, pred_xgb_pixelwise, color='blue', label='Pixelwise')
plt.scatter(gt_all, pred_xgb_avg, color='red', label='Avg')
plt.plot([0, 100], [0, 100], 'g--')
plt.title('XGB Prediction vs Actual')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted Value')
plt.legend()

plt.subplot(1,3,3)
plt.scatter(gt_all, DL_prediction, color='blue', label='DL')
plt.plot([0, 100], [0, 100], 'g--')
plt.title('DL Prediction vs Actual')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted Value')
plt.legend()

plt.savefig("prediction_vs_actual.pdf")


