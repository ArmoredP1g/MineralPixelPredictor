import spectral
import pandas as pd
import torch
import os
import numpy as np
import h5py
import ast
from PIL import Image

included_extensions = ['hdr']
data_dir = 'D:/new'
window_size = 9

hdr_files = [fn for fn in os.listdir(data_dir) if any(fn.endswith(ext) for ext in included_extensions)]
mask_rgb_values = [[255,242,0],[34,177,76],[255,0,88]]

from random import sample

spectral.settings.envi_support_nonlowercase_params = True

csv_data = []
data_id = 0
f = h5py.File("spectral_data_winsize{}.hdf5".format(window_size), "w")
g1 = f.create_group("data")

for file_name in hdr_files:
    number = file_name.split('-')[0]
    img = spectral.envi.open(data_dir+"/"+file_name)
    gt_TFe = ast.literal_eval(img.metadata['gt_TFe'])
    gt_FeO = ast.literal_eval(img.metadata['gt_FeO'])
    gt_SiO2 = ast.literal_eval(img.metadata['gt_SiO2'])
    gt_F3O4 = ast.literal_eval(img.metadata['gt_F3O4'])
    gt_F2O3 = ast.literal_eval(img.metadata['gt_F2O3'])
    gt_Fe2_plus = ast.literal_eval(img.metadata['gt_Fe2_plus'])
    gt_Fe3_plus = ast.literal_eval(img.metadata['gt_Fe3_plus'])
    gt_Magnetic_rate = ast.literal_eval(img.metadata['gt_Magnetic_rate'])
    img = torch.Tensor(img.asarray()/6000)
    mask = np.array(Image.open(data_dir+"/"+file_name+"_mask.png"))
    r,c,_ = mask.shape

    # 压缩数据量
    for row in range(r-window_size+1):
        for col in range(c-window_size+1):
            # A valid window is considered 
            # when the RGB values of the four corners 
            # of the window match a certain category in MASK

            if col%4 != 0 or row% 4!= 0:
                continue

            for i in range(3):
                # 都是他妈PNG，但画图处理后的多了一一个维度，3d画图没有，需要额外判断
                # 跑了俩小时白玩儿
                if mask.shape[2] == 4:
                    mask = mask[:,:,:-1]

                if mask[row,col].tolist() == mask_rgb_values[i] and \
                        mask[row+window_size-1,col].tolist() == mask_rgb_values[i] and \
                        mask[row,col+window_size-1].tolist() == mask_rgb_values[i] and \
                        mask[row+window_size-1,col+window_size-1].tolist() == mask_rgb_values[i]:
                    sample_id = "{}_{}".format(number,chr(ord('A') + i))
                    csv_data.append({
                        'sample_id': sample_id,
                        'data_id': str(data_id),
                        'gt_TFe': gt_TFe[i],
                        'gt_FeO': gt_FeO[i],
                        'gt_SiO2': gt_SiO2[i],
                        'gt_F3O4': gt_F3O4[i],
                        'gt_F2O3': gt_F2O3[i],
                        'gt_Fe2_plus': gt_Fe2_plus[i],
                        'gt_Fe3_plus': gt_Fe3_plus[i],
                        'gt_Magnetic_rate': gt_Magnetic_rate[i]
                    })
                    g1.create_dataset(str(data_id), data=img[row:row+window_size,col:col+window_size,:].mean(dim=(0,1)).tolist(), dtype=np.float16)
                    data_id += 1
    print("文件‘{}’处理完成".format(file_name))
pd.DataFrame(csv_data, columns=['sample_id','data_id','gt_TFe','gt_FeO','gt_SiO2','gt_F3O4','gt_F2O3','gt_Fe2_plus','gt_Fe3_plus','gt_Magnetic_rate']).to_pickle("spectral_data_winsize{}.pkl".format(window_size))
f.close()
    