from unicodedata import name
from models.attention_series import feature_conbined_regression
import h5py
from models.fc import fc_single_variable
from dataloaders.dataloaders import dataset_iron
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_regression
from configs.training_cfg import device
import torch

if __name__ == "__main__":
    dataset = dataset_iron("notbooks/spectral_data_winsize10.csv",500)
    train_set, test_set = random_split(dataset, [dataset.__len__()-2, 2], generator=torch.Generator().manual_seed(42))
    del dataset

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=1, num_workers=0, drop_last=True)
    
    model = feature_conbined_regression(freeze=False).to(device)
    train_regression(train_loader, test_loader, model, 100, lr=0.0001, tag="test2", vis=model.visualization)


