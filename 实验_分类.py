from unicodedata import name
from models.attention_series import feature_conbined
from models.fc import fc_single_variable
from dataloaders.dataloaders import dataset_xiongan
from torch.utils.data import DataLoader
from torch.utils.data  import random_split
from trainer import train_classifier
from configs.training_cfg import device
import torch

if __name__ == "__main__":
    dataset = dataset_xiongan("E:/xiongan")
    train_set, test_set = random_split(dataset, [3677110-4000, 4000], generator=torch.Generator().manual_seed(42))
    del dataset

    train_loader = DataLoader(train_set, shuffle=True, batch_size=8, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=8, num_workers=0, drop_last=True)
    
    model = feature_conbined().to(device)
    train(train_loader, test_loader, model, 5, lr=0.0001, tag="嗷2", vis=model.visualization)


