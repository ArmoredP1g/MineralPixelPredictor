from models.models import AE_Encoder, AE_Decoder, Predictor
from trainer import train_AE, train_predictor
import torch.nn as nn
from configs.training_cfg import device, learning_rate, lr_decay, lr_decay_step, lr_lower_bound, session_tag
from deep_learning_experiment import build_deep_learning_folds, build_test_data, build_train_loaders, split_fold_lists, write_training_manifest

if __name__ == "__main__":
    folds = build_deep_learning_folds()
    write_training_manifest(folds)

    encoder = AE_Encoder().to(device)
    decoder = AE_Decoder().to(device)
    predictor = Predictor().to(device)
    for i in range(5):
        training_list, test_list = split_fold_lists(folds, i)
        model_ae = nn.Sequential(encoder, decoder).to(device)
        print("Training AE model for fold {}".format(i))
        trainloader_ae, trainloader_predictor = build_train_loaders(training_list)

        encoder = train_AE(trainloader_ae, model_ae, i, 
                lr=learning_rate, 
                tag=session_tag+"_fold{}".format(i), 
                lr_decay_step=lr_decay_step, 
                step=0)

        train_predictor(trainloader_predictor, encoder, predictor, i, 
                lr=learning_rate, 
                tag=session_tag+"_fold{}".format(i), 
                pretrain_step=-1, 
                lr_decay=float(lr_decay), 
                lr_decay_step=1000, 
                lr_lower_bound=lr_lower_bound, 
                step=1, test_data=build_test_data(test_list))



        