from torch.optim import AdamW
from torch.cuda.amp import autocast as autocast
import torch
import os
import numpy as np
import spectral
from spectral import imshow
import ast
from PIL import Image
from torch.nn import CrossEntropyLoss
# from models.loss import Lognorm_Loss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from configs.training_cfg import *
spectral.settings.envi_support_nonlowercase_params = True

def delete_files_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                file_path = os.path.join(root, file)
                os.remove(file_path)

class learning_rate_adjuster():
    def __init__(self, lr_decay, start_lr, update_step, lower_limit) -> None:
        self.stop_sign = False
        self.lr_decay = lr_decay
        self.cur_lr = start_lr
        self.update_step = update_step
        self.lower_limit = lower_limit

    def step(self, cur_step, optimizer):
        if self.stop_sign:
            return

        if self.stop_sign==False and cur_step%self.update_step == 0:
            if self.cur_lr * self.lr_decay <= self.lower_limit:
                self.cur_lr = self.lower_limit
                self.stop_sign = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.cur_lr
                print("lr update to：{}".format(self.cur_lr))
            else:
                self.cur_lr *= self.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.cur_lr
                print("lr update to：{}".format(self.cur_lr))
            


def train_regression(train_loader, model, fold, lr=0.001, tag="unamed", pretrain_step=0, lr_decay=0, lr_decay_step=5000, lr_lower_bound=5e-7, step=0, test_data=None, vis=None):
    # 训练相关
    sum_writer = SummaryWriter("./runs/{}".format(tag))
    optimizer = AdamW(model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0.005,
                    amsgrad=False)
    lr_adjuster = learning_rate_adjuster(lr_decay, start_lr=lr, update_step=lr_decay_step, lower_limit=lr_lower_bound)
    # loss_fn=torch.nn.HuberLoss(reduction='mean', delta=8)
    mse_loss = torch.nn.MSELoss()

    total_step = step

    loss_sum = 0
    MSE_loss_sum = 0
    transboundary_loss_sum = 0

    for _, (data, gt) in enumerate(train_loader, 0):
        if total_step == pretrain_step:
            model.decoder1.pretrain_off()   # 结束预训练

        total_step+=1
        avg_label = torch.Tensor([
            torch.Tensor(gt)
        ]).to(device)


        optimizer.zero_grad() 
        output, transboundary_loss = model(data.to(device).squeeze(0))  # [batch, tasks]
        # MSE_loss = mse_loss(torch.log(output+1e-6).mean(dim=0),torch.log(avg_label))
        MSE_loss = mse_loss(output.mean(dim=0),avg_label)

        loss = MSE_loss + transboundary_loss*10
        loss.backward()
        optimizer.step()
        # print(total_step)
        loss_sum += loss
        MSE_loss_sum += MSE_loss
        transboundary_loss_sum += transboundary_loss

        if total_step%50 == 0:
            print("step:{}  loss:{}".format(total_step,loss))

        if total_step%200 == 0:
            sum_writer.add_scalar(tag='loss',
                                    scalar_value=loss_sum / 200,
                                    global_step=total_step
                                )
            sum_writer.add_scalar(tag='MSE_loss',
                                    scalar_value=MSE_loss_sum / 200,
                                    global_step=total_step
                                )
            sum_writer.add_scalar(tag='TransBoundary_loss',
                                    scalar_value=transboundary_loss_sum / 200,
                                    global_step=total_step
                                )

            loss_sum = 0
            MSE_loss_sum = 0
            transboundary_loss_sum = 0

        # 可视化内容
        if total_step%500 == 0:
            if vis != None:
                vis(sum_writer, total_step)

        if total_step % 1000 == 0:
            # 测试样本绝对+相对误差
            model.eval()
            err = 0
            re = 0
            err_square = 0
            err_count = 0
            gt_avg = 0
            gt_all = []

            for d in test_data:
                data = d['tensor']
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
                prediction = pixelwise_prediction.mean()*100
                err += torch.abs(gt - prediction)
                err_square += (gt - prediction)**2
                re += 100*torch.abs(gt - prediction)/gt
                err_count += 1
                gt_avg += gt
                gt_all.append(gt)
                print("测试样本")


            avg_err = err/(err_count)
            avg_re = re/(err_count)
            gt_avg = gt_avg/(err_count)
            tss = 0
            for i in gt_all:
                tss += (i-gt_avg) ** 2

            rss = err_square
            R2 = 1-(rss/tss)

            rmse = torch.sqrt(err_square/err_count)
            nrmse = rmse/(max(gt_all)-min(gt_all))

            sum_writer.add_scalar(tag='MAE',
                            scalar_value=avg_err,
                            global_step=total_step
                        )
            
            sum_writer.add_scalar(tag='RE',
                            scalar_value=avg_re,
                            global_step=total_step
                        )

            sum_writer.add_scalar(tag='RMSE',
                            scalar_value=rmse,
                            global_step=total_step
                        )
            
            sum_writer.add_scalar(tag='NRMSE',
                            scalar_value=nrmse,
                            global_step=total_step
                        )
            sum_writer.add_scalar(tag='R2',
                            scalar_value=R2,
                            global_step=total_step
                        )
            
            delete_files_with_prefix(ckpt_path+"/"+session_tag, "fold{}".format(fold))            
            torch.save(model.state_dict(), ckpt_path+"/"+session_tag+"/fold{}_step{}.pt".format(fold, total_step))

            model.train()
        lr_adjuster.step(total_step, optimizer)
