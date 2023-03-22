from torch.optim import AdamW
from torch.cuda.amp import autocast as autocast
import torch
import numpy as np
import spectral
from spectral import imshow
import ast
from PIL import Image
from torch.nn import CrossEntropyLoss
from models.loss import Lognorm_Loss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from configs.training_cfg import device
spectral.settings.envi_support_nonlowercase_params = True

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
            


def train_regression(train_loader, model, epoch, lr=0.001, tag="unamed", pretrain_step=0, lr_decay=0, lr_decay_step=5000, lr_lower_bound=5e-7, step=0, test_data=None, vis=None):
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

    for epoch in range(epoch):
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

                for d in test_data:
                    data = d['tensor']
                    gt = d['gt']
                    with torch.no_grad():
                        len = data.shape[0]
                        cur = 0
                        pixelwise_prediction = []
                        while cur + 100 <= len:
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
                    print("测试样本")


                avg_err = err/(err_count)
                avg_re = re/(err_count)
                rmse = torch.sqrt(err_square/err_count)
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
                            
                torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))

                model.train()
            lr_adjuster.step(total_step, optimizer)

    torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))


#   多任务训练流程
def train_regression_multitask(train_loader, model, epoch, lr=0.001, tag="unamed", pretrain_step=0, lr_decay=0, lr_decay_step=5000, lr_lower_bound=5e-7, step=0, test_data=None, vis=None):
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

    for epoch in range(epoch):
        for _, (data, gt) in enumerate(train_loader, 0):
            if total_step == pretrain_step:
                model.decoder1.pretrain_off()   # 结束预训练

            total_step+=1
            avg_label = torch.Tensor([
                torch.Tensor(gt['TFe']),
                torch.Tensor(gt['Fe3']),
                torch.Tensor(gt['SiO2'])
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
                # 测试样本绝对误差
                model.eval()
                err = 0
                err_count = 0

                for d in test_data:
                    data = d['tensor']
                    gt = d['gt']
                    with torch.no_grad():
                        len = data.shape[0]
                        cur = 0
                        pixelwise_prediction = []
                        while cur + 100 <= len:
                            pixelwise_prediction.append(model(data[cur:cur+100]).to("cpu"))
                            cur += 100
                            torch.cuda.empty_cache()

                        if cur != len:
                            pixelwise_prediction.append(model(data[cur:len]).to("cpu"))

                        pixelwise_prediction = torch.cat(pixelwise_prediction, dim=0)
                    prediction = pixelwise_prediction.mean(dim=0)*100
                    err += torch.abs(gt.to(prediction.device) - prediction)
                    err_count += 1
                    print("测试样本")


                avg_err = err/(err_count)
                sum_writer.add_scalar(tag='平均绝对误差',
                                scalar_value=avg_err.mean(),
                                global_step=total_step
                            )
                sum_writer.add_scalar(tag='TFe绝对误差',
                                scalar_value=avg_err[0],
                                global_step=total_step
                            )
                sum_writer.add_scalar(tag='Fe3绝对误差',
                                scalar_value=avg_err[1],
                                global_step=total_step
                            )
                sum_writer.add_scalar(tag='SiO2绝对误差',
                                scalar_value=avg_err[2],
                                global_step=total_step
                            )
                torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))

                model.train()
            lr_adjuster.step(total_step, optimizer)

    torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))