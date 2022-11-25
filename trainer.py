from torch.optim import Adam
from torch.cuda.amp import autocast as autocast
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from models.loss import Lognorm_KL_Loss, Gamma_KL_Loss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from configs.training_cfg import device

class learning_rate_adjuster():
    def __init__(self, start_step, update_step, lr_try_list) -> None:
        self.start_step = start_step
        self.update_step = update_step
        self.lr_try_list = lr_try_list

        self.cur_idx = 0
        self.lr_count = self.lr_try_list.__len__()


    def step(self, cur_step, optimizer):
        if self.cur_idx >= self.lr_count:
            return

        if cur_step >= self.start_step and (cur_step-self.start_step)%self.update_step==0:
            idx = (cur_step-self.start_step)//self.update_step 
            if idx < self.lr_count:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr_try_list[idx]
                print("权重更新为：{}".format(self.lr_try_list[idx]))
            else:
                return
        else:
            return
            

def train_classifier(train_loader, test_loader, model, epoch, lr=0.001, tag="unamed", vis=None):
    sum_writer = SummaryWriter("./runs/{}".format(tag))
    
    optimizer = Adam(model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    # weight_decay=0.00001,
                    amsgrad=False)
    loss_fn = CrossEntropyLoss()
    
    total_step = 0
    loss_sum = 0
    for epoch in range(epoch):
        for _, (data, label) in enumerate(train_loader, 0):
            total_step += 1
            label = torch.Tensor(label).to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(data.to(device))
                loss = loss_fn(output, label-1) #0是无效的
            loss.backward()
            optimizer.step()
            print(total_step)

            loss_sum += loss
            if total_step%50 == 0:
                print("loss: {}".format(loss))

            if total_step%200 == 0:
                sum_writer.add_scalar(tag='loss',
                                        scalar_value=loss_sum / 200,
                                        global_step=total_step
                                    )
                loss_sum = 0

            # 可视化内容
            if total_step%10000 == 0:
                if vis != None:
                    vis(sum_writer, total_step)

            if total_step % 20000 == 0:
                # 测试集测试准确率
                correct = 0
                for  _, (data, label) in enumerate(test_loader, 0):
                    label = torch.Tensor(label).to(device)
                    output = model(data.to(device))
                    prediction = torch.argmax(output, dim=1)
                    correct += (prediction == label-1).sum().item()

                acc = correct/(test_loader.__len__()*8) * 100
                sum_writer.add_scalar(tag='acc',
                                scalar_value=acc,
                                global_step=total_step
                            )
                torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))


    torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))
                               

def train_regression(train_loader, model, epoch, test_loader=False, lr=0.001, tag="unamed", distctrl_factor=0.1, vis=None):
    sum_writer = SummaryWriter("./runs/{}".format(tag))
    optimizer = Adam(model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0.0001,
                    amsgrad=False)
    # lr_adjuster = learning_rate_adjuster(400, 200, [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    # loss_fn=torch.nn.HuberLoss(reduction='mean', delta=8)
    dis_loss = Lognorm_KL_Loss()
    bce_loss = torch.nn.BCELoss()

    total_step = 0
    loss_sum = 0
    BCE_loss_sum = 0
    distribution_loss_sum = 0
    for epoch in range(epoch):
        for _, (data, gt) in enumerate(train_loader, 0):
            # 如需测试学习率
            # lr_adjuster.step(total_step, optimizer)
            total_step += 1
            label = gt['gt_TFe'].float().to(device)
            batch = data.shape[1]
            optimizer.zero_grad() 
            output = model(data.to(device).squeeze(0))
            # loss = loss_fn(output*100, label.unsqueeze(0).repeat(batch,1).float()*100)
            # loss = ((output*100).mean()-label*100).pow(2)
            # loss = -(label*torch.log(output.mean())+(1-label)*torch.log(1-output.mean()))
            BCE_loss = bce_loss(output.mean(dim=0),label)
            # distribution_loss = dis_loss(output.squeeze(1), label)
            # loss = BCE_loss + distctrl_factor*distribution_loss
            loss = BCE_loss
            loss.backward()
            optimizer.step()
            print(total_step)
            loss_sum += loss
            BCE_loss_sum += BCE_loss
            # distribution_loss_sum += distribution_loss
            if total_step%50 == 0:
                print("loss: {}".format(loss.item()))

            if total_step%200 == 0:
                sum_writer.add_scalar(tag='loss',
                                        scalar_value=loss_sum / 200,
                                        global_step=total_step
                                    )
                sum_writer.add_scalar(tag='BCE_loss',
                                        scalar_value=BCE_loss_sum / 200,
                                        global_step=total_step
                                    )
                sum_writer.add_scalar(tag='distribution_loss',
                                        scalar_value=distribution_loss_sum / 200,
                                        global_step=total_step
                                    )
                loss_sum = 0
                BCE_loss_sum = 0
                distribution_loss_sum = 0

            # 可视化内容
            if total_step%1000 == 0:
                if vis != None:
                    vis(sum_writer, total_step)

            if total_step % 1000 == 0:
                # 测试集测试均方误差
                total_mse = 0
                if test_loader:
                    for  _, (data, label) in enumerate(test_loader, 0):
                        label = label['gt_TFe'].to(device)
                        output = model(data.to(device).squeeze(0))
                        total_mse += (output.mean()-label).pow(2)

                    avg_mse = total_mse/(test_loader.__len__()*8)
                    sum_writer.add_scalar(tag='平均MSE',
                                    scalar_value=avg_mse,
                                    global_step=total_step
                                )
                torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))


    torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))