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
            

def train_classifier(train_loader, test_loader, model, epoch, lr=0.001, tag="unamed", vis=None):
    sum_writer = SummaryWriter("./runs/{}".format(tag))
    
    optimizer = AdamW(model.parameters(),
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
            # print(total_step)

            loss_sum += loss
            if total_step%50 == 0:
                print("step:{}  loss:{}".format(total_step,loss))

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
                            

def train_regression_mix(train_loader, model, epoch, lr=0.001, tag="unamed", lr_decay=0, lr_decay_step=5000, lr_lower_bound=5e-7, step=0, vis=None):
    # 评估相关
    pool = torch.nn.AvgPool2d(3,3)
    mask_rgb_values = [[255,242,0],[34,177,76],[255,0,88]]
    spec_id = [14,19,54]
    mask_list = []
    tensor_list = []
    label_list = []
    for id in spec_id:
        img = spectral.envi.open("E:\\近红外部分\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr".format(id))

        # 根据模型使用波段选择
        # img_data = torch.Tensor(img.asarray()/6000)[:,:,:-4]
        img_data = torch.Tensor(img.asarray()/6000)[:,:,:]
        # img_data = pool(img_data.permute(1,2,0)).permute(2,0,1)
        mask = np.array(Image.open("E:\\近红外部分\\spectral_data\\{}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png".format(id)))
        mask_list.append(mask)
        gt_TFe = ast.literal_eval(img.metadata['gt_TFe'])
        label_list.append(gt_TFe)

        with torch.no_grad():
            img_data = pool(img_data.permute(2,0,1)).permute(1,2,0)
            tensor_list.append(img_data)


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

    for epoch in range(epoch):
        for _, (data, gt) in enumerate(train_loader, 0):
            total_step+=1
            avg_label = torch.Tensor([
                torch.Tensor(gt['gt_TFe']).mean()
            ]).to(device)


            optimizer.zero_grad() 
            output = model(data.to(device).squeeze(0))[:,0].unsqueeze(1)  # [450, tasks]
            # MSE_loss = mse_loss(torch.log(output+1e-6).mean(dim=0),torch.log(avg_label))
            MSE_loss = mse_loss(output.mean(dim=0),avg_label)

            loss = MSE_loss
            loss.backward()
            optimizer.step()
            print(total_step)
            loss_sum += loss
            MSE_loss_sum += MSE_loss

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


                loss_sum = 0
                MSE_loss_sum = 0

            # 可视化内容
            if total_step%500 == 0:
                if vis != None:
                    vis(sum_writer, total_step)

            if total_step % 500 == 0:
                # 测试样本绝对误差
                model.eval()
                err = 0
                err_count = 0

                for idx in range(tensor_list.__len__()):
                    with torch.no_grad():
                        row,col,_ = tensor_list[idx].shape
                        heat_map = []

                        for i in range(row):
                            heat_map.append(model(tensor_list[idx][i].to(device)).squeeze(1).unsqueeze(0).to("cpu"))   # 只评估Tfe
                            torch.cuda.empty_cache()
                            if (i+1)%50 == 0:
                                print("\r已生成{}行结果".format(i+1), end="")
                        
                        print("")

                    heat_map = torch.cat(heat_map, dim=0)

                    predict_sum = torch.Tensor([0.,0.,0.])
                    pixel_count = torch.Tensor([0, 0, 0])
                    gt = torch.Tensor(label_list[idx])

                    values = [[],[],[]] # 分析数据分布

                    for r in range(row):
                        for c in range(col):
                            for i in range(3):
                                if mask_list[idx][r*3+1,c*3+1].tolist() == mask_rgb_values[i]:  # 对应可见光部分得改
                                    predict_sum[i] += heat_map[r,c] 
                                    values[i].append(heat_map[r,c])    # 分析数据分布
                                    pixel_count[i] += 1

                    prediction = predict_sum / pixel_count * 100

                    err_list = ((prediction-gt)**2).tolist()
                    for e in err_list:
                        err += e
                        err_count += 1


                avg_err = err/(err_count)
                sum_writer.add_scalar(tag='平均绝对误差',
                                scalar_value=avg_err,
                                global_step=total_step
                            )
                torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))

                model.train()
            lr_adjuster.step(total_step, optimizer)

    torch.save(model.state_dict(), "./ckpt/{}_{}.pt".format(tag, total_step))

