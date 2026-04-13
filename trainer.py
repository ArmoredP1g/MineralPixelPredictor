import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter

from configs.training_cfg import ae_step_per_fold, ckpt_path, device, pd_step_per_fold, session_tag


def delete_files_with_prefix(directory, prefix):
    if not os.path.isdir(directory):
        return
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                file_path = os.path.join(root, file)
                os.remove(file_path)


def _resolve_session_dir(session_name: Optional[str] = None):
    resolved_session = session_tag if session_name is None else session_name
    path = os.path.join(ckpt_path, resolved_session)
    os.makedirs(path, exist_ok=True)
    return path


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, global_range: float):
    diff = y_pred - y_true
    mse = np.mean(np.square(diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    re = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)
    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else float("nan")
    nrmse = rmse / global_range if global_range > 1e-8 else float("nan")
    return {"rmse": rmse, "mae": mae, "re": re, "r2": r2, "nrmse": nrmse}


def evaluate_predictor_metrics(model, test_data, chunk_size=100):
    model.eval()

    y_true = []
    y_sample = []
    y_pixel_mean = []
    y_pixel_median = []
    details = []

    with torch.no_grad():
        for d in test_data:
            sample_pixels = d["tensor"]
            gt_value = float(d["gt"].detach().cpu().item())

            cur = 0
            predictions = []
            pixel_count = sample_pixels.shape[0]
            while cur < pixel_count:
                end = min(cur + chunk_size, pixel_count)
                predictions.append(model(sample_pixels[cur:end]).detach().cpu().view(-1))
                cur = end

            pixelwise_prediction = torch.cat(predictions, dim=0).numpy() * 100.0
            sample_pred = float(np.mean(pixelwise_prediction))
            pixel_mean_pred = float(np.mean(pixelwise_prediction))
            pixel_median_pred = float(np.median(pixelwise_prediction))

            y_true.append(gt_value)
            y_sample.append(sample_pred)
            y_pixel_mean.append(pixel_mean_pred)
            y_pixel_median.append(pixel_median_pred)

            details.append(
                {
                    "sample_id": d.get("sample_id", "unknown"),
                    "y_true": gt_value,
                    "sample_pred": sample_pred,
                    "pixel_mean_pred": pixel_mean_pred,
                    "pixel_median_pred": pixel_median_pred,
                    "sample_abs_err": abs(sample_pred - gt_value),
                    "pixel_mean_abs_err": abs(pixel_mean_pred - gt_value),
                    "pixel_median_abs_err": abs(pixel_median_pred - gt_value),
                }
            )

    y_true = np.asarray(y_true, dtype=np.float32)
    y_sample = np.asarray(y_sample, dtype=np.float32)
    y_pixel_mean = np.asarray(y_pixel_mean, dtype=np.float32)
    y_pixel_median = np.asarray(y_pixel_median, dtype=np.float32)

    global_range = float(np.max(y_true) - np.min(y_true)) if y_true.size else float("nan")
    return {
        "test_sample": _compute_metrics(y_true, y_sample, global_range),
        "pixel_mean": _compute_metrics(y_true, y_pixel_mean, global_range),
        "pixel_median": _compute_metrics(y_true, y_pixel_median, global_range),
        "details": details,
    }


class learning_rate_adjuster():
    def __init__(self, lr_decay, start_lr, update_step, lower_limit) -> None:
        self.stop_sign = False
        self.lr_decay = lr_decay
        self.cur_lr = start_lr
        self.update_step = update_step
        self.lower_limit = lower_limit

    def step(self, cur_step, optimizer):
        if self.stop_sign or self.update_step <= 0:
            return

        if self.stop_sign == False and cur_step % self.update_step == 0:
            if self.cur_lr * self.lr_decay <= self.lower_limit:
                self.cur_lr = self.lower_limit
                self.stop_sign = True
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.cur_lr
                print("lr update to：{}".format(self.cur_lr))
            else:
                self.cur_lr *= self.lr_decay
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.cur_lr
                print("lr update to：{}".format(self.cur_lr))


def train_AE(
    train_loader,
    model,
    fold,
    lr=0.001,
    tag="unamed",
    lr_decay_step=5000,
    step=0,
    criterion=None,
    session_name=None,
    checkpoint_mode="legacy",
    max_steps=None,
    return_info=False,
):
    del lr_decay_step
    sum_writer = SummaryWriter("./runs/{}".format(tag))
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.005,
        amsgrad=False,
    )

    total_step = step
    target_steps = ae_step_per_fold if max_steps is None else int(max_steps)
    base_mse_loss = torch.nn.MSELoss()
    objective = base_mse_loss if criterion is None else criterion

    loss_sum = 0.0
    mse_loss_sum = 0.0
    supcon_loss_sum = 0.0

    best_loss = float("inf")
    best_step = None
    best_checkpoint = None

    for _, batch in enumerate(train_loader, 0):
        if total_step > target_steps:
            break

        if len(batch) == 3:
            data, gt, sample_indices = batch
        else:
            data, gt = batch
            sample_indices = None

        if data.dim() == 2:
            data = data.unsqueeze(0)

        total_step += 1
        optimizer.zero_grad()

        data = data.to(device)
        sample_count = data.shape[0]
        pixel_count = data.shape[1]
        flat_data = data.reshape(-1, data.shape[-1])

        flat_labels = None
        if gt is not None:
            gt_tensor = torch.as_tensor(gt, dtype=torch.float32, device=device).reshape(-1)
            if gt_tensor.numel() == sample_count:
                flat_labels = gt_tensor.unsqueeze(1).repeat(1, pixel_count).reshape(-1)

        flat_sample_indices = None
        if sample_indices is not None:
            sample_tensor = torch.as_tensor(sample_indices, dtype=torch.long, device=device).reshape(-1)
            if sample_tensor.numel() == sample_count:
                flat_sample_indices = sample_tensor.unsqueeze(1).repeat(1, pixel_count).reshape(-1)

        if isinstance(model, torch.nn.Sequential) and len(model) >= 2:
            encoded = model[0](flat_data)
            output = model[1](encoded)
        else:
            encoded = None
            output = model(flat_data)

        mse_component = base_mse_loss(flat_data, output)
        supcon_component = torch.tensor(0.0, device=output.device)

        if isinstance(objective, torch.nn.MSELoss):
            loss = objective(flat_data, output)
            mse_component = loss
        else:
            loss_out = objective(output, flat_data, encoded, flat_labels, flat_sample_indices)
            if isinstance(loss_out, dict):
                loss = loss_out.get("loss", mse_component)
                mse_component = loss_out.get("mse", mse_component)
                supcon_component = loss_out.get("supcon", supcon_component)
            elif isinstance(loss_out, tuple):
                if len(loss_out) == 3:
                    loss, mse_component, supcon_component = loss_out
                elif len(loss_out) == 2:
                    loss, mse_component = loss_out
                else:
                    loss = loss_out[0]
            else:
                loss = loss_out

        loss.backward()
        optimizer.step()

        loss_value = _to_float(loss)
        loss_sum += loss_value
        mse_loss_sum += _to_float(mse_component)
        supcon_loss_sum += _to_float(supcon_component)

        if checkpoint_mode == "best_last" and loss_value < best_loss:
            best_loss = loss_value
            best_step = total_step
            best_checkpoint = os.path.join(_resolve_session_dir(session_name), "encoder_fold{}_best.pt".format(fold))
            torch.save(model.state_dict(), best_checkpoint)

        if total_step % 50 == 0:
            print("step:{}  loss:{}".format(total_step, loss_value))

        if total_step % 200 == 0:
            sum_writer.add_scalar(tag="encoder loss", scalar_value=loss_sum / 200.0, global_step=total_step)
            sum_writer.add_scalar(tag="encoder_mse_loss", scalar_value=mse_loss_sum / 200.0, global_step=total_step)
            sum_writer.add_scalar(
                tag="encoder_supcon_loss",
                scalar_value=supcon_loss_sum / 200.0,
                global_step=total_step,
            )
            loss_sum = 0.0
            mse_loss_sum = 0.0
            supcon_loss_sum = 0.0

    session_dir = _resolve_session_dir(session_name)
    if checkpoint_mode == "best_last":
        last_checkpoint = os.path.join(session_dir, "encoder_fold{}_last.pt".format(fold))
        torch.save(model.state_dict(), last_checkpoint)
        if best_checkpoint is None:
            best_checkpoint = last_checkpoint
            best_step = total_step
            best_loss = loss_value if total_step > step else float("nan")
    else:
        last_checkpoint = os.path.join(session_dir, "encoder_fold{}_step{}.pt".format(fold, total_step))
        torch.save(model.state_dict(), last_checkpoint)
        best_checkpoint = last_checkpoint
        best_step = total_step
        best_loss = loss_value if total_step > step else float("nan")

    info = {
        "final_step": total_step,
        "best_step": best_step,
        "best_loss": best_loss,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
    }
    if return_info:
        return model[0], info
    return model[0]


def train_predictor(
    train_loader,
    encoder,
    predictor,
    fold,
    lr=0.001,
    tag="unamed",
    pretrain_step=0,
    lr_decay=0.0,
    lr_decay_step=5000,
    lr_lower_bound=5e-7,
    step=0,
    test_data=None,
    vis=None,
    freeze=True,
    session_name=None,
    checkpoint_mode="legacy",
    max_steps=None,
):
    print("Training predictor for fold {}".format(fold))
    for param in encoder.parameters():
        param.requires_grad = not (freeze is True)

    model = torch.nn.Sequential(encoder, predictor)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.005,
        amsgrad=False,
    )

    sum_writer = SummaryWriter("./runs/{}".format(tag))

    lr_adjuster = learning_rate_adjuster(lr_decay, start_lr=lr, update_step=lr_decay_step, lower_limit=lr_lower_bound)
    mse_loss = torch.nn.MSELoss()

    total_step = step
    target_steps = pd_step_per_fold if max_steps is None else int(max_steps)

    loss_sum = 0.0
    mse_loss_sum = 0.0

    best_rmse = float("inf")
    best_step = None
    best_checkpoint = None
    best_metrics = None

    for _, batch in enumerate(train_loader, 0):
        if total_step > target_steps:
            break

        data = batch[0]
        gt = batch[1]

        if data.dim() != 3 or data.shape[0] != 1:
            raise ValueError("Predictor training expects DataLoader batch_size=1 and data shape [1, N, C].")

        if total_step == pretrain_step and hasattr(model, "decoder1"):
            model.decoder1.pretrain_off()

        total_step += 1
        avg_label = torch.as_tensor(gt, dtype=torch.float32, device=device).reshape(-1)

        optimizer.zero_grad()
        output = model(data.to(device).squeeze(0))
        current_mse_loss = mse_loss(output.mean(dim=0), avg_label)
        loss = current_mse_loss
        loss.backward()
        optimizer.step()

        loss_sum += _to_float(loss)
        mse_loss_sum += _to_float(current_mse_loss)

        if total_step % 50 == 0:
            print("step:{}  loss:{}".format(total_step, _to_float(loss)))

        if total_step % 200 == 0:
            sum_writer.add_scalar(tag="loss", scalar_value=loss_sum / 200.0, global_step=total_step)
            sum_writer.add_scalar(tag="MSE_loss", scalar_value=mse_loss_sum / 200.0, global_step=total_step)
            loss_sum = 0.0
            mse_loss_sum = 0.0

        if total_step % 500 == 0 and vis is not None:
            vis(sum_writer, total_step)

        if total_step % 1000 == 0 and test_data is not None:
            metrics = evaluate_predictor_metrics(model, test_data)
            test_metrics = metrics["test_sample"]

            sum_writer.add_scalar(tag="MAE", scalar_value=test_metrics["mae"], global_step=total_step)
            sum_writer.add_scalar(tag="RE", scalar_value=test_metrics["re"], global_step=total_step)
            sum_writer.add_scalar(tag="RMSE", scalar_value=test_metrics["rmse"], global_step=total_step)
            sum_writer.add_scalar(tag="NRMSE", scalar_value=test_metrics["nrmse"], global_step=total_step)
            sum_writer.add_scalar(tag="R2", scalar_value=test_metrics["r2"], global_step=total_step)

            session_dir = _resolve_session_dir(session_name)
            if checkpoint_mode == "best_last":
                if test_metrics["rmse"] < best_rmse:
                    best_rmse = test_metrics["rmse"]
                    best_step = total_step
                    best_metrics = test_metrics
                    best_checkpoint = os.path.join(session_dir, "fold{}_best.pt".format(fold))
                    torch.save(model.state_dict(), best_checkpoint)
            else:
                delete_files_with_prefix(session_dir, "fold{}".format(fold))
                step_checkpoint = os.path.join(session_dir, "fold{}_step{}.pt".format(fold, total_step))
                torch.save(model.state_dict(), step_checkpoint)

            model.train()

        lr_adjuster.step(total_step, optimizer)

    session_dir = _resolve_session_dir(session_name)
    if checkpoint_mode == "best_last":
        last_checkpoint = os.path.join(session_dir, "fold{}_last.pt".format(fold))
        torch.save(model.state_dict(), last_checkpoint)

        if test_data is not None and best_checkpoint is None:
            metrics = evaluate_predictor_metrics(model, test_data)
            best_rmse = metrics["test_sample"]["rmse"]
            best_step = total_step
            best_metrics = metrics["test_sample"]
            best_checkpoint = os.path.join(session_dir, "fold{}_best.pt".format(fold))
            torch.save(model.state_dict(), best_checkpoint)
    else:
        last_checkpoint = os.path.join(session_dir, "fold{}_step{}.pt".format(fold, total_step))
        torch.save(model.state_dict(), last_checkpoint)
        best_checkpoint = last_checkpoint
        best_step = total_step
        best_rmse = float("nan")

    return {
        "final_step": total_step,
        "best_step": best_step,
        "best_rmse": best_rmse,
        "best_metrics": best_metrics,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
    }
