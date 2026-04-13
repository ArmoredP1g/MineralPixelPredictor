from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradeWeightedSupConLoss(nn.Module):
    """Continuous-label supervised contrastive loss with grade-distance weighting."""

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        if features is None or labels is None or sample_indices is None:
            return torch.tensor(0.0, device=features.device if features is not None else "cpu")

        if features.ndim != 2:
            raise ValueError("features must have shape [N, D].")

        labels = labels.reshape(-1)
        sample_indices = sample_indices.reshape(-1)
        if not (features.shape[0] == labels.shape[0] == sample_indices.shape[0]):
            raise ValueError("features, labels, and sample_indices must have the same length.")

        n = features.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=features.device)

        z = F.normalize(features, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        eye_mask = torch.eye(n, dtype=torch.bool, device=features.device)
        pos_mask = (sample_indices.unsqueeze(0) == sample_indices.unsqueeze(1)) & (~eye_mask)
        neg_mask = (sample_indices.unsqueeze(0) != sample_indices.unsqueeze(1))

        if not torch.any(pos_mask):
            return torch.tensor(0.0, device=features.device)

        label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
        diff_min = torch.min(label_diff)
        diff_max = torch.max(label_diff)
        diff_range = torch.clamp(diff_max - diff_min, min=self.eps)
        grade_weights = (label_diff - diff_min) / diff_range

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (~eye_mask)

        pos_sum = (exp_logits * pos_mask).sum(dim=1)
        neg_sum = (exp_logits * neg_mask * grade_weights).sum(dim=1)

        valid = pos_sum > self.eps
        if not torch.any(valid):
            return torch.tensor(0.0, device=features.device)

        denom = pos_sum + neg_sum + self.eps
        loss = -torch.log((pos_sum + self.eps) / denom)
        return loss[valid].mean()


class ReconstructionSupConLoss(nn.Module):
    """L = MSE + lambda * GradeWeightedSupCon."""

    def __init__(self, lambda_supcon: float = 0.1, temperature: float = 0.07) -> None:
        super().__init__()
        self.lambda_supcon = lambda_supcon
        self.mse = nn.MSELoss()
        self.supcon = GradeWeightedSupConLoss(temperature=temperature)

    def forward(
        self,
        reconstructed: torch.Tensor,
        inputs: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_indices: torch.Tensor,
    ):
        mse_loss = self.mse(reconstructed, inputs)

        if features is None or labels is None or sample_indices is None:
            supcon_loss = torch.tensor(0.0, device=mse_loss.device)
            total = mse_loss
        else:
            supcon_loss = self.supcon(features, labels, sample_indices)
            total = mse_loss + self.lambda_supcon * supcon_loss

        return {"loss": total, "mse": mse_loss, "supcon": supcon_loss}
