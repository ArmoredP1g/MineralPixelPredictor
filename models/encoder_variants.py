from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn

from models.models import AE_Encoder


LATENT_DIM = 21
INPUT_BANDS = 168


class StandardCAE(AE_Encoder):
    """Compatibility wrapper over the existing baseline encoder."""


class ASPP_CAE(nn.Module):
    """1D ASPP-style encoder for multi-scale spectral receptive fields."""

    def __init__(self, latent_dim: int = LATENT_DIM, branch_channels: int = 16) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(INPUT_BANDS)
        self.ln2 = nn.LayerNorm(84)
        self.ln3 = nn.LayerNorm(42)

        self.branch_1x1 = nn.Conv1d(1, branch_channels, kernel_size=1, bias=True)
        self.branch_d2 = nn.Conv1d(
            1,
            branch_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=True,
            padding_mode="reflect",
        )
        self.branch_d4 = nn.Conv1d(
            1,
            branch_channels,
            kernel_size=3,
            padding=4,
            dilation=4,
            bias=True,
            padding_mode="reflect",
        )
        self.branch_d8 = nn.Conv1d(
            1,
            branch_channels,
            kernel_size=3,
            padding=8,
            dilation=8,
            bias=True,
            padding_mode="reflect",
        )

        self.fuse = nn.Conv1d(branch_channels * 4, 24, kernel_size=1, bias=True)
        self.conv_reduce1 = nn.Conv1d(24, 12, kernel_size=3, padding=1, bias=True, padding_mode="reflect")
        self.conv_reduce2 = nn.Conv1d(12, 3, kernel_size=3, padding=1, bias=True, padding_mode="reflect")

        self.pool = nn.AvgPool1d(2, 2)
        self.dense = nn.Linear(63, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = x.unsqueeze(1)  # [batch, 1, 168]

        b1 = torch.relu(self.branch_1x1(x))
        b2 = torch.relu(self.branch_d2(x))
        b3 = torch.relu(self.branch_d4(x))
        b4 = torch.relu(self.branch_d8(x))

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = torch.relu(self.fuse(x))

        x = self.pool(x)  # [batch, 24, 84]
        x = self.ln2(x)
        x = torch.relu(self.conv_reduce1(x))

        x = self.pool(x)  # [batch, 12, 42]
        x = self.ln3(x)
        x = torch.relu(self.conv_reduce2(x))

        x = self.pool(x)  # [batch, 3, 21]
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches: int, hidden_dim: int, token_mlp_dim: int, channel_mlp_dim: int) -> None:
        super().__init__()
        self.norm_token = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_patches),
        )

        self.norm_channel = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_mixed = self.norm_token(x).transpose(1, 2)
        token_mixed = self.token_mlp(token_mixed).transpose(1, 2)
        x = x + token_mixed

        channel_mixed = self.channel_mlp(self.norm_channel(x))
        x = x + channel_mixed
        return x


class MixerAE(nn.Module):
    """Fixed patch 1D MLP-Mixer style encoder for 168-band spectra."""

    def __init__(
        self,
        patch_size: int = 12,
        hidden_dim: int = 64,
        token_mlp_dim: int = 64,
        channel_mlp_dim: int = 128,
        depth: int = 2,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        if INPUT_BANDS % patch_size != 0:
            raise ValueError("patch_size must divide 168 exactly for fixed slicing.")

        self.patch_size = patch_size
        self.num_patches = INPUT_BANDS // patch_size
        self.input_norm = nn.LayerNorm(INPUT_BANDS)
        self.patch_embed = nn.Linear(patch_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    num_patches=self.num_patches,
                    hidden_dim=hidden_dim,
                    token_mlp_dim=token_mlp_dim,
                    channel_mlp_dim=channel_mlp_dim,
                )
                for _ in range(depth)
            ]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.reshape(x.shape[0], self.num_patches, self.patch_size)
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.head_norm(x)
        x = x.mean(dim=1)
        return self.head(x)


ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "StandardCAE": StandardCAE,
    "ASPP_CAE": ASPP_CAE,
    "MixerAE": MixerAE,
}


def build_encoder(encoder_type: str, **kwargs) -> nn.Module:
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    return ENCODER_REGISTRY[encoder_type](**kwargs)
