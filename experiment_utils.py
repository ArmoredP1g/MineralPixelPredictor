from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import spectral
import torch
from PIL import Image


spectral.settings.envi_support_nonlowercase_params = True


MASK_RGB_VALUES = [[255, 242, 0], [34, 177, 76], [255, 0, 88]]
POOL = torch.nn.AvgPool2d(3, 3)


ALL_SAMPLES = [
    "13_A",
    "11_A",
    "12_A",
    "12_C",
    "25_A",
    "42_B",
    "55_C",
    "15_A",
    "56_B",
    "4_B",
    "42_A",
    "57_A",
    "14_B",
    "36_B",
    "43_C",
    "26_A",
    "9_C",
    "43_A",
    "53_A",
    "3_B",
    "30_C",
    "27_A",
    "22_B",
    "27_C",
    "31_C",
    "53_B",
    "32_A",
    "6_B",
    "52_B",
    "8_B",
    "41_B",
    "31_A",
    "34_A",
    "7_B",
    "53_C",
    "54_C",
    "29_B",
    "16_B",
    "47_A",
    "49_B",
    "10_C",
    "21_C",
    "50_A",
    "18_A",
    "22_C",
    "52_C",
    "38_A",
    "17_A",
    "59_A",
    "4_A",
    "57_B",
    "33_C",
    "7_A",
    "49_C",
    "58_B",
    "4_C",
    "52_A",
    "17_C",
    "23_A",
    "7_C",
    "46_B",
    "30_B",
    "46_A",
    "18_C",
    "24_A",
    "55_A",
    "40_A",
    "55_B",
    "6_A",
    "59_B",
    "3_C",
    "27_B",
    "18_B",
    "5_A",
    "29_A",
    "25_B",
    "49_A",
    "32_C",
    "45_C",
    "12_B",
    "20_A",
    "9_A",
    "28_C",
    "29_C",
    "5_C",
    "46_C",
    "14_C",
    "19_A",
    "23_B",
    "9_B",
    "40_B",
    "35_C",
    "13_C",
    "50_B",
    "35_B",
    "15_B",
    "45_A",
    "23_C",
    "1_C",
    "1_B",
    "35_A",
    "32_B",
    "6_C",
    "51_B",
    "28_B",
    "2_B",
    "58_C",
    "38_C",
    "2_A",
    "26_B",
    "2_C",
    "16_C",
    "43_B",
    "24_C",
    "54_B",
    "15_C",
    "42_C",
    "36_A",
    "37_A",
    "41_C",
    "44_B",
    "19_C",
    "51_C",
    "1_A",
    "39_B",
    "28_A",
    "39_C",
    "30_A",
    "39_A",
    "54_A",
    "61_C",
    "61_A",
    "37_B",
    "48_C",
    "21_A",
    "22_A",
    "48_B",
    "48_A",
    "14_A",
    "11_C",
    "16_A",
    "13_B",
    "21_B",
    "36_C",
    "34_B",
    "56_A",
    "47_C",
    "8_A",
    "19_B",
    "3_A",
    "62_A",
    "33_B",
    "10_A",
    "24_B",
    "60_B",
    "11_B",
    "59_C",
    "10_B",
    "8_C",
    "41_A",
    "38_B",
    "57_C",
    "61_B",
    "37_C",
]


def spectral_header_path(dataset_root: str, img_id: str) -> Path:
    return Path(dataset_root) / "spectral_data" / (
        f"{img_id}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr"
    )


def spectral_mask_path(dataset_root: str, img_id: str) -> Path:
    return Path(dataset_root) / "spectral_data" / (
        f"{img_id}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png"
    )


def sample_target_value(sample_id: str, dataset_root: str) -> float:
    img_id, sample_suffix = sample_id.split("_")
    sample_idx = ord(sample_suffix) - 65
    metadata = spectral.envi.open(str(spectral_header_path(dataset_root, img_id))).metadata
    gt_values = ast.literal_eval(metadata["gt_TFe"])
    return float(gt_values[sample_idx])


def build_ranked_sample_index(sample_ids: Sequence[str], dataset_root: str, device: str) -> List[Dict[str, object]]:
    sample_index: List[Dict[str, object]] = []
    for sample_id in sample_ids:
        gt_value = sample_target_value(sample_id, dataset_root)
        sample_index.append(
            {
                "id": sample_id,
                "gt_value": gt_value,
                "gt": torch.Tensor([gt_value]).to(device),
            }
        )
    return sample_index


def create_target_balanced_folds(sample_index: Sequence[Dict[str, object]], fold_count: int = 5) -> List[List[str]]:
    sorted_index = sorted(sample_index, key=lambda item: float(item["gt_value"]))
    folds: List[List[str]] = [[] for _ in range(fold_count)]
    for rank, item in enumerate(sorted_index):
        folds[rank % fold_count].append(str(item["id"]))
    return folds


def prepare_test_data(sample_ids: Sequence[str], dataset_root: str, device: str) -> List[Dict[str, torch.Tensor]]:
    test_data_list = []
    for sample_id in sample_ids:
        pixel_list = []
        img_id, sample_suffix = sample_id.split("_")
        sample_idx = ord(sample_suffix) - 65
        img_data = spectral.envi.open(str(spectral_header_path(dataset_root, img_id)))
        gt_values = ast.literal_eval(img_data.metadata["gt_TFe"])
        tensor = torch.Tensor(img_data.asarray() / 6000)[:, :, :]
        pooled_tensor = POOL(tensor.permute(2, 0, 1)).permute(1, 2, 0)
        mask = np.array(Image.open(spectral_mask_path(dataset_root, img_id)))
        row, col, _ = pooled_tensor.shape

        for row_idx in range(row):
            for col_idx in range(col):
                if mask[row_idx * 3 + 1, col_idx * 3 + 1].tolist() == MASK_RGB_VALUES[sample_idx]:
                    pixel_list.append(pooled_tensor[row_idx, col_idx].unsqueeze(0))

        pixel_tensor = torch.cat(pixel_list, dim=0)
        test_data_list.append(
            {
                "sample_id": sample_id,
                "tensor": pixel_tensor.to(device),
                "gt": torch.Tensor([gt_values[sample_idx]]).to(device),
            }
        )
    return test_data_list