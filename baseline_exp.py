"""Single-threaded randomized search for SVR, PLSR, and XGB on spectral data.

This script preserves the original data loading, pooling, Savitzky-Golay smoothing,
and dimensionality-reduction mechanisms from the previous baseline implementation
while providing a deterministic, single-process workflow to discover high-performing
hyperparameters for three regression models.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import spectral
import torch
from PIL import Image
from scipy.signal import savgol_filter
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap
from sklearn.svm import SVR
from xgboost import XGBRegressor

sys.path.append("../")
from configs.training_cfg import dataset_path  # noqa: E402
from models.sigmoid_plsr import (
    PLSR_TARGET_SCALE,
    SigmoidBoundedPLSRegression,
    SigmoidTargetTransform,
)


spectral.settings.envi_support_nonlowercase_params = True


MASK_RGB_VALUES = torch.tensor(
    [[255.0, 242.0, 0.0], [34.0, 177.0, 76.0], [255.0, 0.0, 88.0]], dtype=torch.float32
)
POOL_SIZE = 3
EPS = 1e-8


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


@dataclass
class DatasetBundle:
    sample_ids: List[str]
    features: np.ndarray
    targets: np.ndarray
    folds: List[List[int]]


def _spectral_header_path(img_id: str) -> Path:
    return Path(dataset_path) / "spectral_data" / (
        f"{img_id}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr"
    )


def _mask_path(img_id: str) -> Path:
    return Path(dataset_path) / "spectral_data" / (
        f"{img_id}-Radiance From Raw Data-Reflectance from Radiance Data and Measured Reference Spectrum.bip.hdr_mask.png"
    )


class SampleAggregator:
    """Loads, pools, and aggregates hyperspectral samples on demand with caching."""

    def __init__(self, pool_size: int = POOL_SIZE):
        self.pool = torch.nn.AvgPool2d(pool_size, pool_size)
        self.image_cache: Dict[str, torch.Tensor] = {}
        self.mask_cache: Dict[str, torch.Tensor] = {}
        self.gt_cache: Dict[str, Sequence[float]] = {}

    def _load_image_bundle(self, img_id: str) -> Tuple[torch.Tensor, torch.Tensor, Sequence[float]]:
        if img_id not in self.image_cache:
            header_path = _spectral_header_path(img_id)
            if not header_path.exists():
                raise FileNotFoundError(f"Spectral header not found: {header_path}")
            img_data = spectral.envi.open(str(header_path))
            tensor = torch.tensor(img_data.asarray() / 6000.0, dtype=torch.float32)
            pooled_img = self.pool(tensor.permute(2, 0, 1)).permute(1, 2, 0).contiguous()

            mask_path = _mask_path(img_id)
            mask_array = np.array(Image.open(mask_path))
            mask_tensor = torch.tensor(mask_array, dtype=torch.float32)
            pooled_mask = self.pool(mask_tensor.permute(2, 0, 1)).permute(1, 2, 0).contiguous()

            gt_values = ast.literal_eval(img_data.metadata["gt_TFe"])

            self.image_cache[img_id] = pooled_img
            self.mask_cache[img_id] = pooled_mask
            self.gt_cache[img_id] = gt_values

        return (
            self.image_cache[img_id],
            self.mask_cache[img_id],
            self.gt_cache[img_id],
        )

    def get_sample(self, sample_id: str) -> Tuple[torch.Tensor, float]:
        img_id, sample_suffix = sample_id.split("_")
        sample_idx = ord(sample_suffix) - 65
        if sample_idx < 0 or sample_idx >= MASK_RGB_VALUES.shape[0]:
            raise ValueError(f"Unexpected sample suffix for {sample_id}")

        pooled_img, pooled_mask, gt_values = self._load_image_bundle(img_id)
        target_rgb = MASK_RGB_VALUES[sample_idx]

        rounded_mask = pooled_mask.round()
        matches = torch.all(rounded_mask == target_rgb, dim=-1)
        if not torch.any(matches):
            matches = torch.all(torch.abs(pooled_mask - target_rgb) <= 1.0, dim=-1)

        pixels = pooled_img[matches]
        if pixels.shape[0] == 0:
            raise RuntimeError(f"No pixels found for sample {sample_id}")

        sample_tensor = pixels.mean(dim=0)
        target_value = float(gt_values[sample_idx])
        return sample_tensor, target_value


def build_feature_vector(sample_tensor: torch.Tensor) -> np.ndarray:
    spectral_values = sample_tensor.flatten()
    if spectral_values.ndim != 1:
        spectral_values = spectral_values.view(-1)
    return spectral_values.cpu().numpy().astype(np.float32)


def build_dataset(sample_ids: Sequence[str]) -> DatasetBundle:
    aggregator = SampleAggregator(pool_size=POOL_SIZE)
    features: List[np.ndarray] = []
    targets: List[float] = []

    for sample_id in sample_ids:
        tensor, target_value = aggregator.get_sample(sample_id)
        features.append(build_feature_vector(tensor))
        targets.append(target_value)

    feature_matrix = np.vstack(features)
    target_array = np.array(targets, dtype=np.float32)

    sorted_indices = np.argsort(target_array, kind="mergesort")
    folds: List[List[int]] = [[] for _ in range(5)]
    for rank, original_idx in enumerate(sorted_indices):
        folds[rank % 5].append(int(original_idx))

    return DatasetBundle(list(sample_ids), feature_matrix, target_array, folds)


def adjust_sg_params(window_length: int, poly_order: int, feature_len: int) -> Tuple[int, int]:
    if feature_len < 3:
        raise ValueError("Feature length too small for Savitzky-Golay filter")

    max_valid_window = feature_len if feature_len % 2 == 1 else feature_len - 1
    window = min(window_length, max_valid_window)
    if window < 3:
        raise ValueError("Savitzky-Golay window must be >= 3")
    if window % 2 == 0:
        window -= 1
    poly = max(1, min(poly_order, window - 1))
    if poly >= window:
        raise ValueError("Savitzky-Golay polynomial order must be < window length")
    return window, poly


def append_first_order_diffs(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("Expected 2D array for first-order difference computation")
    if arr.shape[1] < 2:
        return arr
    diffs = np.diff(arr, axis=1)
    return np.concatenate([arr, diffs], axis=1)


def apply_preprocessing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: Dict[str, Any],
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train_processed = np.copy(X_train)
    val_processed = np.copy(X_val)
    normalized_config: Dict[str, Any] = dict(config)

    if normalized_config.get("use_sg", False):
        window, poly = adjust_sg_params(
            int(normalized_config.get("sg_window", 7)),
            int(normalized_config.get("sg_poly", 3)),
            train_processed.shape[1],
        )
        train_processed = savgol_filter(train_processed, window_length=window, polyorder=poly, axis=1)
        val_processed = savgol_filter(val_processed, window_length=window, polyorder=poly, axis=1)
        normalized_config["use_sg"] = True
        normalized_config["sg_window"] = int(window)
        normalized_config["sg_poly"] = int(poly)
    else:
        normalized_config["use_sg"] = False

    normalization = str(normalized_config.get("normalization", "NONE")).upper()
    if normalization not in {"NONE", "MINMAX", "SNV"}:
        raise ValueError(f"Unsupported normalization method: {normalization}")

    if normalization == "MINMAX":
        feature_min = train_processed.min(axis=0)
        feature_max = train_processed.max(axis=0)
        denom = feature_max - feature_min
        denom[np.abs(denom) < EPS] = 1.0
        train_processed = (train_processed - feature_min) / denom
        val_processed = (val_processed - feature_min) / denom
        val_processed = np.clip(val_processed, 0.0, 1.0)
        normalized_config["normalization"] = "MINMAX"
    elif normalization == "SNV":
        def _snv(arr: np.ndarray) -> np.ndarray:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            std[np.abs(std) < EPS] = 1.0
            return (arr - mean) / std

        train_processed = _snv(train_processed)
        val_processed = _snv(val_processed)
        normalized_config["normalization"] = "SNV"
    else:
        normalized_config["normalization"] = "NONE"

    train_processed = append_first_order_diffs(train_processed)
    val_processed = append_first_order_diffs(val_processed)
    normalized_config["diffs_appended_post_normalization"] = True

    dim_method = normalized_config.get("dim_method", "NONE")
    if model_name == "PLSR":
        dim_method = "NONE"

    if dim_method == "NONE":
        normalized_config["dim_method"] = "NONE"
        return train_processed, val_processed, normalized_config

    n_components = int(normalized_config.get("n_components", min(20, train_processed.shape[1])))
    n_components = max(1, min(n_components, train_processed.shape[1] - 1, train_processed.shape[0]))

    if n_components < 1:
        raise ValueError("Invalid number of components after adjustment")

    normalized_config["n_components"] = int(n_components)

    if dim_method == "PCA":
        reducer = PCA(n_components=n_components)
        normalized_config["dim_method"] = "PCA"
    elif dim_method == "KPCA":
        kernel = normalized_config.get("kpca_kernel", "rbf")
        reducer = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=False)
        normalized_config["dim_method"] = "KPCA"
        normalized_config["kpca_kernel"] = kernel
    elif dim_method == "ISOMAP":
        n_neighbors = int(normalized_config.get("isomap_neighbors", 8))
        n_neighbors = max(2, min(n_neighbors, train_processed.shape[0] - 1))
        reducer = Isomap(n_neighbors=n_neighbors, n_components=n_components, p=int(normalized_config.get("isomap_p", 2)))
        normalized_config["dim_method"] = "ISOMAP"
        normalized_config["isomap_neighbors"] = int(n_neighbors)
        normalized_config["isomap_p"] = int(normalized_config.get("isomap_p", 2))
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {dim_method}")

    train_reduced = reducer.fit_transform(train_processed)
    val_reduced = reducer.transform(val_processed)
    return train_reduced, val_reduced, normalized_config


def instantiate_model(model_name: str, config: Dict[str, Any]) -> Any:
    if model_name == "SVR":
        degree = int(config.get("svr_degree", 3))
        if degree < 1:
            degree = 1
        return SVR(
            C=float(config.get("svr_C", 10.0)),
            epsilon=float(config.get("svr_epsilon", 0.5)),
            kernel=config.get("svr_kernel", "rbf"),
            degree=degree,
        )

    if model_name == "PLSR":
        n_components = int(config.get("plsr_n_components", min(15, config.get("n_components", 10))))
        max_iter = int(config.get("plsr_max_iter", 600))
        tol = float(config.get("plsr_tol", 1e-6))
        return SigmoidBoundedPLSRegression(n_components=n_components, max_iter=max_iter, tol=tol)

    if model_name == "XGB":
        return XGBRegressor(
            n_estimators=int(config.get("xgb_n_estimators", 120)),
            learning_rate=float(config.get("xgb_eta", 0.1)),
            max_depth=int(config.get("xgb_max_depth", 6)),
            subsample=float(config.get("xgb_subsample", 0.8)),
            colsample_bytree=float(config.get("xgb_colsample", 0.8)),
            reg_lambda=float(config.get("xgb_reg_lambda", 1.0)),
            reg_alpha=float(config.get("xgb_reg_alpha", 0.0)),
            gamma=float(config.get("xgb_gamma", 0.0)),
            tree_method="hist",
            n_jobs=1,
            random_state=int(config.get("random_state", 42)),
            verbosity=0,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    global_range: float,
) -> Dict[str, float]:
    diff = y_pred - y_true
    mse = np.mean(np.square(diff))
    rmse = float(np.sqrt(mse))
    target_range = float(np.max(y_true) - np.min(y_true))
    denom_range = global_range if target_range < EPS else target_range
    mae = float(np.mean(np.abs(diff)))
    rel_error = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), EPS)))
    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
    nrmse = rmse / denom_range if denom_range > EPS else float("nan")
    return {"rmse": rmse, "mae": mae, "re": rel_error, "r2": r2, "nrmse": nrmse}


def evaluate_config(
    model_name: str,
    config: Dict[str, Any],
    dataset: DatasetBundle,
    global_range: float,
) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
    features = dataset.features
    targets = dataset.targets

    if len(dataset.folds) < 2:
        raise ValueError("Dataset folds must contain at least two folds (train/test)")

    # Align with bestsvr: use fold 0 as validation/test, remaining folds for training
    val_idx = np.array(dataset.folds[0], dtype=int)
    train_indices = [idx for fold in dataset.folds[1:] for idx in fold]
    train_idx = np.array(train_indices, dtype=int)

    if train_idx.size == 0 or val_idx.size == 0:
        return None

    X_train = features[train_idx]
    y_train = targets[train_idx]
    X_val = features[val_idx]
    y_val = targets[val_idx]

    try:
        X_train_proc, X_val_proc, normalized_config = apply_preprocessing(X_train, X_val, config, model_name)
    except Exception:
        return None

    model = instantiate_model(model_name, normalized_config)

    if model_name == "PLSR":
        max_components = min(X_train_proc.shape[1], X_train_proc.shape[0])
        if getattr(model, "n_components", 1) > max_components:
            model.n_components = max_components
            normalized_config["plsr_n_components"] = max_components
        if model.n_components < 1:
            return None
        normalized_config["plsr_target_transform"] = str(SigmoidTargetTransform())

    try:
        model.fit(X_train_proc, y_train)
    except Exception:
        return None

    predictions = model.predict(X_val_proc)
    if predictions.ndim > 1:
        predictions = predictions.reshape(-1)

    metrics = compute_metrics(y_val, predictions, global_range)
    if math.isnan(metrics["nrmse"]):
        return None

    return metrics, normalized_config


def random_config(model_name: str, rng: random.Random) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    config["use_sg"] = rng.choice([True, False])
    config["sg_window"] = rng.choice([5, 7, 9, 11, 13, 15, 17, 19])
    config["sg_poly"] = rng.choice([2, 3, 4, 5])
    config["normalization"] = rng.choice(["NONE", "MINMAX", "SNV"])

    dim_options = ["NONE", "PCA", "KPCA", "ISOMAP"] if model_name != "PLSR" else ["NONE"]
    config["dim_method"] = rng.choice(dim_options)
    if config["dim_method"] != "NONE":
        config["n_components"] = rng.choice([6, 10, 14, 18, 24, 32, 40])
        config["kpca_kernel"] = rng.choice(["linear", "rbf", "poly", "sigmoid"])
        config["isomap_neighbors"] = rng.choice([5, 7, 9, 11, 13])
        config["isomap_p"] = rng.choice([1, 2, 3])

    if model_name == "SVR":
        config["svr_C"] = rng.uniform(0.1, 200.0)
        config["svr_epsilon"] = rng.uniform(0.01, 3.0)
        config["svr_kernel"] = rng.choice(["rbf", "poly", "sigmoid", "linear"])
        config["svr_degree"] = rng.choice([2, 3, 4, 5, 6])

    elif model_name == "PLSR":
        config["plsr_n_components"] = rng.choice([4, 6, 8, 10, 12, 15, 20, 25])
        config["plsr_max_iter"] = rng.choice([300, 500, 800, 1200, 1600])
        config["plsr_tol"] = rng.choice([1e-6, 5e-6, 1e-5, 5e-5, 1e-4])

    elif model_name == "XGB":
        config["xgb_n_estimators"] = rng.choice([60, 100, 140, 180, 220, 260])
        config["xgb_eta"] = rng.uniform(0.01, 0.3)
        config["xgb_max_depth"] = rng.choice([3, 4, 5, 6, 7, 8])
        config["xgb_subsample"] = rng.uniform(0.5, 1.0)
        config["xgb_colsample"] = rng.uniform(0.5, 1.0)
        config["xgb_reg_lambda"] = rng.uniform(0.1, 5.0)
        config["xgb_reg_alpha"] = rng.uniform(0.0, 2.0)
        config["xgb_gamma"] = rng.uniform(0.0, 2.0)
        config["random_state"] = rng.randint(1, 50_000)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return config


def run_random_search(
    model_name: str,
    dataset: DatasetBundle,
    trials: int,
    rng: random.Random,
) -> Dict[str, Any]:
    global_range = float(np.max(dataset.targets) - np.min(dataset.targets))
    best_result: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []

    for attempt in range(1, trials + 1):
        config = random_config(model_name, rng)

        eval_result = evaluate_config(model_name, config, dataset, global_range)
        if eval_result is None:
            continue

        metrics, normalized_config = eval_result

        record = {"config": normalized_config, "metrics": metrics}
        history.append(record)

        if best_result is None or metrics["nrmse"] < best_result["metrics"]["nrmse"]:
            best_result = record

        if attempt % max(1, trials // 10) == 0 or attempt == trials:
            best_nrmse = best_result["metrics"]["nrmse"] if best_result else float("nan")
            print(
                f"[{model_name}] Trials: {attempt}/{trials} | Current best NRMSE: {best_nrmse:.4f}",
                flush=True,
            )

    if best_result is None:
        raise RuntimeError(f"No valid configurations found for {model_name}")

    return {"best": best_result, "history": history}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomized hyperparameter search for regression models")
    parser.add_argument("--trials", type=int, default=200, help="Number of random configurations per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baseline_random_search_results.json"),
        help="Optional JSON file to store best configurations and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("Loading dataset and constructing folds...", flush=True)
    dataset = build_dataset(ALL_SAMPLES)
    print(
        f"Loaded {len(dataset.sample_ids)} samples | Feature dim: {dataset.features.shape[1]}",
        flush=True,
    )

    results: Dict[str, Any] = {"metadata": {"trials_per_model": args.trials, "seed": args.seed}}

    for model_name in ["SVR", "PLSR", "XGB"]:
        print(f"\n=== Searching best configuration for {model_name} ===", flush=True)
        start_time = time.time()
        search_result = run_random_search(model_name, dataset, args.trials, rng)
        elapsed = time.time() - start_time

        best_metrics = search_result["best"]["metrics"]
        best_config = search_result["best"]["config"]
        print(
            f"Best {model_name} -> NRMSE: {best_metrics['nrmse']:.4f} | RMSE: {best_metrics['rmse']:.4f} | "
            f"R2: {best_metrics['r2']:.4f} | MAE: {best_metrics['mae']:.4f} | RE: {best_metrics['re']:.4f}",
            flush=True,
        )
        print(f"Config: {json.dumps(best_config, indent=2)}", flush=True)
        print(f"Search time: {elapsed/60:.2f} minutes", flush=True)

        results[model_name] = {
            "best_metrics": best_metrics,
            "best_config": best_config,
            "elapsed_seconds": elapsed,
        }

    if args.output:
        try:
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Saved search summary to {args.output}", flush=True)
        except Exception as exc:
            print(f"Failed to write results file: {exc}", flush=True)


if __name__ == "__main__":
    main()





