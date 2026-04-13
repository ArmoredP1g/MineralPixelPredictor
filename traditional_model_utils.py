"""Utility helpers to keep classical model scripts aligned with baseline configuration rules."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap

from models.sigmoid_plsr import SigmoidTargetTransform
from classical_baseline_core import DatasetBundle, EPS, adjust_sg_params, append_first_order_diffs, build_dataset, compute_metrics, instantiate_model
from experiment_utils import ALL_SAMPLES


@lru_cache(maxsize=1)
def load_full_dataset() -> DatasetBundle:
    """Load and cache the dataset constructed in ``baseline_exp``."""
    return build_dataset(ALL_SAMPLES)


def split_train_test(dataset: DatasetBundle, test_fold: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the cached dataset into train/test partitions exactly like ``baseline_exp``."""
    if not 0 <= test_fold < len(dataset.folds):
        raise ValueError(f"Invalid test fold index {test_fold}")

    val_idx = np.array(dataset.folds[test_fold], dtype=int)
    train_idx = np.array([idx for fold_id, fold in enumerate(dataset.folds) if fold_id != test_fold for idx in fold], dtype=int)

    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError("Train/test split produced empty partitions")

    features = dataset.features
    targets = dataset.targets

    X_train = features[train_idx]
    y_train = targets[train_idx]
    X_test = features[val_idx]
    y_test = targets[val_idx]

    return X_train, y_train, X_test, y_test


def _normalize_dimension_config(train_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    n_components = int(config.get("n_components", min(20, train_data.shape[1])))
    n_components = max(1, min(n_components, train_data.shape[1] - 1, train_data.shape[0]))
    normalized = dict(config)
    normalized["n_components"] = n_components
    return normalized


def _extract_raw_spectra(
    features: np.ndarray,
    raw_band_count: int,
    baseline_feature_dim: int,
) -> np.ndarray:
    """Coerce incoming features to raw spectral layout prior to preprocessing."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    else:
        array = np.array(array, copy=True)

    if raw_band_count <= 0 and array.shape[1] >= 2:
        return array

    if array.shape[1] == raw_band_count:
        return array

    if baseline_feature_dim > 0 and array.shape[1] == baseline_feature_dim:
        return array[:, :raw_band_count]

    raise ValueError(
        "Feature dimensionality mismatch: expected raw spectra with "
        f"{raw_band_count} bands (or {baseline_feature_dim} columns including diffs), "
        f"but received {array.shape[1]} columns."
    )


def apply_preprocessing_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    config: Dict[str, Any],
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Any]:
    """
    Apply Savitzky-Golay filtering and dimensionality reduction using the same rules as ``baseline_exp``.

    Returns the processed data, the normalized configuration, and the fitted reducer (if any).
    """
    train_processed = np.copy(X_train)
    test_processed = np.copy(X_test)
    normalized_config: Dict[str, Any] = dict(config)
    reducer = None

    if normalized_config.get("use_sg", False):
        window, poly = adjust_sg_params(
            int(normalized_config.get("sg_window", 7)),
            int(normalized_config.get("sg_poly", 3)),
            train_processed.shape[1],
        )
        train_processed = savgol_filter(train_processed, window_length=window, polyorder=poly, axis=1)
        test_processed = savgol_filter(test_processed, window_length=window, polyorder=poly, axis=1)
        normalized_config["use_sg"] = True
        normalized_config["sg_window"] = int(window)
        normalized_config["sg_poly"] = int(poly)
    else:
        normalized_config["use_sg"] = False

    normalization = str(normalized_config.get("normalization", "NONE")).upper()
    if normalization not in {"NONE", "MINMAX", "SNV"}:
        raise ValueError(f"Unsupported normalization method: {normalization}")
    normalized_config["normalization"] = normalization

    normalization_params: Optional[Dict[str, Any]] = None
    raw_band_count = int(train_processed.shape[1])
    if normalization == "MINMAX":
        feature_min = train_processed.min(axis=0)
        feature_max = train_processed.max(axis=0)
        denom = feature_max - feature_min
        denom[np.abs(denom) < EPS] = 1.0
        train_processed = (train_processed - feature_min) / denom
        test_processed = (test_processed - feature_min) / denom
        test_processed = np.clip(test_processed, 0.0, 1.0)
        normalization_params = {"min": feature_min.astype(np.float32), "denom": denom.astype(np.float32)}
    elif normalization == "SNV":

        def _snv(arr: np.ndarray) -> np.ndarray:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            std[np.abs(std) < EPS] = 1.0
            return (arr - mean) / std

        train_processed = _snv(train_processed)
        test_processed = _snv(test_processed)

    normalized_config["normalization_params"] = normalization_params
    normalized_config["baseline_raw_band_count"] = raw_band_count

    train_processed = append_first_order_diffs(train_processed)
    test_processed = append_first_order_diffs(test_processed)
    normalized_config["diffs_appended_post_normalization"] = True

    feature_dim = int(train_processed.shape[1])
    normalized_config["baseline_feature_dim"] = feature_dim

    dim_method = normalized_config.get("dim_method", "NONE").upper()
    if model_name == "PLSR":
        dim_method = "NONE"
        normalized_config["plsr_target_transform"] = str(SigmoidTargetTransform())
    normalized_config["dim_method"] = dim_method

    if dim_method == "NONE":
        return train_processed, test_processed, normalized_config, None

    normalized_config = _normalize_dimension_config(train_processed, normalized_config)
    n_components = normalized_config["n_components"]

    if dim_method == "PCA":
        reducer = PCA(n_components=n_components)
    elif dim_method == "KPCA":
        kernel = normalized_config.get("kpca_kernel", "rbf")
        reducer = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=False)
        normalized_config["kpca_kernel"] = kernel
    elif dim_method == "ISOMAP":
        n_neighbors = int(normalized_config.get("isomap_neighbors", 8))
        n_neighbors = max(2, min(n_neighbors, train_processed.shape[0] - 1))
        p_val = int(normalized_config.get("isomap_p", 2))
        reducer = Isomap(n_neighbors=n_neighbors, n_components=n_components, p=p_val)
        normalized_config["isomap_neighbors"] = int(n_neighbors)
        normalized_config["isomap_p"] = int(p_val)
    else:
        raise ValueError(f"Unsupported dimension reduction method: {dim_method}")

    train_reduced = reducer.fit_transform(train_processed)
    test_reduced = reducer.transform(test_processed)
    return train_reduced, test_reduced, normalized_config, reducer


def transform_with_preprocessing(
    features: np.ndarray,
    config: Dict[str, Any],
    reducer: Any,
) -> np.ndarray:
    """Transform new features using a normalized config and optional reducer."""
    raw_band_count = int(config.get("baseline_raw_band_count", 0))
    feature_dim = int(config.get("baseline_feature_dim", 0))

    if feature_dim <= 0 and reducer is not None and hasattr(reducer, "n_features_in_"):
        feature_dim = int(getattr(reducer, "n_features_in_"))
    if raw_band_count <= 0:
        raw_band_count = feature_dim if feature_dim > 0 else 0

    processed = _extract_raw_spectra(features, raw_band_count, feature_dim)

    if config.get("use_sg", False):
        processed = savgol_filter(
            processed,
            window_length=int(config["sg_window"]),
            polyorder=int(config["sg_poly"]),
            axis=1,
        )

    normalization = str(config.get("normalization", "NONE")).upper()
    if normalization not in {"NONE", "MINMAX", "SNV"}:
        raise ValueError(f"Unsupported normalization method: {normalization}")

    if normalization == "MINMAX":
        params = config.get("normalization_params")
        if not params:
            raise ValueError("Min-max normalization parameters missing from configuration.")
        feature_min = np.asarray(params["min"], dtype=np.float32)
        denom = np.asarray(params["denom"], dtype=np.float32)
        denom[np.abs(denom) < EPS] = 1.0
        processed = (processed - feature_min) / denom
        processed = np.clip(processed, 0.0, 1.0)
    elif normalization == "SNV":
        mean = processed.mean(axis=1, keepdims=True)
        std = processed.std(axis=1, keepdims=True)
        std[np.abs(std) < EPS] = 1.0
        processed = (processed - mean) / std

    processed = append_first_order_diffs(processed)

    dim_method = config.get("dim_method", "NONE").upper()
    if dim_method == "NONE" or reducer is None:
        return processed

    return reducer.transform(processed)


__all__ = [
    "load_full_dataset",
    "split_train_test",
    "apply_preprocessing_pipeline",
    "transform_with_preprocessing",
    "compute_metrics",
    "instantiate_model",
]
