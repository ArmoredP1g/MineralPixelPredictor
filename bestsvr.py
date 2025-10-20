from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from traditional_model_utils import (
    apply_preprocessing_pipeline,
    compute_metrics,
    instantiate_model,
    load_full_dataset,
    split_train_test,
    transform_with_preprocessing,
)

JSON_PATH = Path(__file__).resolve().parent / "baseline_random_search_results.json"
with JSON_PATH.open("r", encoding="utf-8") as f:
    RANDOM_SEARCH_RESULTS: Dict[str, Any] = json.load(f)

SVR_RESULTS = RANDOM_SEARCH_RESULTS["SVR"]
BEST_CONFIG = SVR_RESULTS["best_config"]
BEST_METRICS = SVR_RESULTS["best_metrics"]

MODEL_PATH = Path("saved_models") / "best_svr_model.pkl"


def _spectral_to_feature_matrix(spectral_data: np.ndarray) -> np.ndarray:
    array = np.asarray(spectral_data, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] < 2:
        raise ValueError("Spectral input must contain at least two bands")
    return array


def _prepare_data(test_fold: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    dataset = load_full_dataset()
    X_train, y_train, X_test, y_test = split_train_test(dataset, test_fold)
    global_range = float(np.max(dataset.targets) - np.min(dataset.targets))
    return X_train, y_train, X_test, y_test, global_range


def train_best_svr_model(test_fold: int = 0):
    print("开始训练最优SVR模型...")
    print(f"使用配置: {BEST_CONFIG}")
    print(f"基准搜索性能: {BEST_METRICS}")

    X_train, y_train, X_test, y_test, global_range = _prepare_data(test_fold)

    train_processed, test_processed, normalized_config, reducer = apply_preprocessing_pipeline(
        X_train, X_test, BEST_CONFIG, "SVR"
    )

    model = instantiate_model("SVR", normalized_config)
    print("开始训练SVR模型...")
    model.fit(train_processed, y_train)

    train_pred = model.predict(train_processed)
    test_pred = model.predict(test_processed)

    train_metrics = compute_metrics(y_train, train_pred, global_range)
    test_metrics = compute_metrics(y_test, test_pred, global_range)

    print("\n=== 模型训练完成 ===")
    print("训练集性能:")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  NRMSE: {train_metrics['nrmse']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    print(f"  相对误差: {train_metrics['re']:.4f}")

    print("\n测试集性能:")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  NRMSE: {test_metrics['nrmse']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  相对误差: {test_metrics['re']:.4f}")

    model_data = {
        "svr_model": model,
        "reducer": reducer,
        "config": normalized_config,
        "search_config": BEST_CONFIG,
        "random_search_metrics": BEST_METRICS,
        "performance": {
            "train": train_metrics,
            "test": test_metrics,
        },
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model_data, f)

    print(f"\n模型已保存到: {MODEL_PATH}")
    return model, model_data


def load_best_svr_model(model_path: Path | str = MODEL_PATH):
    path = Path(model_path)
    with path.open("rb") as f:
        model_data = pickle.load(f)
    return (
        model_data["svr_model"],
        model_data.get("reducer"),
        model_data["config"],
    )


def predict_with_best_model(spectral_data: np.ndarray, model_path: Path | str = MODEL_PATH):
    model, reducer, config = load_best_svr_model(model_path)
    features = _spectral_to_feature_matrix(spectral_data)
    processed = transform_with_preprocessing(features, config, reducer)
    predictions = model.predict(processed)
    return predictions


if __name__ == "__main__":
    svr_model, model_data = train_best_svr_model()

    print("\n=== 演示模型加载和使用 ===")
    loaded_model, loaded_reducer, loaded_config = load_best_svr_model()
    print("模型加载成功！")
    print(f"模型配置: {loaded_config}")
    print(f"模型性能: {model_data['performance']}")

    # 测试 predict_with_best_model 函数（使用缓存数据集中真实的原始光谱）
    dataset = load_full_dataset()
    raw_band_count = int(loaded_config.get("baseline_raw_band_count", (dataset.features.shape[1] + 1) // 2))
    sample_raw = dataset.features[0][:raw_band_count]
    test_spectral_data = sample_raw.reshape(1, -1)
    predictions = predict_with_best_model(test_spectral_data)
    print("\n=== 预测结果 ===")
    print(f"输入光谱数据: {test_spectral_data}")
    print(f"预测值: {predictions}")