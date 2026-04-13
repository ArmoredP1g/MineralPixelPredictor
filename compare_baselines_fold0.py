from __future__ import annotations

import argparse
import json
import pickle
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Suppress scipy internal SparseEfficiencyWarning that arises when sklearn's
# Isomap graph-completion routine changes the sparsity structure of a CSR matrix.
# This is entirely internal to sklearn/scipy and not actionable from user code.
warnings.filterwarnings(
    "ignore",
    message="Changing the sparsity structure",
    category=Warning,
    module="scipy",
)

from classical_baseline_core import SampleAggregator, build_dataset, compute_metrics, instantiate_model, run_random_search
from experiment_utils import ALL_SAMPLES
from traditional_model_utils import apply_preprocessing_pipeline, split_train_test, transform_with_preprocessing

ALL_METHODS = ["SVR", "PLSR", "XGB", "RF", "LR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classical baselines on fold0 with sample-level training and pixel-level inference."
    )
    parser.add_argument("--test-fold", type=int, default=0, help="Fold index used as test set (default: 0)")
    parser.add_argument("--trials", type=int, default=120, help="Random search trials per method when training")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        help=f"Methods to run. Supported: {', '.join(ALL_METHODS)}",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("saved_models") / "comparison_fold0",
        help="Directory for saved model artifacts",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("comparison_fold0_results.json"),
        help="Path to save structured experiment results",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("comparison_fold0_results.html"),
        help="Path to save color-enhanced HTML report",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain all selected methods even if artifacts already exist",
    )
    return parser.parse_args()


def _validate_methods(methods: Sequence[str]) -> List[str]:
    """
    Validates and deduplicates a list of method names.

    Args:
        methods (Sequence[str]): A sequence of method names to validate.

    Returns:
        List[str]: A list of unique, validated method names in uppercase.

    Raises:
        ValueError: If any method in the input sequence is not supported.

    Notes:
        - Supported methods are defined in the global `ALL_METHODS` list.
        - The function ensures that the returned list contains no duplicates.
    """
    unique: List[str] = []  # Initialize an empty list to store unique method names.
    for method in methods:
        name = method.upper()  # Convert the method name to uppercase for case-insensitive comparison.
        if name not in ALL_METHODS:  # Check if the method is in the list of supported methods.
            raise ValueError(f"Unsupported method: {method}")  # Raise an error for unsupported methods.
        if name not in unique:  # Add the method to the unique list if it's not already included.
            unique.append(name)
    return unique  # Return the list of unique, validated method names.


def _artifact_path(artifacts_dir: Path, method: str, fold: int) -> Path:
    return artifacts_dir / f"{method.lower()}_fold{fold}.pkl"


def _method_seed(base_seed: int, method: str) -> int:
    return base_seed + sum(ord(ch) for ch in method)


def _train_method(
    method: str,
    dataset: Any,
    test_fold: int,
    trials: int,
    base_seed: int,
) -> Dict[str, Any]:
    method_rng = random.Random(_method_seed(base_seed, method))
    search_result = run_random_search(method, dataset, trials, method_rng)

    best_config = search_result["best"]["config"]
    X_train, y_train, X_test, y_test = split_train_test(dataset, test_fold=test_fold)
    global_range = float(np.max(dataset.targets) - np.min(dataset.targets))

    train_processed, test_processed, normalized_config, reducer = apply_preprocessing_pipeline(
        X_train, X_test, best_config, method
    )
    model = instantiate_model(method, normalized_config)
    model.fit(train_processed, y_train)

    train_pred = model.predict(train_processed).reshape(-1)
    test_pred = model.predict(test_processed).reshape(-1)

    train_metrics = compute_metrics(y_train, train_pred, global_range)
    test_metrics = compute_metrics(y_test, test_pred, global_range)

    return {
        "method": method,
        "model": model,
        "reducer": reducer,
        "config": normalized_config,
        "best_config": best_config,
        "search": {
            "best_metrics": search_result["best"]["metrics"],
            "trials": trials,
        },
        "performance": {
            "train": train_metrics,
            "test_sample": test_metrics,
        },
    }


def _load_or_train_method(
    method: str,
    dataset: Any,
    test_fold: int,
    trials: int,
    base_seed: int,
    artifacts_dir: Path,
    force_retrain: bool,
) -> Tuple[Dict[str, Any], str, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact = _artifact_path(artifacts_dir, method, test_fold)

    if artifact.exists() and not force_retrain:
        with artifact.open("rb") as f:
            payload = pickle.load(f)
        return payload, "loaded", artifact

    payload = _train_method(method, dataset, test_fold, trials, base_seed)
    with artifact.open("wb") as f:
        pickle.dump(payload, f)
    return payload, "trained", artifact


def _get_test_indices(dataset: Any, test_fold: int) -> List[int]:
    return [int(i) for i in dataset.folds[test_fold]]


def _predict_for_sample_and_pixels(
    model: Any,
    reducer: Any,
    config: Dict[str, Any],
    sample_raw: np.ndarray,
    pixel_raw: np.ndarray,
) -> Tuple[float, float, float]:
    sample_proc = transform_with_preprocessing(sample_raw.reshape(1, -1), config, reducer)
    sample_pred = float(np.asarray(model.predict(sample_proc)).reshape(-1)[0])

    pixel_proc = transform_with_preprocessing(pixel_raw, config, reducer)
    pixel_preds = np.asarray(model.predict(pixel_proc)).reshape(-1)
    pixel_mean = float(np.mean(pixel_preds))
    pixel_median = float(np.median(pixel_preds))
    return sample_pred, pixel_mean, pixel_median


def _evaluate_pixelized_results(
    dataset: Any,
    test_fold: int,
    model_payload: Dict[str, Any],
) -> Dict[str, Any]:
    test_indices = _get_test_indices(dataset, test_fold)
    sample_ids = dataset.sample_ids
    features = dataset.features
    targets = dataset.targets

    model = model_payload["model"]
    reducer = model_payload.get("reducer")
    config = model_payload["config"]

    raw_band_count = int(config.get("baseline_raw_band_count", (features.shape[1] + 1) // 2))
    global_range = float(np.max(targets) - np.min(targets))

    aggregator = SampleAggregator()

    y_true: List[float] = []
    y_sample: List[float] = []
    y_pixel_mean: List[float] = []
    y_pixel_median: List[float] = []
    details: List[Dict[str, Any]] = []

    for idx in test_indices:
        sample_id = sample_ids[idx]
        y_val = float(targets[idx])
        sample_raw = np.asarray(features[idx][:raw_band_count], dtype=np.float32)

        pixel_tensor, target_from_pixels = aggregator.get_sample_pixels(sample_id)
        pixel_raw = pixel_tensor.cpu().numpy().astype(np.float32)

        sample_pred, pixel_mean, pixel_median = _predict_for_sample_and_pixels(
            model=model,
            reducer=reducer,
            config=config,
            sample_raw=sample_raw,
            pixel_raw=pixel_raw,
        )

        # Allow small floating-point representation drift between two data paths.
        if not np.isclose(target_from_pixels, y_val, rtol=1e-5, atol=1e-4):
            raise RuntimeError(f"Target mismatch for sample {sample_id}: {target_from_pixels} vs {y_val}")

        y_true.append(y_val)
        y_sample.append(sample_pred)
        y_pixel_mean.append(pixel_mean)
        y_pixel_median.append(pixel_median)

        details.append(
            {
                "sample_id": sample_id,
                "y_true": y_val,
                "sample_pred": sample_pred,
                "pixel_mean_pred": pixel_mean,
                "pixel_median_pred": pixel_median,
                "sample_abs_err": abs(sample_pred - y_val),
                "pixel_mean_abs_err": abs(pixel_mean - y_val),
                "pixel_median_abs_err": abs(pixel_median - y_val),
            }
        )

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    sample_arr = np.asarray(y_sample, dtype=np.float32)
    mean_arr = np.asarray(y_pixel_mean, dtype=np.float32)
    median_arr = np.asarray(y_pixel_median, dtype=np.float32)

    return {
        "test_sample": compute_metrics(y_true_arr, sample_arr, global_range),
        "pixel_mean": compute_metrics(y_true_arr, mean_arr, global_range),
        "pixel_median": compute_metrics(y_true_arr, median_arr, global_range),
        "details": details,
    }


def _normalize_for_color(values: List[float], reverse: bool) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if abs(v_max - v_min) < 1e-12:
        return [0.5 for _ in values]

    norm = (arr - v_min) / (v_max - v_min)
    if reverse:
        norm = 1.0 - norm
    return [float(v) for v in norm]


def _score_to_rgb(score: float) -> str:
    score = max(0.0, min(1.0, score))
    # White -> green gradient, higher score means better.
    r = int(245 - score * 60)
    g = int(245 - score * -8)
    b = int(245 - score * 92)
    return f"rgb({r}, {g}, {b})"


def _metric_cell(value: float, score: float, best: bool) -> str:
    style = f"background:{_score_to_rgb(score)};"
    content = f"{value:.6f}"
    if best:
        content = f"<strong>{content}</strong>"
    return f"<td style='{style}'>{content}</td>"


def _build_summary_table_rows(method_results: List[Dict[str, Any]]) -> str:
    columns = [
        ("sample_nrmse", "test_sample", "nrmse", True),
        ("sample_rmse", "test_sample", "rmse", True),
        ("sample_r2", "test_sample", "r2", False),
        ("pixel_mean_nrmse", "pixel_mean", "nrmse", True),
        ("pixel_mean_rmse", "pixel_mean", "rmse", True),
        ("pixel_mean_r2", "pixel_mean", "r2", False),
        ("pixel_median_nrmse", "pixel_median", "nrmse", True),
        ("pixel_median_rmse", "pixel_median", "rmse", True),
        ("pixel_median_r2", "pixel_median", "r2", False),
    ]

    matrix: Dict[str, List[float]] = {}
    for col_name, bucket, key, _ in columns:
        matrix[col_name] = [float(item["evaluation"][bucket][key]) for item in method_results]

    color_scores: Dict[str, List[float]] = {}
    best_index: Dict[str, int] = {}
    for col_name, _, _, lower_is_better in columns:
        scores = _normalize_for_color(matrix[col_name], reverse=not lower_is_better)
        color_scores[col_name] = scores
        values = matrix[col_name]
        if lower_is_better:
            best_index[col_name] = int(np.argmin(values))
        else:
            best_index[col_name] = int(np.argmax(values))

    rows: List[str] = []
    for row_idx, item in enumerate(method_results):
        method = item["method"]
        status = item["status"]
        search_nrmse = float(item["search_best_nrmse"])
        search_nrmse_style = "color:#0b6f31;font-weight:700;" if status == "loaded" else "color:#8a5a00;font-weight:700;"

        row = [
            f"<td><strong>{method}</strong></td>",
            f"<td style='{search_nrmse_style}'>{status}</td>",
            f"<td>{search_nrmse:.6f}</td>",
        ]

        for col_name, bucket, key, _ in columns:
            val = float(item["evaluation"][bucket][key])
            score = color_scores[col_name][row_idx]
            is_best = best_index[col_name] == row_idx
            row.append(_metric_cell(val, score, is_best))

        rows.append("<tr>" + "".join(row) + "</tr>")

    return "\n".join(rows)


def _build_detail_sections(method_results: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for item in method_results:
        method = item["method"]
        details = item["evaluation"]["details"]

        rows = []
        for d in details:
            rows.append(
                "<tr>"
                f"<td>{d['sample_id']}</td>"
                f"<td>{d['y_true']:.6f}</td>"
                f"<td>{d['sample_pred']:.6f}</td>"
                f"<td>{d['pixel_mean_pred']:.6f}</td>"
                f"<td>{d['pixel_median_pred']:.6f}</td>"
                f"<td>{d['sample_abs_err']:.6f}</td>"
                f"<td>{d['pixel_mean_abs_err']:.6f}</td>"
                f"<td>{d['pixel_median_abs_err']:.6f}</td>"
                "</tr>"
            )

        section = f"""
        <h3>{method} - Fold0 Test Sample Details</h3>
        <table class='details'>
            <thead>
                <tr>
                    <th>Sample ID</th>
                    <th>True</th>
                    <th>Sample Pred</th>
                    <th>Pixel Mean Pred</th>
                    <th>Pixel Median Pred</th>
                    <th>|Err| Sample</th>
                    <th>|Err| Pixel Mean</th>
                    <th>|Err| Pixel Median</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
        sections.append(section)

    return "\n".join(sections)


def _build_html_report(method_results: List[Dict[str, Any]], output_html: Path, meta: Dict[str, Any]) -> None:
    summary_rows = _build_summary_table_rows(method_results)
    detail_sections = _build_detail_sections(method_results)

    html = f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Fold0 Baseline Comparison</title>
  <style>
    :root {{
      --bg: #f6f8f2;
      --card: #ffffff;
      --line: #dce3d1;
      --text: #172016;
      --sub: #4e5d46;
      --accent: #2f7d32;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #eaf5dd 0%, var(--bg) 55%);
    }}
    .wrap {{ max-width: 1280px; margin: 24px auto 60px; padding: 0 18px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 8px 24px rgba(23,32,22,0.06); margin-bottom: 18px; }}
    h1 {{ margin: 0 0 12px; color: var(--accent); }}
    h2 {{ margin: 0 0 10px; }}
    h3 {{ margin: 22px 0 8px; color: #365c2d; }}
    p, li {{ color: var(--sub); }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #d7e1cb; padding: 8px 10px; text-align: right; font-size: 13px; }}
    th {{ background: #edf4e5; color: #264424; position: sticky; top: 0; z-index: 1; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .details th, .details td {{ font-size: 12px; }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
    .chip {{ background: #eef6e8; border: 1px solid #cfe1bf; border-radius: 999px; padding: 4px 10px; font-size: 12px; color: #36542f; }}
    @media (max-width: 900px) {{
      th, td {{ padding: 6px 7px; font-size: 11px; }}
    }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='card'>
      <h1>Classical Baseline Comparison Report (Fold0)</h1>
      <ul>
        <li>Test fold: {meta['test_fold']}</li>
        <li>Trials per method (when training): {meta['trials']}</li>
        <li>Seed: {meta['seed']}</li>
        <li>Methods: {', '.join(meta['methods'])}</li>
      </ul>
      <div class='legend'>
        <span class='chip'>Green background = better value in column</span>
        <span class='chip'>Bold = best among methods in column</span>
        <span class='chip'>Sample metrics from sample-level prediction</span>
        <span class='chip'>Pixel Mean/Median from pixel-level inference aggregation</span>
      </div>
    </div>

    <div class='card'>
      <h2>Method Overview</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Status</th>
            <th>Search Best NRMSE</th>
            <th>Sample NRMSE</th>
            <th>Sample RMSE</th>
            <th>Sample R2</th>
            <th>Pixel Mean NRMSE</th>
            <th>Pixel Mean RMSE</th>
            <th>Pixel Mean R2</th>
            <th>Pixel Median NRMSE</th>
            <th>Pixel Median RMSE</th>
            <th>Pixel Median R2</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
    </div>

    <div class='card'>
      <h2>Per-sample Detailed Results</h2>
      {detail_sections}
    </div>
  </div>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    methods = _validate_methods(args.methods)

    start_all = time.time()
    print("Loading dataset and folds...", flush=True)
    dataset = build_dataset(ALL_SAMPLES)
    print(f"Dataset size: {len(dataset.sample_ids)} | Feature dim: {dataset.features.shape[1]}", flush=True)

    result_bundle: Dict[str, Any] = {
        "metadata": {
            "test_fold": args.test_fold,
            "trials": args.trials,
            "seed": args.seed,
            "methods": methods,
            "artifacts_dir": str(args.artifacts_dir),
        },
        "results": {},
    }

    method_results: List[Dict[str, Any]] = []

    for method in methods:
        print(f"\n=== Running method: {method} ===", flush=True)
        t0 = time.time()

        payload, status, artifact_path = _load_or_train_method(
            method=method,
            dataset=dataset,
            test_fold=args.test_fold,
            trials=args.trials,
            base_seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            force_retrain=args.force_retrain,
        )

        eval_result = _evaluate_pixelized_results(dataset, args.test_fold, payload)
        elapsed = time.time() - t0

        result_bundle["results"][method] = {
            "status": status,
            "artifact": str(artifact_path),
            "elapsed_seconds": elapsed,
            "search": _to_jsonable(payload.get("search", {})),
            "config": _to_jsonable(payload.get("config", {})),
            "best_config": _to_jsonable(payload.get("best_config", {})),
            "performance": _to_jsonable(payload.get("performance", {})),
            "evaluation": _to_jsonable(eval_result),
        }

        method_results.append(
            {
                "method": method,
                "status": status,
                "search_best_nrmse": float(payload.get("search", {}).get("best_metrics", {}).get("nrmse", np.nan)),
                "evaluation": eval_result,
            }
        )

        print(
            f"Status: {status} | Sample NRMSE: {eval_result['test_sample']['nrmse']:.4f} | "
            f"PixelMean NRMSE: {eval_result['pixel_mean']['nrmse']:.4f} | "
            f"PixelMedian NRMSE: {eval_result['pixel_median']['nrmse']:.4f} | "
            f"Elapsed: {elapsed:.1f}s",
            flush=True,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result_bundle, f, indent=2)

    _build_html_report(method_results, args.output_html, result_bundle["metadata"])

    total_elapsed = time.time() - start_all
    print("\n=== Comparison completed ===", flush=True)
    print(f"JSON report: {args.output_json}", flush=True)
    print(f"HTML report: {args.output_html}", flush=True)
    print(f"Total elapsed: {total_elapsed / 60.0:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
