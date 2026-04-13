from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from configs.training_cfg import device, learning_rate, lr_decay, lr_lower_bound
from deep_learning_experiment import (
    build_deep_learning_folds,
    build_test_data,
    build_train_loaders,
    split_fold_lists,
    write_training_manifest,
)
from models.contrastive_losses import ReconstructionSupConLoss
from models.encoder_variants import build_encoder
from models.models import AE_Decoder, Predictor
from trainer import evaluate_predictor_metrics, train_AE, train_predictor


EXPERIMENT_CONFIGS = {
    "1": {
        "exp_id": 1,
        "name": "Baseline",
        "encoder_type": "StandardCAE",
        "use_supcon": False,
        "description": "标准1D-CAE，纯MSE重构",
    },
    "2": {
        "exp_id": 2,
        "name": "ASPP-CAE",
        "encoder_type": "ASPP_CAE",
        "use_supcon": False,
        "description": "ASPP多尺度感受野，纯MSE重构",
    },
    "3": {
        "exp_id": 3,
        "name": "Mixer-AE",
        "encoder_type": "MixerAE",
        "use_supcon": False,
        "description": "固定Patch的MLP-Mixer编码器，纯MSE重构",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fold0 ablation experiments (#1-#4).")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for test split")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="Subset of experiments to run from {1,2,3,4}",
    )
    parser.add_argument("--ae-steps", type=int, default=None, help="Override AE training steps for quick smoke tests")
    parser.add_argument(
        "--pd-steps",
        type=int,
        default=None,
        help="Override predictor training steps for quick smoke tests",
    )
    parser.add_argument(
        "--ae-loader-batch",
        type=int,
        default=4,
        help="AE DataLoader batch size used when SupCon is enabled",
    )
    parser.add_argument("--samplepoint", type=int, default=500, help="Sampled pixels per sample during training")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for ablation runs")
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for test samples (useful for smoke runs)",
    )
    parser.add_argument("--mixer-patch-size", type=int, default=12, choices=[12, 14], help="Fixed Mixer patch size")
    parser.add_argument("--tau", type=float, default=0.07, help="SupCon temperature")
    parser.add_argument("--lambda-supcon", type=float, default=0.1, help="SupCon weight")
    parser.add_argument("--session-base", type=str, default="ablation_fold0", help="Prefix for checkpoint sessions")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("comparison_fold0_results.json"),
        help="Optional classical baseline JSON for side-by-side comparison",
    )
    parser.add_argument("--output-json", type=Path, default=Path("ablation_fold0_results.json"))
    parser.add_argument("--output-html", type=Path, default=Path("ablation_fold0_results.html"))
    return parser.parse_args()


def _load_baseline_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    result = {}
    for method, method_payload in payload.get("results", {}).items():
        metrics = method_payload.get("evaluation", {}).get("test_sample", {})
        if not metrics:
            continue
        result[method] = {
            "rmse": float(metrics.get("rmse", float("nan"))),
            "r2": float(metrics.get("r2", float("nan"))),
            "nrmse": float(metrics.get("nrmse", float("nan"))),
        }
    return result


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _run_single_experiment(
    exp_id: int,
    name: str,
    encoder_type: str,
    use_supcon: bool,
    description: str,
    folds: List[List[str]],
    fold_idx: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    training_list, test_list = split_fold_lists(folds, fold_idx)
    if args.max_test_samples is not None:
        test_list = test_list[: args.max_test_samples]
    test_data = build_test_data(test_list)

    session_name = f"{args.session_base}_exp{exp_id}_fold{fold_idx}"
    write_training_manifest(
        folds,
        session_name=session_name,
        extra_payload={
            "ablation_experiment_id": exp_id,
            "ablation_experiment_name": name,
            "encoder_type": encoder_type,
            "use_supcon": use_supcon,
            "lambda_supcon": args.lambda_supcon,
            "tau": args.tau,
            "fold": fold_idx,
            "checkpoint_policy": "best_last",
            "mixer_patch_size": args.mixer_patch_size,
        },
    )

    encoder_kwargs = {}
    if encoder_type == "MixerAE":
        encoder_kwargs["patch_size"] = args.mixer_patch_size

    encoder = build_encoder(encoder_type, **encoder_kwargs).to(device)
    decoder = AE_Decoder().to(device)
    predictor = Predictor().to(device)

    trainloader_ae, trainloader_predictor = build_train_loaders(
        training_list,
        ae_steps=args.ae_steps,
        predictor_steps=args.pd_steps,
        loader_batch_size=(args.ae_loader_batch if use_supcon else 1),
        samplepoint=args.samplepoint,
        return_sample_index=use_supcon,
        num_workers_override=args.num_workers,
    )

    criterion = None
    if use_supcon:
        criterion = ReconstructionSupConLoss(lambda_supcon=args.lambda_supcon, temperature=args.tau)

    t0 = time.time()
    ae_model = nn.Sequential(encoder, decoder).to(device)
    encoder, ae_info = train_AE(
        trainloader_ae,
        ae_model,
        fold=fold_idx,
        lr=learning_rate,
        tag=session_name,
        step=0,
        criterion=criterion,
        session_name=session_name,
        checkpoint_mode="best_last",
        max_steps=args.ae_steps,
        return_info=True,
    )

    predictor_info = train_predictor(
        trainloader_predictor,
        encoder,
        predictor,
        fold=fold_idx,
        lr=learning_rate,
        tag=session_name,
        pretrain_step=-1,
        lr_decay=float(lr_decay),
        lr_decay_step=1000,
        lr_lower_bound=lr_lower_bound,
        step=1,
        test_data=test_data,
        freeze=True,
        session_name=session_name,
        checkpoint_mode="best_last",
        max_steps=args.pd_steps,
    )

    best_checkpoint = predictor_info.get("best_checkpoint")
    eval_model = nn.Sequential(encoder, predictor).to(device)
    if best_checkpoint is not None and Path(best_checkpoint).exists():
        eval_model.load_state_dict(torch.load(best_checkpoint, map_location=device))

    evaluation = evaluate_predictor_metrics(eval_model, test_data)
    elapsed = time.time() - t0

    return {
        "exp_id": exp_id,
        "name": name,
        "encoder_type": encoder_type,
        "use_supcon": use_supcon,
        "description": description,
        "fold": fold_idx,
        "elapsed_seconds": elapsed,
        "ae": ae_info,
        "predictor": predictor_info,
        "evaluation": evaluation,
    }


def _build_html_report(
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, float]],
    output_html: Path,
):
    rows = []
    for item in sorted(results, key=lambda x: x["exp_id"]):
        metrics = item["evaluation"]["test_sample"]
        rows.append(
            "<tr>"
            f"<td>#{item['exp_id']}</td>"
            f"<td>{item['name']}</td>"
            f"<td>{item['encoder_type']}</td>"
            f"<td>{'Yes' if item['use_supcon'] else 'No'}</td>"
            f"<td>{metrics['rmse']:.6f}</td>"
            f"<td>{metrics['r2']:.6f}</td>"
            f"<td>{item['evaluation']['pixel_mean']['rmse']:.6f}</td>"
            f"<td>{item['evaluation']['pixel_median']['rmse']:.6f}</td>"
            f"<td>{item['elapsed_seconds'] / 60.0:.2f} min</td>"
            f"<td>{item['description']}</td>"
            "</tr>"
        )

    comparison_html = ""
    if baseline_metrics and len(results) > 0:
        best_exp = min(results, key=lambda x: x["evaluation"]["test_sample"]["rmse"])
        best_metrics = best_exp["evaluation"]["test_sample"]

        baseline_rows = []
        for method, metrics in sorted(baseline_metrics.items()):
            baseline_rows.append(
                "<tr>"
                f"<td>{method}</td>"
                f"<td>{metrics['rmse']:.6f}</td>"
                f"<td>{metrics['r2']:.6f}</td>"
                f"<td>{metrics['nrmse']:.6f}</td>"
                "</tr>"
            )

        comparison_html = f"""
        <div class='card'>
          <h2>Best Deep Variant vs Classical Baselines</h2>
          <p><strong>Best Deep Variant:</strong> #{best_exp['exp_id']} {best_exp['name']} (RMSE={best_metrics['rmse']:.6f}, R2={best_metrics['r2']:.6f})</p>
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th>RMSE</th>
                <th>R2</th>
                <th>NRMSE</th>
              </tr>
            </thead>
            <tbody>
              {''.join(baseline_rows)}
            </tbody>
          </table>
        </div>
        """

    html = f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Ablation Fold0 Report</title>
  <style>
    :root {{
      --bg: #f8fbf4;
      --card: #ffffff;
      --line: #dce6d0;
      --text: #1c2a18;
      --accent: #2f7d32;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #e5f4d7 0%, var(--bg) 58%);
    }}
    .wrap {{ max-width: 1280px; margin: 20px auto 36px; padding: 0 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    h1 {{ margin: 0 0 10px; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #d8e2cc; padding: 8px 10px; text-align: right; font-size: 13px; }}
    th {{ background: #edf4e5; color: #2b4a27; }}
    th:first-child, td:first-child {{ text-align: left; }}
    td:nth-child(2), td:nth-child(3), td:nth-child(10) {{ text-align: left; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='card'>
      <h1>Fold {metadata['fold']} Ablation Report</h1>
      <p>Session base: {metadata['session_base']}</p>
      <p>Mixer patch size: {metadata['mixer_patch_size']} | tau: {metadata['tau']} | lambda_supcon: {metadata['lambda_supcon']}</p>
    </div>
    <div class='card'>
      <h2>Ablation Matrix</h2>
      <table>
        <thead>
          <tr>
            <th>Exp#</th>
            <th>Name</th>
            <th>Encoder</th>
            <th>SupCon</th>
            <th>Test RMSE</th>
            <th>Test R2</th>
            <th>PixelMean RMSE</th>
            <th>PixelMedian RMSE</th>
            <th>Elapsed</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    {comparison_html}
  </div>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_set = set(args.experiments)

    folds = build_deep_learning_folds()
    all_results: List[Dict[str, Any]] = []

    def run_if_needed(exp_key: str) -> Dict[str, Any]:
        for existing in all_results:
            if str(existing["exp_id"]) == exp_key:
                return existing

        cfg = EXPERIMENT_CONFIGS[exp_key]
        result = _run_single_experiment(
            exp_id=cfg["exp_id"],
            name=cfg["name"],
            encoder_type=cfg["encoder_type"],
            use_supcon=cfg["use_supcon"],
            description=cfg["description"],
            folds=folds,
            fold_idx=args.fold,
            args=args,
        )
        all_results.append(result)
        print(
            f"Exp#{cfg['exp_id']} done: RMSE={result['evaluation']['test_sample']['rmse']:.6f}, "
            f"R2={result['evaluation']['test_sample']['r2']:.6f}"
        )
        return result

    if "1" in run_set:
        run_if_needed("1")
    if "2" in run_set or "4" in run_set:
        aspp_result = run_if_needed("2")
    else:
        aspp_result = None
    if "3" in run_set or "4" in run_set:
        mixer_result = run_if_needed("3")
    else:
        mixer_result = None

    if "4" in run_set:
        if aspp_result is None or mixer_result is None:
            raise RuntimeError("Experiment #4 requires #2 and #3 to determine the best encoder.")

        aspp_rmse = aspp_result["evaluation"]["test_sample"]["rmse"]
        mixer_rmse = mixer_result["evaluation"]["test_sample"]["rmse"]
        chosen_encoder = "ASPP_CAE" if aspp_rmse <= mixer_rmse else "MixerAE"
        chosen_name = "ASPP-CAE" if chosen_encoder == "ASPP_CAE" else "Mixer-AE"

        supcon_result = _run_single_experiment(
            exp_id=4,
            name="BestEncoder + GradeWeightedSupCon",
            encoder_type=chosen_encoder,
            use_supcon=True,
            description=f"在#2/#3中选择更优编码器({chosen_name})并引入Grade-Weighted SupCon",
            folds=folds,
            fold_idx=args.fold,
            args=args,
        )
        supcon_result["chosen_best_encoder"] = chosen_encoder
        all_results.append(supcon_result)
        print(
            f"Exp#4 done (encoder={chosen_encoder}): RMSE={supcon_result['evaluation']['test_sample']['rmse']:.6f}, "
            f"R2={supcon_result['evaluation']['test_sample']['r2']:.6f}"
        )

    metadata = {
        "fold": args.fold,
        "session_base": args.session_base,
        "tau": args.tau,
        "lambda_supcon": args.lambda_supcon,
        "mixer_patch_size": args.mixer_patch_size,
        "ae_steps": args.ae_steps,
        "pd_steps": args.pd_steps,
        "samplepoint": args.samplepoint,
        "num_workers": args.num_workers,
        "max_test_samples": args.max_test_samples,
        "experiments": sorted([str(r["exp_id"]) for r in all_results], key=int),
    }

    baseline_metrics = _load_baseline_metrics(args.baseline_json)

    payload = {
        "metadata": metadata,
        "results": sorted(all_results, key=lambda x: x["exp_id"]),
        "baseline_metrics": baseline_metrics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    _build_html_report(payload["results"], metadata, baseline_metrics, args.output_html)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved HTML: {args.output_html}")


if __name__ == "__main__":
    main()
