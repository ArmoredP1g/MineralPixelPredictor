# MineralPixelPredictor

## Project overview
- Deep learning training entry: `train.py`
- Classical baseline comparison entry: `compare_baselines_fold0.py`
- Core classical pipeline: `classical_baseline_core.py` and `traditional_model_utils.py`
- Deep learning training logic: `trainer.py` and `deep_learning_experiment.py`
- Model definitions: `models/models.py`

Detailed Chinese project documentation:
- `工程项目详细说明.md`

## Environment
Recommended to run with conda environment `SI-NET`.

```powershell
conda run -n SI-NET python --version
```

Dataset and runtime defaults are defined in `configs/training_cfg.py`:
- `dataset_path`: hyperspectral data root
- `ckpt_path`: checkpoint directory
- `session_tag`: experiment tag for checkpoint naming

If needed, update `dataset_path` before running training.

## Deep learning training
Run 5-fold training (AE pretrain + predictor train) with:

```powershell
conda run -n SI-NET python train.py
```

What this does:
- Builds target-balanced 5 folds.
- Writes fold/config manifest to `ckpt/<session_tag>/cfg.json`.
- For each fold, trains AE and predictor via `trainer.py`.

Outputs:
- Checkpoints under `ckpt/<session_tag>/`.
- TensorBoard logs under `runs/` (if enabled by trainer settings).

## Classical baseline comparison experiment
Run full comparison on fold0 (test fold = 0):

```powershell
conda run -n SI-NET python compare_baselines_fold0.py --test-fold 0 --trials 120 --output-json comparison_fold0_results.json --output-html comparison_fold0_results.html
```

Supported methods:
- `SVR`
- `PLSR`
- `XGB`
- `RF`
- `LR`

Behavior:
- For each method, if model artifact exists it is loaded.
- If artifact does not exist, the script runs random search and trains a model.
- Use `--force-retrain` to ignore existing artifacts and retrain.

Artifacts and reports:
- Model artifacts: `saved_models/comparison_fold0/<method>_fold0.pkl`
- Structured report: `comparison_fold0_results.json`
- Color HTML report: `comparison_fold0_results.html`

Useful options:

```powershell
# Quick smoke run
conda run -n SI-NET python compare_baselines_fold0.py --methods LR SVR --trials 3 --force-retrain --output-json smoke.json --output-html smoke.html

# Run selected methods only
conda run -n SI-NET python compare_baselines_fold0.py --methods SVR PLSR XGB --trials 120
```

## Notes
- `--trials` means random-search attempts per method. Larger values usually improve hyperparameter quality but increase runtime.
- The comparison script includes warning suppression for scipy sparse-structure efficiency messages caused by sklearn Isomap internals.
