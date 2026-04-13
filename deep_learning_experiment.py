from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

from torch.utils.data import DataLoader

from configs.training_cfg import (
    ae_step_per_fold,
    batch_size,
    ckpt_path,
    dataset_path,
    device,
    learning_rate,
    lr_decay,
    lr_decay_step,
    lr_lower_bound,
    num_workers,
    pd_step_per_fold,
    session_tag,
)
from dataloaders.dataloaders import dataset_iron_balanced_mixed
from experiment_utils import ALL_SAMPLES, build_ranked_sample_index, create_target_balanced_folds, prepare_test_data


def get_training_config_payload() -> Dict[str, float | int | str]:
    return {
        "ae_step_per_fold": ae_step_per_fold,
        "pd_step_per_fold": pd_step_per_fold,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_decay": lr_decay,
        "lr_decay_step": lr_decay_step,
        "lr_lower_bound": lr_lower_bound,
        "session_tag": session_tag,
    }


def get_session_dir(session_name: str | None = None) -> str:
    resolved = session_tag if session_name is None else session_name
    return ckpt_path + "/" + resolved


def ensure_session_dir(session_name: str | None = None) -> str:
    session_dir = get_session_dir(session_name=session_name)
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
    return session_dir


def write_training_manifest(
    folds: Sequence[Sequence[str]],
    extra_payload: Dict[str, Any] | None = None,
    session_name: str | None = None,
) -> str:
    session_dir = ensure_session_dir(session_name=session_name)
    payload = get_training_config_payload()
    payload["folds"] = [list(fold) for fold in folds]
    if extra_payload:
        payload.update(extra_payload)
    manifest_path = session_dir + "/cfg.json"
    with open(manifest_path, "w") as file:
        json.dump(payload, file)
    return manifest_path


def build_deep_learning_folds(sample_ids: Sequence[str] = ALL_SAMPLES) -> List[List[str]]:
    sample_index = build_ranked_sample_index(sample_ids, dataset_path, device)
    return create_target_balanced_folds(sample_index, fold_count=5)


def split_fold_lists(folds: Sequence[Sequence[str]], fold_index: int) -> Tuple[List[str], List[str]]:
    training_list: List[str] = []
    test_list: List[str] = []
    for current_fold, fold_samples in enumerate(folds):
        if current_fold == fold_index:
            test_list += list(fold_samples)
        else:
            training_list += list(fold_samples)
    return training_list, test_list


def get_dataset_paths() -> Tuple[str, str]:
    csv_path = dataset_path + "/spectral_data_IR_winsize3.csv"
    hdf5_path = dataset_path + "/spectral_data_IR_winsize3.hdf5"
    return csv_path, hdf5_path


def build_train_datasets(
    training_list: Sequence[str],
    ae_steps: int | None = None,
    predictor_steps: int | None = None,
    samplepoint: int = batch_size,
    return_sample_index: bool = False,
):
    csv_path, hdf5_path = get_dataset_paths()
    ae_sample_count = ae_step_per_fold if ae_steps is None else ae_steps
    predictor_sample_count = pd_step_per_fold if predictor_steps is None else predictor_steps
    trainset_ae = dataset_iron_balanced_mixed(
        csv_path,
        hdf5_path,
        ae_sample_count,
        list(training_list),
        samplepoint=samplepoint,
        balance=True,
        return_sample_index=return_sample_index,
    )
    trainset_predictor = dataset_iron_balanced_mixed(
        csv_path,
        hdf5_path,
        predictor_sample_count,
        list(training_list),
        samplepoint=samplepoint,
        balance=True,
        return_sample_index=False,
    )
    return trainset_ae, trainset_predictor


def build_train_loaders(
    training_list: Sequence[str],
    ae_steps: int | None = None,
    predictor_steps: int | None = None,
    loader_batch_size: int = 1,
    samplepoint: int = batch_size,
    return_sample_index: bool = False,
    num_workers_override: int | None = None,
):
    trainset_ae, trainset_predictor = build_train_datasets(
        training_list,
        ae_steps=ae_steps,
        predictor_steps=predictor_steps,
        samplepoint=samplepoint,
        return_sample_index=return_sample_index,
    )
    worker_count = num_workers if num_workers_override is None else num_workers_override
    trainloader_ae = DataLoader(
        trainset_ae,
        shuffle=True,
        batch_size=loader_batch_size,
        num_workers=worker_count,
        drop_last=True,
    )
    trainloader_predictor = DataLoader(
        trainset_predictor,
        shuffle=True,
        batch_size=1,
        num_workers=worker_count,
        drop_last=True,
    )
    return trainloader_ae, trainloader_predictor


def build_test_data(test_list: Sequence[str]):
    return prepare_test_data(test_list, dataset_path, device)