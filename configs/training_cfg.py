from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"
    dataset_path: str = "D:\\IR_DATA"
    ckpt_path: str = "ckpt"
    num_workers: int = 6
    session_tag: str = "final"


@dataclass(frozen=True)
class TrainingConfig:
    ae_step_per_fold: int = 40000
    pd_step_per_fold: int = 150000
    batch_size: int = 12
    learning_rate: float = 1e-3
    lr_decay: float = 0.98
    lr_decay_step: int = 1000
    lr_lower_bound: float = 1e-8


@dataclass(frozen=True)
class ProjectConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


DEFAULT_CONFIG = ProjectConfig()


device = DEFAULT_CONFIG.runtime.device
dataset_path = DEFAULT_CONFIG.runtime.dataset_path
ckpt_path = DEFAULT_CONFIG.runtime.ckpt_path
num_workers = DEFAULT_CONFIG.runtime.num_workers
session_tag = DEFAULT_CONFIG.runtime.session_tag

ae_step_per_fold = DEFAULT_CONFIG.training.ae_step_per_fold
pd_step_per_fold = DEFAULT_CONFIG.training.pd_step_per_fold
batch_size = DEFAULT_CONFIG.training.batch_size
learning_rate = DEFAULT_CONFIG.training.learning_rate
lr_decay = DEFAULT_CONFIG.training.lr_decay
lr_decay_step = DEFAULT_CONFIG.training.lr_decay_step
lr_lower_bound = DEFAULT_CONFIG.training.lr_lower_bound



# min_nrmse: 0.07905751279119358, r2: 0.8075416333615556, rmse: 4.292822582665238
plsr = {
    'SG': True,
    'SG_win_len': 6,
    'SG_poly': 4,
    'dim_reduction_method': 'ISOMAP',
    'KPCA_kernel': 'rbf',
    'ISOMAP_n_neighbors': 8,
    'ISOMAP_p': 1,
    'SVR_C': 9.773572184449225,
    'SVR_epsilon': 7.1367850298175,
    'SVR_kernel': 'poly',
    'SVR_degree': 4,
    'XGB_eta': 0.04491695659427874,
    'XGB_n_estimators': 123,
    'XGB_max_depth': 7,
    'XGB_subsample': 0.6272730445996957,
    'PLSR_max_iter': 694,
    'n_components': 18
}

# total_try: 8726, min_nrmse: 0.07378145307302475, r2: 0.8323726505041122, rmse: 4.0063323974609375, cfg: 
xgb = {
    'SG': False,
    'SG_win_len': 13,
    'SG_poly': 3,
    'dim_reduction_method': 'PCA',
    'KPCA_kernel': 'poly',
    'ISOMAP_n_neighbors': 9,
    'ISOMAP_p': 1,
    'SVR_C': 9.60346211831687,
    'SVR_epsilon': 8.709969371370246,
    'SVR_kernel': 'sigmoid',
    'SVR_degree': 1,
    'XGB_eta': 0.15569648689259247,
    'XGB_n_estimators': 75,
    'XGB_max_depth': 9,
    'XGB_subsample': 0.7769908421679166,
    'PLSR_max_iter': 626,
    'n_components': 6
}


# min_nrmse: 0.07905751279119358, r2: 0.8075416333615556, rmse: 4.292822582665238
plsr_best = {
    'SG': True,
    'SG_win_len': 6,
    'SG_poly': 4,
    'dim_reduction_method': 'ISOMAP',
    'KPCA_kernel': 'linear',
    'ISOMAP_n_neighbors': 6,
    'ISOMAP_p': 1,
    'SVR_C': 7.264871939876779,
    'SVR_epsilon': 1.6795702588958292,
    'SVR_kernel': 'rbf',
    'SVR_degree': 3,
    'XGB_eta': 0.1981205917064724,
    'XGB_n_estimators': 58,
    'XGB_max_depth': 4,
    'XGB_subsample': 0.7047146210632055,
    'PLSR_max_iter': 371,
    'PLSR_tol': 0.0006755715707312003,
    'n_components': 18
}