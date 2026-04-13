"""Unified baseline method framework with abstract base class and concrete implementations."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

from models.sigmoid_plsr import SigmoidBoundedPLSRegression


class BaselineMethod(ABC):
    """Abstract base class for all baseline methods.
    
    Each baseline method (SVR, PLSR, XGB, RF, LR) implements:
    - get_random_config(): Generate random hyperparameter configuration
    - instantiate_model(): Create and return model instance from config
    """

    @abstractmethod
    def get_random_config(self, rng: random.Random) -> Dict[str, Any]:
        """Generate a random configuration for hyperparameter search.
        
        Args:
            rng: Random number generator for reproducibility
            
        Returns:
            Dictionary containing preprocessing and model-specific hyperparameters
        """
        pass

    @abstractmethod
    def instantiate_model(self, config: Dict[str, Any]) -> Any:
        """Create a model instance from configuration.
        
        Args:
            config: Configuration dictionary with model hyperparameters
            
        Returns:
            Instantiated sklearn/custom model object
        """
        pass

    @abstractmethod
    def fit(self, X_train: Any, y_train: Any) -> None:
        """Fit the model to training data."""
        pass


    def get_common_config(self, rng: random.Random) -> Dict[str, Any]:
        """Generate common preprocessing configuration shared by most methods.
        
        Includes: Savitzky-Golay filtering, normalization, dimensionality reduction.
        
        Args:
            rng: Random number generator
            
        Returns:
            Dictionary with common preprocessing parameters
        """
        config: Dict[str, Any] = {}
        config["use_sg"] = rng.choice([True, False])
        config["sg_window"] = rng.choice([5, 7, 9, 11, 13, 15, 17, 19])
        config["sg_poly"] = rng.choice([2, 3, 4, 5])
        config["normalization"] = rng.choice(["NONE", "MINMAX", "SNV"])
        
        # Dimensionality reduction (subclasses may override dim_options)
        dim_options = ["NONE", "PCA", "KPCA", "ISOMAP"]
        config["dim_method"] = rng.choice(dim_options)
        if config["dim_method"] != "NONE":
            config["n_components"] = rng.choice([6, 10, 14, 18, 24, 32, 40])
            config["kpca_kernel"] = rng.choice(["linear", "rbf", "poly", "sigmoid"])
            config["isomap_neighbors"] = rng.choice([10, 15, 20, 25, 30])
            config["isomap_p"] = rng.choice([1, 2, 3])
        
        return config


class SVRMethod(BaselineMethod):
    """Support Vector Regression baseline method."""

    def get_random_config(self, rng: random.Random) -> Dict[str, Any]:
        config = self.get_common_config(rng)
        config["svr_C"] = rng.uniform(0.1, 200.0)
        config["svr_epsilon"] = rng.uniform(0.01, 3.0)
        config["svr_kernel"] = rng.choice(["rbf", "poly", "sigmoid", "linear"])
        config["svr_degree"] = rng.choice([2, 3, 4, 5, 6])
        return config

    def instantiate_model(self, config: Dict[str, Any]) -> Any:
        degree = int(config.get("svr_degree", 3))
        if degree < 1:
            degree = 1
        return SVR(
            C=float(config.get("svr_C", 10.0)),
            epsilon=float(config.get("svr_epsilon", 0.5)),
            kernel=config.get("svr_kernel", "rbf"),
            degree=degree,
        )
    
    def fit(self, X_train: Any, y_train: Any) -> None:
        """SVR does not require special fitting logic beyond sklearn's fit."""
        pass


class PLSRMethod(BaselineMethod):
    """Partial Least Squares Regression baseline method (sigmoid-bounded)."""

    def get_random_config(self, rng: random.Random) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        # PLSR does not support dimensionality reduction (fixed to NONE)
        config["use_sg"] = rng.choice([True, False])
        config["sg_window"] = rng.choice([5, 7, 9, 11, 13, 15, 17, 19])
        config["sg_poly"] = rng.choice([2, 3, 4, 5])
        config["normalization"] = rng.choice(["NONE", "MINMAX", "SNV"])
        config["dim_method"] = "NONE"  # Forced for PLSR
        
        # PLSR-specific parameters
        config["plsr_n_components"] = rng.choice([4, 6, 8, 10, 12, 15, 20, 25])
        config["plsr_max_iter"] = rng.choice([300, 500, 800, 1200, 1600])
        config["plsr_tol"] = rng.choice([1e-6, 5e-6, 1e-5, 5e-5, 1e-4])
        return config

    def instantiate_model(self, config: Dict[str, Any]) -> Any:
        n_components = int(config.get("plsr_n_components", min(15, config.get("n_components", 10))))
        max_iter = int(config.get("plsr_max_iter", 600))
        tol = float(config.get("plsr_tol", 1e-6))
        return SigmoidBoundedPLSRegression(n_components=n_components, max_iter=max_iter, tol=tol)


class XGBMethod(BaselineMethod):
    """XGBoost baseline method."""

    def get_random_config(self, rng: random.Random) -> Dict[str, Any]:
        config = self.get_common_config(rng)
        config["xgb_n_estimators"] = rng.choice([60, 100, 140, 180, 220, 260])
        config["xgb_eta"] = rng.uniform(0.01, 0.3)
        config["xgb_max_depth"] = rng.choice([3, 4, 5, 6, 7, 8])
        config["xgb_subsample"] = rng.uniform(0.5, 1.0)
        config["xgb_colsample"] = rng.uniform(0.5, 1.0)
        config["xgb_reg_lambda"] = rng.uniform(0.1, 5.0)
        config["xgb_reg_alpha"] = rng.uniform(0.0, 2.0)
        config["xgb_gamma"] = rng.uniform(0.0, 2.0)
        config["random_state"] = rng.randint(1, 50_000)
        return config

    def instantiate_model(self, config: Dict[str, Any]) -> Any:
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


class RFMethod(BaselineMethod):
    """Random Forest baseline method."""

    def get_random_config(self, rng: random.Random) -> Dict[str, Any]:
        config = self.get_common_config(rng)
        config["rf_n_estimators"] = rng.choice([100, 150, 200, 300, 400, 500])
        config["rf_max_depth"] = rng.choice([None, 6, 10, 14, 18, 24, 32])
        config["rf_min_samples_split"] = rng.choice([2, 4, 6, 8])
        config["rf_min_samples_leaf"] = rng.choice([1, 2, 3, 4])
        config["rf_max_features"] = rng.choice(["sqrt", "log2", 0.5, 0.8, 1.0])
        config["random_state"] = rng.randint(1, 50_000)
        return config

    def instantiate_model(self, config: Dict[str, Any]) -> Any:
        return RandomForestRegressor(
            n_estimators=int(config.get("rf_n_estimators", 300)),
            max_depth=(None if config.get("rf_max_depth", None) in (None, "None") else int(config.get("rf_max_depth"))),
            min_samples_split=int(config.get("rf_min_samples_split", 2)),
            min_samples_leaf=int(config.get("rf_min_samples_leaf", 1)),
            max_features=config.get("rf_max_features", "sqrt"),
            random_state=int(config.get("random_state", 42)),
            n_jobs=1,
        )


class LRMethod(BaselineMethod):
    """Linear Regression baseline method."""

    def get_random_config(self, rng: random.Random) -> Dict[str, Any]:
        # LR has custom preprocessing options (subset of common features)
        config: Dict[str, Any] = {}
        config["use_sg"] = rng.choice([False, True])
        config["normalization"] = rng.choice(["NONE", "MINMAX", "SNV"])
        config["dim_method"] = rng.choice(["NONE", "PCA"])
        if config["dim_method"] != "NONE":
            config["n_components"] = rng.choice([10, 14, 18, 24, 32, 40])
        
        # LR-specific parameters
        config["lr_fit_intercept"] = rng.choice([True, False])
        config["lr_positive"] = rng.choice([False, True])
        return config

    def instantiate_model(self, config: Dict[str, Any]) -> Any:
        return LinearRegression(
            fit_intercept=bool(config.get("lr_fit_intercept", True)),
            positive=bool(config.get("lr_positive", False)),
        )


# Factory mapping for method names to class instances
METHOD_REGISTRY = {
    "SVR": SVRMethod(),
    "PLSR": PLSRMethod(),
    "XGB": XGBMethod(),
    "RF": RFMethod(),
    "LR": LRMethod(),
}


def get_method_instance(model_name: str) -> BaselineMethod:
    """Get the baseline method instance for the given model name.
    
    Args:
        model_name: Name of the method (SVR, PLSR, XGB, RF, LR)
        
    Returns:
        BaselineMethod instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported method: {model_name}. "
            f"Supported: {', '.join(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[model_name]
