from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted


PLSR_TARGET_SCALE = 6.0


class SigmoidTargetTransform:
    def __init__(self, scale: float = PLSR_TARGET_SCALE, eps: float = 1e-4):
        self.scale = float(scale)
        self.eps = float(eps)

    def fit(self, y: np.ndarray) -> "SigmoidTargetTransform":
        target = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.target_min_ = float(np.min(target))
        self.target_max_ = float(np.max(target))
        self.target_range_ = max(self.target_max_ - self.target_min_, self.eps)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        target = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        normalized = (target - self.target_min_) / self.target_range_
        normalized = np.clip(normalized, self.eps, 1.0 - self.eps)
        logits = np.log(normalized / (1.0 - normalized))
        return logits / self.scale

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        transformed = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        scaled = np.clip(transformed * self.scale, -60.0, 60.0)
        normalized = 1.0 / (1.0 + np.exp(-scaled))
        restored = normalized * self.target_range_ + self.target_min_
        return restored

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def __str__(self) -> str:
        return f"SigmoidTargetTransform(scale={self.scale:g})"


class SigmoidBoundedPLSRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int = 2, max_iter: int = 500, tol: float = 1e-6, scale: float = PLSR_TARGET_SCALE):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scale = float(scale)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SigmoidBoundedPLSRegression":
        features = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        max_components = max(1, min(self.n_components, features.shape[0], features.shape[1]))
        self.transformer_ = SigmoidTargetTransform(scale=self.scale)
        transformed_target = self.transformer_.fit_transform(target)
        self.model_ = PLSRegression(n_components=max_components, max_iter=self.max_iter, tol=self.tol)
        self.model_.fit(features, transformed_target)
        self.n_components_ = max_components
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "transformer_"])
        features = np.asarray(X, dtype=np.float64)
        transformed_prediction = self.model_.predict(features)
        prediction = self.transformer_.inverse_transform(transformed_prediction)
        return prediction.reshape(-1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "scale": self.scale,
        }

    def set_params(self, **params: Any) -> "SigmoidBoundedPLSRegression":
        for key, value in params.items():
            setattr(self, key, value)
        self.n_components = int(self.n_components)
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)
        self.scale = float(self.scale)
        return self