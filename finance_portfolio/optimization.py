from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import finance_portfolio as _native

__all__ = ["minimum_variance_weights", "mean_variance_weights", "risk_parity_weights", "hierarchical_risk_parity_weights"]


def _labels(assets: Sequence[str] | None, size: int) -> tuple[str, ...]:
    labels = tuple(str(asset) for asset in assets) if assets is not None else tuple(f"asset_{i}" for i in range(size))
    if len(labels) != size:
        raise ValueError("assets length must match covariance dimensions")
    return labels


def _covariance(covariance: Any) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    return cov


def _as_dict(weights: np.ndarray, labels: tuple[str, ...]) -> dict[str, float]:
    return {asset: float(weight) for asset, weight in zip(labels, weights)}


def minimum_variance_weights(covariance: Any, *, assets: Sequence[str] | None = None) -> dict[str, float]:
    cov = _covariance(covariance)
    labels = _labels(assets, cov.shape[0])
    weights = np.asarray(_native.minimum_variance_weights(cov.tolist()), dtype=float)
    return _as_dict(weights, labels)


def mean_variance_weights(
    expected_returns: Any, covariance: Any, *, risk_aversion: float = 1.0, assets: Sequence[str] | None = None
) -> dict[str, float]:
    cov = _covariance(covariance)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    if mu.size != cov.shape[0]:
        raise ValueError("expected_returns length must match covariance dimensions")
    labels = _labels(assets, cov.shape[0])
    weights = np.asarray(_native.mean_variance_weights(mu.tolist(), cov.tolist(), float(risk_aversion)), dtype=float)
    return _as_dict(weights, labels)


def risk_parity_weights(covariance: Any, *, assets: Sequence[str] | None = None, max_iter: int = 1_000, tolerance: float = 1e-10) -> dict[str, float]:
    cov = _covariance(covariance)
    labels = _labels(assets, cov.shape[0])
    weights = np.asarray(_native.risk_parity_weights(cov.tolist(), max_iter, tolerance), dtype=float)
    return _as_dict(weights, labels)


def hierarchical_risk_parity_weights(covariance: Any, *, assets: Sequence[str] | None = None) -> dict[str, float]:
    cov = _covariance(covariance)
    labels = _labels(assets, cov.shape[0])
    weights = np.asarray(_native.hierarchical_risk_parity_weights(cov.tolist()), dtype=float)
    return _as_dict(weights, labels)
