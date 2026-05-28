from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import finance_portfolio as _native

__all__ = [
    "RiskContribution",
    "portfolio_variance",
    "portfolio_volatility",
    "risk_contribution",
    "tracking_error",
    "active_share",
]


class RiskContribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    marginal: np.ndarray
    component: np.ndarray
    percentage: np.ndarray


def _weights_vector(weights: Any, assets: Sequence[str] | None = None) -> np.ndarray:
    if isinstance(weights, Mapping):
        labels = tuple(assets) if assets is not None else tuple(weights)
        return np.asarray([float(weights.get(asset, 0.0)) for asset in labels], dtype=float)
    return np.asarray(weights, dtype=float).reshape(-1)


def portfolio_variance(weights: Any, covariance: Any) -> float:
    w = _weights_vector(weights)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (w.size, w.size):
        raise ValueError("covariance dimensions must match weights")
    return float(_native.portfolio_variance(w.tolist(), cov.tolist()))


def portfolio_volatility(weights: Any, covariance: Any) -> float:
    w = _weights_vector(weights)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (w.size, w.size):
        raise ValueError("covariance dimensions must match weights")
    return float(_native.portfolio_volatility(w.tolist(), cov.tolist()))


def risk_contribution(weights: Any, covariance: Any) -> RiskContribution:
    w = _weights_vector(weights)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (w.size, w.size):
        raise ValueError("covariance dimensions must match weights")
    marginal, component, percentage = _native.risk_contribution(w.tolist(), cov.tolist())
    return RiskContribution(
        marginal=np.asarray(marginal, dtype=float),
        component=np.asarray(component, dtype=float),
        percentage=np.asarray(percentage, dtype=float),
    )


def tracking_error(weights: Mapping[str, float] | Any, benchmark_weights: Mapping[str, float] | Any, covariance: Any) -> float:
    assets = (
        tuple(dict.fromkeys([*weights.keys(), *benchmark_weights.keys()]))
        if isinstance(weights, Mapping) and isinstance(benchmark_weights, Mapping)
        else None
    )
    active = _weights_vector(weights, assets) - _weights_vector(benchmark_weights, assets)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (active.size, active.size):
        raise ValueError("covariance dimensions must match weights")
    return float(_native.tracking_error(active.tolist(), cov.tolist()))


def active_share(weights: Mapping[str, float], benchmark_weights: Mapping[str, float]) -> float:
    assets = set(weights) | set(benchmark_weights)
    active = [float(weights.get(asset, 0.0)) - float(benchmark_weights.get(asset, 0.0)) for asset in assets]
    return float(_native.active_share(active))
