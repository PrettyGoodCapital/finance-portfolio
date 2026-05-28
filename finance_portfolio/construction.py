from __future__ import annotations

from typing import Iterable, Mapping

from . import finance_portfolio as _native

__all__ = ["equal_weights", "rank_weights", "signal_proportional_weights", "target_volatility_weights"]


def _as_dict(labels: tuple[str, ...], weights: list[float]) -> dict[str, float]:
    return {symbol: float(weight) for symbol, weight in zip(labels, weights)}


def equal_weights(assets: Iterable[str]) -> dict[str, float]:
    labels = tuple(str(asset) for asset in assets)
    if not labels:
        raise ValueError("assets must not be empty")
    return _as_dict(labels, _native.equal_weights(len(labels)))


def rank_weights(signals: Mapping[str, float], *, ascending: bool = True) -> dict[str, float]:
    if not signals:
        raise ValueError("signals must not be empty")
    labels = tuple(str(symbol) for symbol in signals)
    return _as_dict(labels, _native.rank_weights([float(value) for value in signals.values()], ascending))


def signal_proportional_weights(signals: Mapping[str, float]) -> dict[str, float]:
    if not signals:
        raise ValueError("signals must not be empty")
    labels = tuple(str(symbol) for symbol in signals)
    return _as_dict(labels, _native.signal_proportional_weights([float(value) for value in signals.values()]))


def target_volatility_weights(volatility: Mapping[str, float]) -> dict[str, float]:
    if not volatility:
        raise ValueError("volatility must not be empty")
    labels = tuple(str(symbol) for symbol in volatility)
    return _as_dict(labels, _native.target_volatility_weights([float(value) for value in volatility.values()]))
