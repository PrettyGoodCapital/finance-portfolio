from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict

from . import finance_portfolio as _native

__all__ = [
    "BrinsonAttribution",
    "ReturnAttribution",
    "brinson_attribution",
    "factor_return_decomposition",
    "return_attribution_decomposition",
]


class BrinsonAttribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float


class ReturnAttribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    components: dict[str, float]
    total_return: float


def brinson_attribution(
    *,
    portfolio_weights: Mapping[str, float],
    benchmark_weights: Mapping[str, float],
    portfolio_returns: Mapping[str, float],
    benchmark_returns: Mapping[str, float],
) -> BrinsonAttribution:
    groups = tuple(dict.fromkeys([*portfolio_weights, *benchmark_weights, *portfolio_returns, *benchmark_returns]))
    allocation, selection, interaction, total = _native.brinson_attribution(
        [float(portfolio_weights.get(group, 0.0)) for group in groups],
        [float(benchmark_weights.get(group, 0.0)) for group in groups],
        [float(portfolio_returns.get(group, 0.0)) for group in groups],
        [float(benchmark_returns.get(group, 0.0)) for group in groups],
    )
    return BrinsonAttribution(
        allocation_effect=allocation,
        selection_effect=selection,
        interaction_effect=interaction,
        total_effect=total,
    )


def factor_return_decomposition(exposures: Mapping[str, float], factor_returns: Mapping[str, float]) -> ReturnAttribution:
    factors = tuple(str(factor) for factor in factor_returns)
    components_raw, total = _native.factor_return_decomposition(
        [float(exposures.get(factor, 0.0)) for factor in factors],
        [float(factor_returns[factor]) for factor in factor_returns],
    )
    components = {factor: float(components_raw[idx]) for idx, factor in enumerate(factors)}
    return ReturnAttribution(components=components, total_return=total)


def return_attribution_decomposition(components: Mapping[str, float]) -> ReturnAttribution:
    normalized = {str(name): float(value) for name, value in components.items()}
    components_raw, total = _native.factor_return_decomposition([1.0 for _name in normalized], list(normalized.values()))
    return ReturnAttribution(components={name: float(components_raw[idx]) for idx, name in enumerate(normalized)}, total_return=total)
