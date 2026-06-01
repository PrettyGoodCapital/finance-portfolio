from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import polars as pl
from finance_enums import Frequency, to_frequency

__all__ = [
    "active_weights",
    "weight_drift",
    "turnover",
    "return_contributions",
    "rebalance_schedule",
    "portfolio_analysis_panel",
]


def _union_symbols(left: Mapping[str, float], right: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*left.keys(), *right.keys()]))


def active_weights(weights: Mapping[str, float], benchmark_weights: Mapping[str, float]) -> dict[str, float]:
    symbols = _union_symbols(weights, benchmark_weights)
    return {symbol: float(weights.get(symbol, 0.0) - benchmark_weights.get(symbol, 0.0)) for symbol in symbols}


def weight_drift(current_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> dict[str, float]:
    symbols = _union_symbols(current_weights, target_weights)
    return {symbol: float(current_weights.get(symbol, 0.0) - target_weights.get(symbol, 0.0)) for symbol in symbols}


def turnover(previous_weights: Mapping[str, float], next_weights: Mapping[str, float]) -> float:
    symbols = _union_symbols(previous_weights, next_weights)
    total_abs_diff = sum(abs(float(previous_weights.get(symbol, 0.0) - next_weights.get(symbol, 0.0))) for symbol in symbols)
    return 0.5 * total_abs_diff


def return_contributions(weights: Mapping[str, float], returns: Mapping[str, float]) -> dict[str, float]:
    symbols = _union_symbols(weights, returns)
    return {symbol: float(weights.get(symbol, 0.0) * returns.get(symbol, 0.0)) for symbol in symbols}


def rebalance_schedule(dates: Sequence[date], frequency: Frequency | str) -> tuple[date, ...]:
    freq = to_frequency(frequency) if isinstance(frequency, str) else frequency
    if not dates:
        return ()
    ordered = sorted(dates)

    def _key(value: date) -> tuple[int, int, int]:
        if freq == Frequency.Week:
            iso_year, iso_week, _ = value.isocalendar()
            return iso_year, iso_week, 1
        if freq == Frequency.Month:
            return value.year, value.month, 1
        if freq == Frequency.Quarter:
            return value.year, ((value.month - 1) // 3) + 1, 1
        if freq == Frequency.Year:
            return value.year, 1, 1
        return value.year, value.month, value.day

    out: list[date] = []
    last = None
    for value in ordered:
        bucket = _key(value)
        if bucket != last:
            out.append(value)
            last = bucket
    return tuple(out)


def portfolio_analysis_panel(
    *,
    as_of: date,
    weights: Mapping[str, float],
    benchmark_weights: Mapping[str, float] | None = None,
    returns: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    benchmark = benchmark_weights or {}
    daily_returns = returns or {}
    symbols = tuple(dict.fromkeys([*weights.keys(), *benchmark.keys(), *daily_returns.keys()]))
    if not symbols:
        return pl.DataFrame(
            {
                "as_of": pl.Series([], dtype=pl.Date),
                "symbol": pl.Series([], dtype=pl.String),
                "weight": pl.Series([], dtype=pl.Float64),
                "benchmark_weight": pl.Series([], dtype=pl.Float64),
                "active_weight": pl.Series([], dtype=pl.Float64),
                "return": pl.Series([], dtype=pl.Float64),
                "contribution": pl.Series([], dtype=pl.Float64),
            }
        )

    rows = []
    for symbol in symbols:
        weight = float(weights.get(symbol, 0.0))
        bench = float(benchmark.get(symbol, 0.0))
        ret = float(daily_returns.get(symbol, 0.0))
        rows.append(
            {
                "as_of": as_of,
                "symbol": symbol,
                "weight": weight,
                "benchmark_weight": bench,
                "active_weight": weight - bench,
                "return": ret,
                "contribution": weight * ret,
            }
        )
    return pl.DataFrame(rows)
