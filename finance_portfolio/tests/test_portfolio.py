from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from finance_enums import Frequency

import finance_portfolio as fp


def test_portfolio_structures_and_weight_construction() -> None:
    portfolio = fp.Portfolio.from_weights(
        {"A": 0.6, "B": 0.4},
        name="core",
        as_of=date(2024, 1, 31),
        calendar="XNYS",
        frequency="monthly",
    )

    assert portfolio.name == "core"
    assert portfolio.as_of == date(2024, 1, 31)
    assert portfolio.calendar == "XNYS"
    assert portfolio.frequency == Frequency.Month
    assert portfolio.net_exposure == pytest.approx(1.0)
    assert portfolio.gross_exposure == pytest.approx(1.0)
    assert portfolio.weight("A") == pytest.approx(0.6)

    assert fp.equal_weights(["A", "B", "C"]) == {"A": pytest.approx(1 / 3), "B": pytest.approx(1 / 3), "C": pytest.approx(1 / 3)}
    assert fp.rank_weights({"A": 3.0, "B": 1.0, "C": 2.0})["A"] > fp.rank_weights({"A": 3.0, "B": 1.0, "C": 2.0})["B"]
    assert fp.signal_proportional_weights({"A": 2.0, "B": -1.0, "C": 0.0}) == {
        "A": pytest.approx(2 / 3),
        "B": pytest.approx(-1 / 3),
        "C": pytest.approx(0.0),
    }
    assert fp.target_volatility_weights({"A": 0.10, "B": 0.20})["A"] == pytest.approx(2 / 3)


def test_optimizers_and_risk_contributions() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.16]])

    min_var = fp.minimum_variance_weights(cov, assets=["A", "B"])
    risk_parity = fp.risk_parity_weights(cov, assets=["A", "B"])
    hrp = fp.hierarchical_risk_parity_weights(cov, assets=["A", "B"])
    contributions = fp.risk_contribution(np.array([min_var["A"], min_var["B"]]), cov)

    assert min_var["A"] == pytest.approx(0.8)
    assert min_var["B"] == pytest.approx(0.2)
    assert risk_parity["A"] == pytest.approx(2 / 3, rel=1e-3)
    assert hrp["A"] + hrp["B"] == pytest.approx(1.0)
    assert contributions.component.sum() == pytest.approx(fp.portfolio_volatility(np.array([0.8, 0.2]), cov))


def test_tracking_active_share_and_attribution() -> None:
    weights = {"A": 0.6, "B": 0.4}
    benchmark_weights = {"A": 0.5, "B": 0.5}
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])

    assert fp.tracking_error(weights, benchmark_weights, cov) > 0.0
    assert fp.active_share(weights, benchmark_weights) == pytest.approx(0.1)

    brinson = fp.brinson_attribution(
        portfolio_weights={"Tech": 0.7, "Energy": 0.3},
        benchmark_weights={"Tech": 0.5, "Energy": 0.5},
        portfolio_returns={"Tech": 0.04, "Energy": 0.01},
        benchmark_returns={"Tech": 0.03, "Energy": 0.02},
    )
    decomposition = fp.return_attribution_decomposition({"alpha": 0.02, "factor": 0.01, "cost": -0.005})

    assert brinson.total_effect == pytest.approx(brinson.allocation_effect + brinson.selection_effect + brinson.interaction_effect)
    assert decomposition.total_return == pytest.approx(0.025)


def test_index_structures_normalize_constituents() -> None:
    index = fp.Index.from_constituents(
        {"A": 50.0, "B": 50.0},
        name="PGC 2",
        as_of=date(2024, 1, 31),
        divisor=1000.0,
        frequency=Frequency.Quarter,
    )

    assert index.name == "PGC 2"
    assert index.as_of == date(2024, 1, 31)
    assert index.divisor == pytest.approx(1000.0)
    assert index.frequency == Frequency.Quarter
    assert index.weights == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}


def test_portfolio_analysis_helpers() -> None:
    weights = {"A": 0.60, "B": 0.40}
    benchmark = {"A": 0.50, "B": 0.50}
    target = {"A": 0.55, "B": 0.45}
    returns = {"A": 0.02, "B": -0.01}

    active = fp.active_weights(weights, benchmark)
    drift = fp.weight_drift(weights, target)
    turn = fp.turnover(weights, target)
    contrib = fp.return_contributions(weights, returns)

    assert active == {"A": pytest.approx(0.10), "B": pytest.approx(-0.10)}
    assert drift == {"A": pytest.approx(0.05), "B": pytest.approx(-0.05)}
    assert turn == pytest.approx(0.05)
    assert contrib == {"A": pytest.approx(0.012), "B": pytest.approx(-0.004)}


def test_rebalance_schedule_and_panel_output() -> None:
    schedule = fp.rebalance_schedule(
        [
            date(2024, 1, 2),
            date(2024, 1, 15),
            date(2024, 2, 1),
            date(2024, 3, 4),
            date(2024, 3, 8),
        ],
        "monthly",
    )
    assert schedule == (date(2024, 1, 2), date(2024, 2, 1), date(2024, 3, 4))

    panel = fp.portfolio_analysis_panel(
        as_of=date(2024, 1, 31),
        weights={"A": 0.60, "B": 0.40},
        benchmark_weights={"A": 0.50, "B": 0.50},
        returns={"A": 0.01, "B": 0.02},
    )
    assert isinstance(panel, pl.DataFrame)
    assert panel.height == 2
    assert set(panel.columns) == {
        "as_of",
        "symbol",
        "weight",
        "benchmark_weight",
        "active_weight",
        "return",
        "contribution",
    }
    assert panel.filter(pl.col("symbol") == "A")["active_weight"].item() == pytest.approx(0.10)
