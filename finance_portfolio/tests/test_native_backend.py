from __future__ import annotations

import numpy as np
import pytest

from finance_portfolio import finance_portfolio as native


def test_native_backend_exposes_portfolio_kernels() -> None:
    weights = native.minimum_variance_weights([[0.04, 0.0], [0.0, 0.16]])
    contribution = native.risk_contribution([0.8, 0.2], [[0.04, 0.0], [0.0, 0.16]])

    assert weights == pytest.approx([0.8, 0.2])
    assert sum(contribution[1]) == pytest.approx(np.sqrt(0.032))


def test_public_optimizer_delegates_to_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import finance_portfolio.optimization as optimization

    called: dict[str, object] = {}

    def fake_minimum_variance(covariance: list[list[float]]) -> list[float]:
        called["covariance"] = covariance
        return [0.25, 0.75]

    monkeypatch.setattr(optimization._native, "minimum_variance_weights", fake_minimum_variance)

    weights = optimization.minimum_variance_weights(np.eye(2), assets=["A", "B"])

    assert called == {"covariance": [[1.0, 0.0], [0.0, 1.0]]}
    assert weights == {"A": pytest.approx(0.25), "B": pytest.approx(0.75)}
