from __future__ import annotations

from datetime import date
from typing import Mapping

from finance_enums import Frequency, to_frequency
from pydantic import BaseModel, ConfigDict

__all__ = ["Holding", "Portfolio", "IndexConstituent", "Index"]


class Holding(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    weight: float
    currency: str | None = None
    exchange: str | None = None


class Portfolio(BaseModel):
    model_config = ConfigDict(frozen=True)

    holdings: tuple[Holding, ...]
    name: str | None = None
    as_of: date | None = None
    calendar: str | None = None
    frequency: Frequency | None = None

    @classmethod
    def _coerce_frequency(cls, value: Frequency | str | None) -> Frequency | None:
        if value is None or isinstance(value, Frequency):
            return value
        return to_frequency(value)

    @classmethod
    def from_weights(
        cls,
        weights: Mapping[str, float],
        *,
        name: str | None = None,
        as_of: date | None = None,
        calendar: str | None = None,
        frequency: Frequency | str | None = None,
    ) -> "Portfolio":
        return cls(
            holdings=tuple(Holding(symbol=str(symbol), weight=float(weight)) for symbol, weight in weights.items()),
            name=name,
            as_of=as_of,
            calendar=calendar,
            frequency=cls._coerce_frequency(frequency),
        )

    @property
    def weights(self) -> dict[str, float]:
        return {holding.symbol: holding.weight for holding in self.holdings}

    @property
    def net_exposure(self) -> float:
        return sum(holding.weight for holding in self.holdings)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(holding.weight) for holding in self.holdings)

    def weight(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.0)


class IndexConstituent(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    weight: float
    free_float: float | None = None


class Index(BaseModel):
    model_config = ConfigDict(frozen=True)

    constituents: tuple[IndexConstituent, ...]
    name: str | None = None
    as_of: date | None = None
    divisor: float | None = None
    frequency: Frequency | None = None

    @classmethod
    def from_constituents(
        cls,
        values: Mapping[str, float],
        *,
        name: str | None = None,
        as_of: date | None = None,
        divisor: float | None = None,
        frequency: Frequency | str | None = None,
    ) -> "Index":
        total = sum(float(value) for value in values.values())
        if total <= 0.0:
            raise ValueError("constituent values must have positive total")
        return cls(
            constituents=tuple(IndexConstituent(symbol=str(symbol), weight=float(value) / total) for symbol, value in values.items()),
            name=name,
            as_of=as_of,
            divisor=divisor,
            frequency=Portfolio._coerce_frequency(frequency),
        )

    @property
    def weights(self) -> dict[str, float]:
        return {constituent.symbol: constituent.weight for constituent in self.constituents}
