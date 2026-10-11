"""Contract layer: the market inputs every pricing engine consumes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


def _check_number(name: str, value: object, *, positive: bool) -> None:
    """Raise if `value` is not a usable real number.

    positive=True  -> must be strictly greater than zero (S, K, T, sigma)
    positive=False -> any finite real is fine, including negative (r)
    """
    # bool is a subclass of int in Python, so True would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    # NaN and infinity are numbers, but they would silently poison every calculation.
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}")


@dataclass(frozen=True)
class MarketData:
    """Immutable snapshot of the inputs for pricing one option.

    Attributes
    ----------
    S : spot price of the underlying (e.g. dollars per barrel)
    K : strike price, in the same units as S
    T : time to expiry in years
    r : continuously compounded risk-free rate, as a decimal (0.05 = 5%)
    sigma : annualised volatility, as a decimal (0.30 = 30%)
    asset_name : ticker/label, e.g. 'BRENT_CRUDE'
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    asset_name: str

    def __post_init__(self) -> None:
        # Runs automatically right after the dataclass sets the fields,
        # so an invalid MarketData can never exist.
        for name in ("S", "K", "T", "sigma"):
            _check_number(name, getattr(self, name), positive=True)
        # Rates can legitimately be zero or negative, so only check it is a finite number.
        _check_number("r", self.r, positive=False)

        if not isinstance(self.asset_name, str):
            raise TypeError(
                f"asset_name must be a string, got {type(self.asset_name).__name__}"
            )
        if not self.asset_name.strip():
            raise ValueError("asset_name must not be empty")
