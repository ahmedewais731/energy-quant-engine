"""Engine layer: pricing models that turn a MarketData into prices and risk numbers.

Contains BlackScholesEngine (closed-form European options).
MonteCarloEngine will be added here in a later phase.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from energy_quant_engine.market_data import MarketData

_VALID_OPTION_TYPES = ("call", "put")


def _check_option_type(option_type: str) -> None:
    """Raise if option_type is not exactly 'call' or 'put'."""
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


class BlackScholesEngine:
    """Closed-form Black-Scholes pricing and Greeks for a European option.

    Greek conventions:
        delta : change in price per 1.00 change in S
        gamma : change in delta per 1.00 change in S
        vega  : change in price per 1.00 change in sigma (1.00 = 100 vol points)
        theta : change in price per YEAR of time passing (negative for a long option)
    """

    def __init__(self, market: MarketData) -> None:
        self.market = market

        # d1 and d2 depend only on the inputs, and MarketData is frozen,
        # so we compute them once here and reuse them everywhere.
        self._sqrt_T = math.sqrt(market.T)
        self._d1 = (
            math.log(market.S / market.K) + (market.r + 0.5 * market.sigma**2) * market.T
        ) / (market.sigma * self._sqrt_T)
        self._d2 = self._d1 - market.sigma * self._sqrt_T
        self._discount = math.exp(-market.r * market.T)  # e^(-rT)

    def price(self, option_type: str = "call") -> float:
        """European option price."""
        _check_option_type(option_type)
        m = self.market
        if option_type == "call":
            value = m.S * norm.cdf(self._d1) - m.K * self._discount * norm.cdf(self._d2)
        else:
            value = m.K * self._discount * norm.cdf(-self._d2) - m.S * norm.cdf(-self._d1)
        return float(value)

    def greeks(self, option_type: str = "call") -> dict:
        """Delta, gamma, vega and theta as a dict of floats."""
        _check_option_type(option_type)
        m = self.market
        pdf_d1 = norm.pdf(self._d1)  # N'(d1), the bell-curve height at d1

        # Gamma and vega are identical for calls and puts.
        gamma = pdf_d1 / (m.S * m.sigma * self._sqrt_T)
        vega = m.S * pdf_d1 * self._sqrt_T

        # Delta and theta differ between calls and puts.
        decay = -m.S * pdf_d1 * m.sigma / (2 * self._sqrt_T)
        if option_type == "call":
            delta = norm.cdf(self._d1)
            theta = decay - m.r * m.K * self._discount * norm.cdf(self._d2)
        else:
            delta = norm.cdf(self._d1) - 1
            theta = decay + m.r * m.K * self._discount * norm.cdf(-self._d2)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
        }
