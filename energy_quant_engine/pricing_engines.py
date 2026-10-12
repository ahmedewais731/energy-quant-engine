"""Engine layer: pricing models that turn a MarketData into prices and risk numbers.

Contains:
    BlackScholesEngine : closed-form European options
    MonteCarloEngine   : simulated price paths and arithmetic-average Asian options
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from energy_quant_engine.market_data import MarketData

_VALID_OPTION_TYPES = ("call", "put")


def _check_option_type(option_type: str) -> None:
    """Raise if option_type is not exactly 'call' or 'put'."""
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _check_positive_int(name: str, value: object) -> int:
    """Return `value` as a plain int, or raise if it is not a positive whole number."""
    # bool is a subclass of int, so True would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


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


class MonteCarloEngine:
    """Monte Carlo engine: simulates asset price paths under risk-neutral GBM.

    Each step multiplies the price by exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z),
    where Z is a fresh standard normal draw.
    """

    # Paths are generated this many at a time so the temporary random-number
    # arrays stay small (about 20 MB) instead of doubling memory use.
    _CHUNK_SIZE = 10_000

    def __init__(self, market: MarketData, seed: int | None = None) -> None:
        self.market = market
        # One random generator per engine. Passing the same seed reproduces
        # exactly the same paths; seed=None gives different paths every run.
        self._rng = np.random.default_rng(seed)

    def simulate_paths(self, n_sims: int = 100_000, n_steps: int = 252) -> np.ndarray:
        """Simulate GBM price paths.

        Returns an array of shape (n_sims, n_steps + 1):
            * each ROW is one simulated path
            * column 0 is the starting spot price S (time 0)
            * column j is the price after j time steps (column n_steps is expiry)
        """
        n_sims = _check_positive_int("n_sims", n_sims)
        n_steps = _check_positive_int("n_steps", n_steps)
        m = self.market

        dt = m.T / n_steps                          # length of one step, in years
        drift = (m.r - 0.5 * m.sigma**2) * dt       # deterministic part of each step
        vol = m.sigma * math.sqrt(dt)               # scale of the random part

        paths = np.empty((n_sims, n_steps + 1))
        paths[:, 0] = m.S

        for start in range(0, n_sims, self._CHUNK_SIZE):
            stop = min(start + self._CHUNK_SIZE, n_sims)
            z = self._rng.standard_normal((stop - start, n_steps))  # the Z's
            log_returns = drift + vol * z                           # one per step
            # cumsum adds up the log-returns along each path, and exp turns the
            # running total back into a price multiplier relative to S.
            paths[start:stop, 1:] = m.S * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def asian_payoffs(self, paths: np.ndarray, option_type: str = "call") -> np.ndarray:
        """Discounted payoff of an arithmetic-average Asian option on each path.

        `paths` is an array as returned by simulate_paths. The average is taken
        over columns 1 onwards (the starting price in column 0 is excluded).
        Returns one discounted payoff per path.
        """
        _check_option_type(option_type)
        m = self.market

        average = paths[:, 1:].mean(axis=1)  # one arithmetic average per path
        if option_type == "call":
            payoff = np.maximum(average - m.K, 0.0)
        else:
            payoff = np.maximum(m.K - average, 0.0)
        return math.exp(-m.r * m.T) * payoff

    def price_asian_option(
        self,
        option_type: str = "call",
        n_sims: int = 100_000,
        n_steps: int = 252,
    ) -> tuple[float, float]:
        """Price an arithmetic-average Asian option (fixed strike) by Monte Carlo.

        Returns (price, standard_error). The standard error is on the same
        scale as the price (i.e. it is discounted too).
        """
        _check_option_type(option_type)
        n_sims = _check_positive_int("n_sims", n_sims)
        if n_sims < 2:
            raise ValueError("n_sims must be at least 2 to estimate a standard error")

        paths = self.simulate_paths(n_sims, n_steps)
        payoffs = self.asian_payoffs(paths, option_type)

        price = payoffs.mean()
        # Standard error of a sample mean: std of the samples / sqrt(count).
        std_error = payoffs.std(ddof=1) / math.sqrt(n_sims)
        return float(price), float(std_error)
