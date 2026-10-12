"""Phase 4 checks for MonteCarloEngine.price_asian_option. Run from the repo root: python -m pytest"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from energy_quant_engine.market_data import MarketData
from energy_quant_engine.pricing_engines import BlackScholesEngine, MonteCarloEngine

M = MarketData(S=100, K=100, T=1.0, r=0.05, sigma=0.20, asset_name="TEST")
N = 200_000
STEPS = 12  # monthly averaging keeps the tests fast


def asian(option_type="call", n_sims=N, n_steps=STEPS, seed=42, market=M):
    return MonteCarloEngine(market, seed=seed).price_asian_option(option_type, n_sims, n_steps)


def test_returns_two_floats():
    price, std_error = asian(n_sims=1_000)
    assert isinstance(price, float) and isinstance(std_error, float)
    assert price > 0 and std_error > 0


def test_asian_call_is_cheaper_than_european_call():
    # Averaging reduces volatility, so the Asian option is worth less.
    price, std_error = asian("call")
    european = BlackScholesEngine(M).price("call")
    assert price + 4 * std_error < european


def test_same_seed_gives_same_result():
    assert asian(n_sims=5_000, seed=1) == asian(n_sims=5_000, seed=1)


def test_standard_error_shrinks_like_one_over_root_n():
    _, se_small = asian(n_sims=20_000)
    _, se_large = asian(n_sims=80_000)  # 4x the paths -> half the error
    assert 1.8 < se_small / se_large < 2.2


def test_geometric_average_matches_closed_form():
    # Independent check of the PATH-DEPENDENT behaviour of simulate_paths.
    # A discretely monitored GEOMETRIC Asian call has an exact formula, so we
    # price one from our simulated paths and compare.
    n = STEPS
    paths = MonteCarloEngine(M, seed=42).simulate_paths(N, n)
    geo_mean = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    discount = math.exp(-M.r * M.T)
    payoff = np.maximum(geo_mean - M.K, 0)
    mc_price = discount * payoff.mean()
    std_error = discount * payoff.std(ddof=1) / math.sqrt(N)

    # ln(geometric mean) is Normal(mu, v):
    mu = math.log(M.S) + (M.r - 0.5 * M.sigma**2) * M.T * (n + 1) / (2 * n)
    v = M.sigma**2 * M.T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d2 = (mu - math.log(M.K)) / math.sqrt(v)
    d1 = d2 + math.sqrt(v)
    exact = discount * (math.exp(mu + v / 2) * norm.cdf(d1) - M.K * norm.cdf(d2))

    assert abs(mc_price - exact) < 4 * std_error


def test_arithmetic_price_is_at_least_geometric_price():
    # For any set of positive numbers, arithmetic mean >= geometric mean,
    # so on identical paths the arithmetic Asian call can never be cheaper.
    arithmetic, _ = asian("call", n_sims=50_000, seed=3)
    paths = MonteCarloEngine(M, seed=3).simulate_paths(50_000, STEPS)
    geo_mean = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    geometric = math.exp(-M.r * M.T) * np.maximum(geo_mean - M.K, 0).mean()
    assert arithmetic >= geometric


def test_rejects_bad_inputs():
    engine = MonteCarloEngine(M, seed=1)
    with pytest.raises(ValueError):
        engine.price_asian_option("straddle")
    with pytest.raises(ValueError):
        engine.price_asian_option(n_sims=1)
    with pytest.raises(TypeError):
        engine.price_asian_option(n_sims=100.5)
