"""Phase 3 checks for MonteCarloEngine.simulate_paths. Run from the repo root: python -m pytest"""

import math

import numpy as np
import pytest

from energy_quant_engine.market_data import MarketData
from energy_quant_engine.pricing_engines import BlackScholesEngine, MonteCarloEngine

M = MarketData(S=100, K=100, T=1.0, r=0.05, sigma=0.20, asset_name="TEST")
N = 200_000


def make_paths(n_sims=N, n_steps=12, seed=42):
    return MonteCarloEngine(M, seed=seed).simulate_paths(n_sims, n_steps)


def test_shape_and_starting_column():
    paths = make_paths(n_sims=1_000, n_steps=252)
    assert paths.shape == (1_000, 253)
    assert np.all(paths[:, 0] == M.S)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)


def test_chunking_fills_every_row():
    # 25,000 is not a multiple of the 10,000 chunk size, so the last chunk is partial.
    paths = make_paths(n_sims=25_000, n_steps=5)
    assert paths.shape == (25_000, 6)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)


def test_seed_gives_reproducible_paths():
    assert np.array_equal(make_paths(seed=7, n_sims=500), make_paths(seed=7, n_sims=500))
    assert not np.array_equal(make_paths(seed=7, n_sims=500), make_paths(seed=8, n_sims=500))


def test_terminal_mean_matches_forward_price():
    # Under risk-neutral GBM, E[S_T] = S * exp(rT)
    s_t = make_paths()[:, -1]
    std_error = s_t.std(ddof=1) / math.sqrt(N)
    assert abs(s_t.mean() - M.S * math.exp(M.r * M.T)) < 4 * std_error


def test_log_returns_have_the_right_distribution():
    # log(S_T / S) should be Normal with mean (r - sigma^2/2)T and std sigma*sqrt(T)
    log_ret = np.log(make_paths()[:, -1] / M.S)
    expected_mean = (M.r - 0.5 * M.sigma**2) * M.T
    assert abs(log_ret.mean() - expected_mean) < 4 * log_ret.std(ddof=1) / math.sqrt(N)
    assert abs(log_ret.std(ddof=1) - M.sigma * math.sqrt(M.T)) < 0.002


def test_european_mc_price_matches_black_scholes():
    s_t = make_paths()[:, -1]
    discount = math.exp(-M.r * M.T)
    payoffs = {"call": np.maximum(s_t - M.K, 0), "put": np.maximum(M.K - s_t, 0)}
    for option_type, payoff in payoffs.items():
        mc_price = discount * payoff.mean()
        std_error = discount * payoff.std(ddof=1) / math.sqrt(N)
        bs_price = BlackScholesEngine(M).price(option_type)
        assert abs(mc_price - bs_price) < 4 * std_error


def test_rejects_bad_sizes():
    engine = MonteCarloEngine(M, seed=1)
    with pytest.raises(ValueError):
        engine.simulate_paths(n_sims=0)
    with pytest.raises(ValueError):
        engine.simulate_paths(n_steps=-5)
    with pytest.raises(TypeError):
        engine.simulate_paths(n_sims=10.5)
    with pytest.raises(TypeError):
        engine.simulate_paths(n_steps=True)
