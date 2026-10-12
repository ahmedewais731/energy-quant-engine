"""Phase 5 checks for the analytics layer. Run from the repo root: python -m pytest"""

import math
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # draw to memory/files only, so tests work without a display
import matplotlib.pyplot as plt
import numpy as np
import pytest

from energy_quant_engine.main_analytics import ScenarioAnalytics, running_estimates
from energy_quant_engine.market_data import MarketData
from energy_quant_engine.pricing_engines import BlackScholesEngine, MonteCarloEngine

M = MarketData(S=100, K=100, T=1.0, r=0.05, sigma=0.20, asset_name="TEST")


def test_greeks_table_layout_and_values():
    table = ScenarioAnalytics(M).generate_greeks_table()
    assert list(table.index) == ["call", "put"]
    assert list(table.columns) == ["price", "delta", "gamma", "vega", "theta"]
    engine = BlackScholesEngine(M)
    for option_type in ("call", "put"):
        assert table.loc[option_type, "price"] == engine.price(option_type)
        assert table.loc[option_type, "delta"] == engine.greeks(option_type)["delta"]
    # gamma and vega are the same for a call and a put
    assert table.loc["call", "gamma"] == table.loc["put", "gamma"]
    assert table.loc["call", "vega"] == table.loc["put", "vega"]


def test_running_estimates_final_row_matches_full_sample():
    values = np.random.default_rng(0).normal(5.0, 2.0, size=10_000)
    est = running_estimates(values)
    last = est.iloc[-1]
    assert last["n_sims"] == 10_000
    assert abs(last["price"] - values.mean()) < 1e-9
    assert abs(last["std_error"] - values.std(ddof=1) / math.sqrt(10_000)) < 1e-9
    assert est["n_sims"].is_monotonic_increasing


def test_running_estimates_matches_direct_calculation_midway():
    values = np.random.default_rng(1).normal(size=5_000)
    est = running_estimates(values)
    row = est.iloc[len(est) // 2]
    n = int(row["n_sims"])
    assert abs(row["price"] - values[:n].mean()) < 1e-9
    assert abs(row["std_error"] - values[:n].std(ddof=1) / math.sqrt(n)) < 1e-9


def test_running_estimates_needs_two_values():
    with pytest.raises(ValueError):
        running_estimates(np.array([1.0]))


def test_convergence_plot_builds_and_saves():
    path = os.path.join(tempfile.mkdtemp(), "convergence.png")
    fig = ScenarioAnalytics(M, seed=1).plot_mc_convergence(
        "call", n_sims=5_000, n_steps=20, n_paths_shown=30, save_path=path
    )
    assert len(fig.axes) == 2
    assert os.path.getsize(path) > 10_000
    plt.close(fig)


def test_convergence_plot_rejects_bad_option_type():
    with pytest.raises(ValueError):
        ScenarioAnalytics(M).plot_mc_convergence("straddle", n_sims=100, n_steps=5)


def test_asian_price_unchanged_by_payoff_refactor():
    # price_asian_option now goes through asian_payoffs; same seed must still agree.
    engine = MonteCarloEngine(M, seed=5)
    price, _ = engine.price_asian_option("call", n_sims=3_000, n_steps=10)
    paths = MonteCarloEngine(M, seed=5).simulate_paths(3_000, 10)
    assert abs(price - MonteCarloEngine(M).asian_payoffs(paths, "call").mean()) < 1e-12
