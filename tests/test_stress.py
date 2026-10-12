"""Phase 7 checks: stress testing and the price surface. Run from the repo root: python -m pytest"""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from energy_quant_engine.main_analytics import ScenarioAnalytics, stress_pnl_grid
from energy_quant_engine.market_data import MarketData
from energy_quant_engine.pricing_engines import BlackScholesEngine

M = MarketData(S=80.0, K=85.0, T=0.5, r=0.04, sigma=0.35, asset_name="BRENT_CRUDE")
A = ScenarioAnalytics(M)


def test_stress_table_layout():
    table = A.stress_test("call")
    assert len(table) == 5 * 4
    assert list(table.columns) == ["spot_shock", "vol_shock", "spot", "sigma", "price", "pnl"]
    row = table[(table.spot_shock == 0.10) & (table.vol_shock == 0.20)].iloc[0]
    assert row["spot"] == pytest.approx(M.S * 1.10, abs=1e-12)
    assert row["sigma"] == pytest.approx(M.sigma + 0.20, abs=1e-12)


def test_zero_shock_has_zero_pnl():
    table = A.stress_test("put")
    row = table[(table.spot_shock == 0.0) & (table.vol_shock == 0.0)].iloc[0]
    assert row["price"] == pytest.approx(BlackScholesEngine(M).price("put"), abs=1e-12)
    assert row["pnl"] == pytest.approx(0.0, abs=1e-12)


def test_pnl_moves_the_right_way():
    for option_type, sign in (("call", +1), ("put", -1)):
        table = A.stress_test(option_type)
        # Calls gain and puts lose when spot rises (holding vol fixed)
        for _, group in table.groupby("vol_shock"):
            pnl = group.sort_values("spot_shock")["pnl"].to_numpy()
            assert np.all(np.diff(pnl) * sign > 0)
        # Both gain when volatility rises (holding spot fixed)
        for _, group in table.groupby("spot_shock"):
            pnl = group.sort_values("vol_shock")["pnl"].to_numpy()
            assert np.all(np.diff(pnl) > 0)


def test_impossible_shock_is_rejected():
    with pytest.raises(ValueError):
        A.stress_test("call", vol_shocks=(-0.35,))  # would make volatility zero
    with pytest.raises(ValueError):
        A.stress_test("straddle")


def test_stress_pnl_grid_labels():
    grid = stress_pnl_grid(A.stress_test("call"))
    assert grid.shape == (5, 4)
    assert grid.index[0] == "-20%" and grid.index[-1] == "+20%"
    assert grid.columns[0] == "-10 pts" and grid.columns[-1] == "+20 pts"


def test_price_surface_shape_and_centre():
    surface = A.price_surface("call")
    assert surface.shape == (7, 9)  # 7 volatilities (rows) x 9 spots (columns)
    centre = surface.iloc[3, 4]  # the middle cell is today's market
    assert centre == pytest.approx(BlackScholesEngine(M).price("call"), abs=1e-12)
    # Call price rises with spot along every row
    assert np.all(np.diff(surface.to_numpy(), axis=1) > 0)


def test_price_surface_rejects_bad_grid():
    with pytest.raises(ValueError):
        A.price_surface("call", n_spot=1)
    with pytest.raises(ValueError):
        A.price_surface("call", vol_range=(-0.5, 1.5))


def test_plots_build_and_save():
    folder = tempfile.mkdtemp()
    for name, make_figure in (("stress.png", A.plot_stress_test), ("surface.png", A.plot_price_surface)):
        path = os.path.join(folder, name)
        fig = make_figure(save_path=path)
        assert len(fig.axes) == 4  # two heatmaps plus their two colourbars
        assert os.path.getsize(path) > 10_000
        plt.close(fig)
