"""Phase 2 checks for BlackScholesEngine. Run from the repo root: python -m pytest"""

import dataclasses
import math

import pytest

from energy_quant_engine.market_data import MarketData
from energy_quant_engine.pricing_engines import BlackScholesEngine

M = MarketData(S=100, K=100, T=1.0, r=0.05, sigma=0.20, asset_name="TEST")


def price(market, option_type="call"):
    return BlackScholesEngine(market).price(option_type)


def test_known_textbook_values():
    # Widely quoted benchmark: S=K=100, T=1, r=5%, sigma=20%
    assert price(M, "call") == pytest.approx(10.4506, abs=1e-4)
    assert price(M, "put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    # C - P must equal S - K*exp(-rT) for any inputs
    for market in [M, dataclasses.replace(M, K=90, sigma=0.4), dataclasses.replace(M, T=0.25, r=-0.01)]:
        lhs = price(market, "call") - price(market, "put")
        rhs = market.S - market.K * math.exp(-market.r * market.T)
        assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_differences(option_type):
    g = BlackScholesEngine(M).greeks(option_type)
    h = 1e-4
    up = lambda **kw: price(dataclasses.replace(M, **kw), option_type)

    delta = (up(S=M.S + h) - up(S=M.S - h)) / (2 * h)
    gamma = (up(S=M.S + h) - 2 * price(M, option_type) + up(S=M.S - h)) / h**2
    vega = (up(sigma=M.sigma + h) - up(sigma=M.sigma - h)) / (2 * h)
    theta = -(up(T=M.T + h) - up(T=M.T - h)) / (2 * h)  # minus: theta is w.r.t. time passing

    assert g["delta"] == pytest.approx(delta, abs=1e-6)
    assert g["gamma"] == pytest.approx(gamma, abs=1e-5)
    assert g["vega"] == pytest.approx(vega, abs=1e-5)
    assert g["theta"] == pytest.approx(theta, abs=1e-5)


def test_rejects_bad_option_type():
    with pytest.raises(ValueError):
        BlackScholesEngine(M).price("straddle")
    with pytest.raises(ValueError):
        BlackScholesEngine(M).greeks("Call")
