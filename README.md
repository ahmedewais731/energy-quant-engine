# energy-quant-engine

An object-oriented Python package for pricing and risk-analysing energy (commodity) options. It prices **European** options with the analytical Black-Scholes formulas and **Asian (average-price)** options with a vectorized Monte Carlo simulation, then reports Greeks, stress tests and price surfaces.

The worked example is a BRENT_CRUDE option (spot 80, strike 85, six months to expiry, 4% rate, 35% volatility).

## Features

- **Black-Scholes engine** for European calls and puts, with analytical Greeks (delta, gamma, vega, theta)
- **Monte Carlo engine** that simulates Geometric Brownian Motion price paths and prices arithmetic-average Asian options, returning the price and its standard error
- **Analytics**: Greeks table, Monte Carlo convergence plot, spot/volatility stress test, and a price surface
- **Validated inputs**: bad market data (negative prices, NaN, wrong types) is rejected with clear errors
- **35 automated tests**, including cross-checks between the two engines

## Installation

```bash
git clone https://github.com/<your-username>/energy-quant-engine.git
cd energy-quant-engine
pip install -r requirements.txt
```

Requires Python 3.10 or newer.

## Run the example

From the repository root:

```bash
python -m energy_quant_engine.main_analytics
```

This prints the results below and saves four plots to an `output/` folder.

```
=== BRENT_CRUDE ===
S=80  K=85  T=0.5 years  r=4.00%  sigma=35.00%

European option: Black-Scholes price and Greeks
                 price      delta      gamma       vega      theta
call            6.4910     0.4839     0.0201    22.5492    -9.1810
put             9.8078    -0.5161     0.0201    22.5492    -5.8483

Asian option (arithmetic average): Monte Carlo, 100,000 paths
  call  price = 2.9136 +/- 0.0191   (European call: 6.4910)
  put   price = 6.9929 +/- 0.0234   (European put: 9.8078)
```

The Asian options are cheaper than the European ones because averaging the price over the option's life dampens its volatility.

## Using the package

```python
from energy_quant_engine import MarketData, BlackScholesEngine, MonteCarloEngine

market = MarketData(S=80.0, K=85.0, T=0.5, r=0.04, sigma=0.35, asset_name="BRENT_CRUDE")

# European option, closed form
bs = BlackScholesEngine(market)
bs.price("call")     # 6.4910
bs.greeks("put")     # {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ...}

# Asian option, Monte Carlo (seed makes the result repeatable)
mc = MonteCarloEngine(market, seed=42)
price, std_error = mc.price_asian_option("call", n_sims=100_000, n_steps=252)
```

The analytics layer lives in `main_analytics.py`:

```python
from energy_quant_engine.main_analytics import ScenarioAnalytics

analytics = ScenarioAnalytics(market, seed=42)
analytics.generate_greeks_table()               # pandas DataFrame
analytics.stress_test("call")                   # spot and volatility shocks
analytics.price_surface("put")                  # price over a spot x volatility grid
analytics.plot_mc_convergence("call", save_path="convergence.png")
```

## Project structure

```
energy-quant-engine/
├── energy_quant_engine/
│   ├── market_data.py       # MarketData: validated, immutable inputs (S, K, T, r, sigma, asset_name)
│   ├── pricing_engines.py   # BlackScholesEngine and MonteCarloEngine
│   └── main_analytics.py    # ScenarioAnalytics, plots, and the runnable example
├── tests/                   # pytest suite
├── requirements.txt
└── README.md
```

The code is split into three layers: **Contract** (what a market looks like), **Engine** (how to price it) and **Analytics** (what to do with the prices).

## The maths

**Black-Scholes (European).** With `d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)` and `d2 = d1 − σ√T`:

```
Call = S·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·N(−d1)
```

**Monte Carlo (Asian).** Price paths follow risk-neutral Geometric Brownian Motion, one step at a time:

```
S(t+Δt) = S(t) · exp[(r − σ²/2)Δt + σ√Δt · Z],   Z ~ N(0, 1)
```

The payoff of an arithmetic-average Asian call is `max(average of S over the monitoring dates − K, 0)`. The price is the discounted mean payoff across all paths, and the standard error is the standard deviation of the discounted payoffs divided by √(number of paths).

## Conventions

- `T` is in years; `r` and `sigma` are annualised decimals (0.05 means 5%)
- `option_type` is the lowercase string `'call'` or `'put'`
- Vega is per 1.00 change in volatility; theta is per year and negative for a long option
- The Asian average uses the monitoring dates after time 0 (the starting price is excluded)

## Testing

```bash
python -m pytest
```

Besides unit checks, the suite verifies the two engines against each other and against theory:

- Black-Scholes matches a published benchmark and satisfies put-call parity
- Analytical Greeks match finite-difference estimates
- Simulated terminal prices have the correct mean and log-return distribution
- A Monte Carlo European price agrees with Black-Scholes within its standard error
- The simulated paths price a geometric-average Asian option to match its exact closed-form value

## Limitations

- Black-Scholes has no cost of carry or convenience yield. For commodities, the Black-76 model on a forward price is usually the better fit.
- Volatility is constant, so the "price surface" varies volatility by hand rather than being an implied-volatility surface.
- Asian averaging is discrete (daily by default), and the strike is fixed.
- The Monte Carlo has no variance reduction, so its error shrinks only as 1/√N.
