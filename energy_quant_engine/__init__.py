"""energy_quant_engine: option pricing and risk for energy/commodity contracts.

Package layout (Contract / Engine / Analytics):

    market_data.py      -> Contract layer: the inputs describing a market (S, K, T, r, sigma)
    pricing_engines.py  -> Engine layer: Black-Scholes and Monte Carlo pricers
    main_analytics.py   -> Analytics layer: Greeks table, plots, and the runnable script

Project-wide conventions (every module follows these):

    * T is time to expiry in YEARS (6 months = 0.5).
    * r and sigma are ANNUALISED DECIMALS (5% = 0.05, 30% vol = 0.30), never percentages.
    * r is a continuously compounded risk-free rate (discount factor = exp(-r * T)).
    * option_type is the lowercase string 'call' or 'put'.
    * Theta is reported per YEAR in the conventional sign (negative for a long option).
"""

from energy_quant_engine.market_data import MarketData

__all__ = ["MarketData"]
