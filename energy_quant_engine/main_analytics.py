"""Analytics layer: Greeks table, Monte Carlo convergence plots, and the runnable script.

Run the example from the repository root (the folder CONTAINING energy_quant_engine/):

    python -m energy_quant_engine.main_analytics
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from energy_quant_engine.market_data import MarketData
from energy_quant_engine.pricing_engines import (
    BlackScholesEngine,
    MonteCarloEngine,
    _check_option_type,
)


def running_estimates(values: np.ndarray, n_points: int = 200) -> pd.DataFrame:
    """Running mean and standard error of `values` as more samples are included.

    Row k answers: "if I had only used the first n_sims samples, what would my
    estimate and its standard error have been?" The n_sims values are spaced
    logarithmically, ending exactly at len(values).

    Returns a DataFrame with columns: n_sims, price, std_error.
    """
    values = np.asarray(values, dtype=float)
    n_total = len(values)
    if n_total < 2:
        raise ValueError("need at least 2 values to estimate a standard error")

    counts = np.geomspace(2, n_total, n_points).astype(int)
    counts = np.unique(np.append(counts, n_total))

    # Cumulative sums let us get the mean and variance of the first n samples
    # for every n at once, without re-averaging from scratch each time.
    cum_sum = np.cumsum(values)
    cum_sum_sq = np.cumsum(values**2)

    mean = cum_sum[counts - 1] / counts
    variance = (cum_sum_sq[counts - 1] - counts * mean**2) / (counts - 1)
    variance = np.maximum(variance, 0.0)  # guard against tiny negative rounding errors
    std_error = np.sqrt(variance / counts)

    return pd.DataFrame({"n_sims": counts, "price": mean, "std_error": std_error})


class ScenarioAnalytics:
    """Builds the summary table and plots for one market scenario."""

    def __init__(self, market: MarketData, seed: int | None = None) -> None:
        self.market = market
        self._bs = BlackScholesEngine(market)
        self._mc = MonteCarloEngine(market, seed=seed)

    def generate_greeks_table(self) -> pd.DataFrame:
        """Black-Scholes price and Greeks for the call and the put.

        One row per option type; columns: price, delta, gamma, vega, theta.
        """
        rows = {}
        for option_type in ("call", "put"):
            rows[option_type] = {
                "price": self._bs.price(option_type),
                **self._bs.greeks(option_type),
            }
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "option_type"
        return table

    def plot_mc_convergence(
        self,
        option_type: str = "call",
        n_sims: int = 100_000,
        n_steps: int = 252,
        n_paths_shown: int = 100,
        save_path: str | None = None,
    ) -> Figure:
        """Two-panel figure: simulated price paths, and Asian price convergence.

        Left  : the first `n_paths_shown` simulated paths, with the strike marked.
        Right : the Monte Carlo Asian price estimate as the number of
                simulations grows to n_sims, with a 95% confidence band.

        If save_path is given the figure is also written to that file.
        """
        _check_option_type(option_type)
        m = self.market

        paths = self._mc.simulate_paths(n_sims, n_steps)
        payoffs = self._mc.asian_payoffs(paths, option_type)
        estimates = running_estimates(payoffs)

        times = np.linspace(0, m.T, n_steps + 1)  # x-axis for the path plot, in years
        shown = paths[: min(n_paths_shown, n_sims)]

        with sns.axes_style("whitegrid"):
            fig, (ax_paths, ax_conv) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: sample paths
        ax_paths.plot(times, shown.T, linewidth=0.7, alpha=0.5)
        ax_paths.axhline(m.K, color="black", linestyle="--", label=f"Strike K = {m.K:g}")
        ax_paths.set_title(f"{m.asset_name}: {len(shown)} simulated price paths")
        ax_paths.set_xlabel("Time (years)")
        ax_paths.set_ylabel("Price")
        ax_paths.legend()

        # Right panel: convergence of the Asian price estimate
        n = estimates["n_sims"]
        price = estimates["price"]
        half_width = 1.96 * estimates["std_error"]
        ax_conv.plot(n, price, color="tab:blue", label=f"Asian {option_type} estimate")
        ax_conv.fill_between(
            n, price - half_width, price + half_width,
            color="tab:blue", alpha=0.25, label="95% confidence band",
        )
        ax_conv.axhline(
            price.iloc[-1], color="black", linestyle="--",
            label=f"Final estimate = {price.iloc[-1]:.3f}",
        )
        ax_conv.set_xscale("log")
        ax_conv.set_title(f"Asian {option_type} price convergence")
        ax_conv.set_xlabel("Number of simulations (log scale)")
        ax_conv.set_ylabel("Estimated price")
        ax_conv.legend()

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


def main(output_dir: str = "output", n_sims: int = 100_000) -> None:
    """Run the BRENT_CRUDE example: print the results and save the plots."""
    market = MarketData(S=80.0, K=85.0, T=0.5, r=0.04, sigma=0.35, asset_name="BRENT_CRUDE")
    analytics = ScenarioAnalytics(market, seed=42)  # fixed seed -> repeatable results
    black_scholes = BlackScholesEngine(market)
    monte_carlo = MonteCarloEngine(market, seed=42)

    print(f"=== {market.asset_name} ===")
    print(
        f"S={market.S:g}  K={market.K:g}  T={market.T:g} years  "
        f"r={market.r:.2%}  sigma={market.sigma:.2%}"
    )

    print("\nEuropean option: Black-Scholes price and Greeks")
    table = analytics.generate_greeks_table()
    print(table.to_string(float_format=lambda x: f"{x:10.4f}"))
    print("(vega is per 1.00 change in sigma; theta is per year)")

    print(f"\nAsian option (arithmetic average): Monte Carlo, {n_sims:,} paths")
    for option_type in ("call", "put"):
        price, std_error = monte_carlo.price_asian_option(option_type, n_sims=n_sims)
        european = black_scholes.price(option_type)
        print(
            f"  {option_type:<4}  price = {price:.4f} +/- {std_error:.4f}"
            f"   (European {option_type}: {european:.4f})"
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {out_path}/ ...")
    for option_type in ("call", "put"):
        file = out_path / f"mc_convergence_{option_type}.png"
        fig = analytics.plot_mc_convergence(option_type, n_sims=n_sims, save_path=str(file))
        plt.close(fig)  # free the figure's memory
        print(f"  saved {file}")


if __name__ == "__main__":
    main()
