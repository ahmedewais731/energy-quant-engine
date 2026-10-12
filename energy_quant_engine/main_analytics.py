"""Analytics layer: Greeks table, Monte Carlo convergence plots, and the runnable script.

Run the example from the repository root (the folder CONTAINING energy_quant_engine/):

    python -m energy_quant_engine.main_analytics
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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


def stress_pnl_grid(table: pd.DataFrame) -> pd.DataFrame:
    """Reshape ScenarioAnalytics.stress_test output into a labelled P&L grid.

    Rows are spot shocks (e.g. '-20%'), columns are volatility shocks in
    percentage points (e.g. '+10 pts'), and each cell is the P&L vs today.
    """
    grid = table.pivot(index="spot_shock", columns="vol_shock", values="pnl")
    grid.index = [f"{x:+.0%}" for x in grid.index]
    grid.columns = [f"{x * 100:+.0f} pts" for x in grid.columns]
    return grid


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

    def stress_test(
        self,
        option_type: str = "call",
        spot_shocks: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20),
        vol_shocks: tuple[float, ...] = (-0.10, 0.0, 0.10, 0.20),
    ) -> pd.DataFrame:
        """Re-price a European option after shocking spot and volatility.

        spot_shocks are RELATIVE moves (-0.20 = spot falls 20%).
        vol_shocks are ABSOLUTE moves in volatility (+0.10 = ten percentage
        points, so 35% vol becomes 45%).

        Returns one row per (spot shock, vol shock) pair with columns:
        spot_shock, vol_shock, spot, sigma, price, pnl (price minus today's price).
        """
        _check_option_type(option_type)
        base_price = self._bs.price(option_type)
        m = self.market

        rows = []
        for spot_shock in spot_shocks:
            for vol_shock in vol_shocks:
                # replace() makes a modified COPY of the frozen MarketData
                # (and re-runs its validation, so impossible shocks raise errors).
                shocked = replace(m, S=m.S * (1 + spot_shock), sigma=m.sigma + vol_shock)
                price = BlackScholesEngine(shocked).price(option_type)
                rows.append(
                    {
                        "spot_shock": spot_shock,
                        "vol_shock": vol_shock,
                        "spot": shocked.S,
                        "sigma": shocked.sigma,
                        "price": price,
                        "pnl": price - base_price,
                    }
                )
        return pd.DataFrame(rows)

    def price_surface(
        self,
        option_type: str = "call",
        spot_range: tuple[float, float] = (0.7, 1.3),
        vol_range: tuple[float, float] = (0.5, 1.5),
        n_spot: int = 9,
        n_vol: int = 7,
    ) -> pd.DataFrame:
        """European option price over a grid of spot and volatility values.

        spot_range and vol_range are MULTIPLES of today's spot and volatility
        (0.7 to 1.3 means 70% to 130% of today's value).

        Returns a DataFrame: rows are volatilities, columns are spot prices.
        """
        _check_option_type(option_type)
        if n_spot < 2 or n_vol < 2:
            raise ValueError("n_spot and n_vol must each be at least 2")
        m = self.market

        spots = np.linspace(*spot_range, n_spot) * m.S
        vols = np.linspace(*vol_range, n_vol) * m.sigma
        grid = [
            [BlackScholesEngine(replace(m, S=s, sigma=v)).price(option_type) for s in spots]
            for v in vols
        ]
        surface = pd.DataFrame(grid, index=vols, columns=spots)
        surface.index.name = "sigma"
        surface.columns.name = "spot"
        return surface

    def plot_stress_test(self, save_path: str | None = None) -> Figure:
        """Heatmaps of call and put P&L under the default spot/volatility shocks."""
        with sns.axes_style("white"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, option_type in zip(axes, ("call", "put")):
            grid = stress_pnl_grid(self.stress_test(option_type))
            sns.heatmap(
                grid, annot=True, fmt="+.2f", cmap="RdYlGn", center=0, ax=ax,
                cbar_kws={"label": "P&L vs today's price"},
            )
            ax.set_title(f"{self.market.asset_name} European {option_type}: P&L under stress")
            ax.set_xlabel("Volatility shock")
            ax.set_ylabel("Spot shock")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_price_surface(self, save_path: str | None = None) -> Figure:
        """Heatmaps of call and put price across spot and volatility.

        The cell outlined in black is today's market (current spot and volatility).
        """
        m = self.market
        with sns.axes_style("white"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, option_type in zip(axes, ("call", "put")):
            surface = self.price_surface(option_type).iloc[::-1]  # high vol at the top
            col = int(np.argmin(np.abs(surface.columns.to_numpy() - m.S)))
            row = int(np.argmin(np.abs(surface.index.to_numpy() - m.sigma)))

            labelled = surface.copy()
            labelled.index = [f"{v:.0%}" for v in surface.index]
            labelled.columns = [f"{s:.0f}" for s in surface.columns]
            sns.heatmap(
                labelled, annot=True, fmt=".1f", cmap="viridis", ax=ax,
                cbar_kws={"label": "Option price"},
            )
            ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
            ax.set_title(f"{m.asset_name} European {option_type}: price by spot and volatility")
            ax.set_xlabel("Spot price")
            ax.set_ylabel("Volatility")

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

    print("\nStress test: change in European option price vs today")
    print("(rows = spot shock, columns = volatility shock)")
    for option_type in ("call", "put"):
        grid = stress_pnl_grid(analytics.stress_test(option_type))
        print(f"\n  {option_type}:")
        print(grid.to_string(float_format=lambda x: f"{x:+8.2f}"))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {out_path}/ ...")
    for option_type in ("call", "put"):
        file = out_path / f"mc_convergence_{option_type}.png"
        fig = analytics.plot_mc_convergence(option_type, n_sims=n_sims, save_path=str(file))
        plt.close(fig)  # free the figure's memory
        print(f"  saved {file}")

    for name, make_figure in (
        ("stress_test.png", analytics.plot_stress_test),
        ("price_surface.png", analytics.plot_price_surface),
    ):
        file = out_path / name
        plt.close(make_figure(save_path=str(file)))
        print(f"  saved {file}")


if __name__ == "__main__":
    main()
