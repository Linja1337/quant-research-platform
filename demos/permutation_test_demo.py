"""
Permutation test demo.

What this shows
---------------
Two strategies, side by side. One has a real edge baked into the synthetic
data; the other is a random-PnL placeholder with no edge. Both are tested
with day-block permutation: shuffle the price-day order, re-run, repeat 500
times, build the null distribution.

The strategy with edge sits in the right tail of its null distribution
(p-value near zero). The placebo sits in the middle (p-value near 0.5).
That contrast is the visual signature of a working permutation test.

Run
---
    python demos/permutation_test_demo.py

Output
------
demos/output/permutation.png
Console: observed Sharpe and p-value for both strategies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _synthetic import StrategyConfig, generate_ohlc

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BARS = 6_000
N_PERMUTATIONS = 500
BARS_PER_DAY = 78


def per_bar_sharpe(per_bar_pnl: np.ndarray) -> float:
    if per_bar_pnl.std() < 1e-12:
        return 0.0
    return float(per_bar_pnl.mean() / per_bar_pnl.std())


def day_shuffled_close(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the order of trading days while preserving intraday bar order."""
    n_days = len(close) // BARS_PER_DAY
    usable = n_days * BARS_PER_DAY
    truncated = close[:usable]
    days = truncated.reshape(n_days, BARS_PER_DAY)
    indices = np.arange(n_days)
    rng.shuffle(indices)
    shuffled = days[indices]
    # Re-stitch so prices are continuous: each new "day" starts at the
    # previous day's close, preserving the within-day shape.
    out = shuffled.copy()
    for i in range(1, n_days):
        out[i] = out[i] + (out[i - 1, -1] - out[i, 0])
    return out.flatten()


def edge_strategy_pnl(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """SMA crossover. Captures regime drift when present in the data."""
    cfg = StrategyConfig(fast=fast, slow=slow, kind="sma")
    pnl, _ = cfg.trades(close)
    return pnl


def placebo_strategy_pnl(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bar-return-magnitude-matched random PnL. Has no signal but reacts to
    the input enough that day-block permutations actually move the result."""
    bar_returns = np.zeros(len(close))
    bar_returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    scale = max(float(np.std(bar_returns)), 1e-9)
    return rng.normal(loc=0.0, scale=scale * 0.5, size=len(close))


def permutation_p_value(
    pnl_fn,
    close: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Run a permutation test. pnl_fn(close) returns the per-bar PnL."""
    observed = per_bar_sharpe(pnl_fn(close))

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = day_shuffled_close(close, rng)
        null[i] = per_bar_sharpe(pnl_fn(shuffled))

    p_value = float(np.mean(null >= observed))
    return observed, p_value, null


def plot(
    edge_observed: float,
    edge_null: np.ndarray,
    edge_p: float,
    placebo_observed: float,
    placebo_null: np.ndarray,
    placebo_p: float,
    out_path: Path,
) -> None:
    plt.style.use("default")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.4),
                                              gridspec_kw={"wspace": 0.32})
    fig.patch.set_facecolor("#fafafa")

    text_color = "#111827"
    accent_pass = "#2563eb"
    accent_fail = "#b45309"
    accent_line = "#dc2626"

    for ax, observed, null, p_value, color, label in (
        (ax_left, edge_observed, edge_null, edge_p, accent_pass, "Strategy with real edge"),
        (ax_right, placebo_observed, placebo_null, placebo_p, accent_fail, "Placebo (random PnL)"),
    ):
        ax.hist(null, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.axvline(observed, color=accent_line, linewidth=1.6, linestyle="--",
                   label=f"observed Sharpe = {observed:.2f}")
        ax.set_xlabel("per-bar Sharpe under day-shuffled null", color=text_color)
        ax.set_ylabel("number of permutations", color=text_color)
        verdict = ("reject the null (p < 0.01)"
                   if p_value < 0.01 else
                   ("reject the null (p < 0.05)" if p_value < 0.05 else "fail to reject"))
        ax.set_title(
            f"{label}\np-value = {p_value:.3f}    {verdict}",
            loc="left", color=text_color, fontsize=11, weight="bold", pad=8,
        )
        ax.legend(frameon=False, loc="upper right")
        _style(ax, text_color)

    fig.suptitle(
        "Permutation test   |   day-block shuffling preserves intraday autocorrelation, destroys inter-day temporal structure",
        x=0.06, y=0.99, ha="left", color=text_color, fontsize=12, weight="bold",
    )
    fig.text(
        0.06, 0.012,
        f"{N_PERMUTATIONS} permutations per strategy. Same OHLC, same engine, same SMA crossover. "
        "Only the PnL-generating mechanism differs: real entries vs random PnL.",
        fontsize=8, color="#6b7280", ha="left",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _style(ax, text_color: str) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d1d5db")
    ax.tick_params(colors=text_color, length=4)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    print("Generating synthetic OHLC...")
    bars = generate_ohlc(n_bars=N_BARS, seed=42)
    close = bars["close"]

    print(f"Running {N_PERMUTATIONS} permutations on the edge strategy...")
    edge_obs, edge_p, edge_null = permutation_p_value(
        lambda c: edge_strategy_pnl(c, fast=8, slow=30),
        close, N_PERMUTATIONS, seed=11,
    )

    print(f"Running {N_PERMUTATIONS} permutations on the placebo...")
    placebo_rng = np.random.default_rng(99)
    placebo_obs, placebo_p, placebo_null = permutation_p_value(
        lambda c: placebo_strategy_pnl(c, placebo_rng),
        close, N_PERMUTATIONS, seed=11,
    )

    out_path = OUT_DIR / "permutation.png"
    plot(edge_obs, edge_null, edge_p, placebo_obs, placebo_null, placebo_p, out_path)

    print()
    print("Result")
    print("------")
    print(f"  Edge strategy   observed Sharpe {edge_obs:>6.2f}   p-value {edge_p:.3f}")
    print(f"  Placebo         observed Sharpe {placebo_obs:>6.2f}   p-value {placebo_p:.3f}")
    print()
    print(f"  Chart written to {out_path.relative_to(out_path.parents[2])}")


if __name__ == "__main__":
    main()
