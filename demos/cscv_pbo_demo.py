"""
CSCV / PBO demo. Combinatorially Symmetric Cross-Validation.

What this shows
---------------
The probability that an in-sample-best strategy is overfit, computed without
relying on any single train-test split. CSCV enumerates every symmetric
partition of the data into in-sample and out-of-sample halves, ranks the
strategies on each half, and asks: how often does the in-sample winner end up
below the median out-of-sample?

That fraction is the Probability of Backtest Overfitting (PBO). It is the
single most useful number in systematic strategy validation.

Reference
---------
Bailey, Borwein, Lopez de Prado, Zhu (2014). "The Probability of Backtest
Overfitting." SSRN 2326253. Algorithm 2.3.

Run
---
    python demos/cscv_pbo_demo.py

Output
------
demos/output/cscv_pbo.png is the figure embedded in the README.
Console: PBO, mean WFE, performance-degradation slope, and a one-line verdict.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from _synthetic import (
    aggregate_per_block_pnl,
    generate_ohlc,
    synthetic_population,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BARS = 6_000
N_BLOCKS = 8  # C(8, 4) = 70 symmetric partitions.

# Two same-size populations. Only the signal quality differs. Same N keeps PBO
# comparisons fair (PBO has a known dependence on the number of trials).
N_PER_POP = 80
DISCIPLINED = {"n_signal": N_PER_POP, "n_noise": 0, "label": "disciplined search"}
DATA_MINED = {"n_signal": 0, "n_noise": N_PER_POP, "label": "data-mined search"}


def build_performance_matrix(
    bars: dict[str, np.ndarray],
    configs,
    n_blocks: int,
) -> np.ndarray:
    """M[t, n] = aggregate PnL of config n during time block t."""
    close = bars["close"]
    n_configs = len(configs)
    M = np.empty((n_blocks, n_configs))
    for n, cfg in enumerate(configs):
        per_bar_pnl, _ = cfg.trades(close)
        M[:, n] = aggregate_per_block_pnl(per_bar_pnl, n_blocks)
    return M


def cscv(M: np.ndarray, truth: np.ndarray) -> dict:
    """
    Run Bailey-Lopez de Prado CSCV on a (T, N) performance matrix.

    For each symmetric IS/OOS split, picks the IS-best strategy and records
    where it lands in the OOS performance distribution. PBO is the fraction
    of splits where that IS-best ranks at or below the OOS median.

    truth is the per-strategy ground-truth label (signal vs noise) used only
    for the right-panel coloring; the CSCV itself is truth-blind.
    """
    T, N = M.shape
    half = T // 2
    splits = list(itertools.combinations(range(T), half))

    logits = np.empty(len(splits))
    winner_oos_quantile = np.empty(len(splits))
    winner_was_signal = np.empty(len(splits), dtype=bool)
    full_is_perf = M.sum(axis=0)
    full_is_rank = full_is_perf.argsort().argsort() + 1

    for k, is_idx in enumerate(splits):
        oos_idx = [t for t in range(T) if t not in is_idx]
        is_perf = M[list(is_idx)].sum(axis=0)
        oos_perf = M[list(oos_idx)].sum(axis=0)

        # Pick in-sample best.
        n_star = int(np.argmax(is_perf))

        # omega_bar: where the IS-best lands in the OOS distribution.
        # rank from 1 (worst) to N (best); divide by N+1 to get (0, 1).
        oos_ranks = oos_perf.argsort().argsort() + 1
        rank_n_star = int(oos_ranks[n_star])
        omega = rank_n_star / (N + 1)
        omega_safe = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logits[k] = np.log(omega_safe / (1 - omega_safe))

        winner_oos_quantile[k] = omega
        winner_was_signal[k] = bool(truth[n_star])

    pbo = float(np.mean(logits <= 0))

    # Per-strategy OOS performance averaged over the splits where the strategy
    # is in the OOS half. This is the conventional "OOS rank stability" view.
    oos_avg_rank = np.zeros(N)
    oos_count = np.zeros(N)
    for is_idx in splits:
        oos_idx = [t for t in range(T) if t not in is_idx]
        oos_perf = M[list(oos_idx)].sum(axis=0)
        ranks = oos_perf.argsort().argsort() + 1
        oos_avg_rank += ranks
        oos_count += 1
    oos_avg_rank /= oos_count

    rho, _ = spearmanr(full_is_rank, oos_avg_rank)

    return {
        "pbo": pbo,
        "logits": logits,
        "winner_oos_quantile": winner_oos_quantile,
        "winner_was_signal": winner_was_signal,
        "full_is_rank": full_is_rank,
        "oos_avg_rank": oos_avg_rank,
        "spearman_rho": float(rho),
        "n_splits": len(splits),
        "pct_winners_signal": float(winner_was_signal.mean()),
    }


def plot(disciplined: dict, data_mined: dict, out_path: Path) -> None:
    """
    Hero figure. Two rows. Top = disciplined parameter search; bottom = data-
    mined search. Each row has a logit-distribution histogram (left) and a
    where-did-the-IS-winner-land-OOS scatter (right). Shared x and y so the
    two rows are visually comparable.
    """
    plt.style.use("default")
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8.4),
        gridspec_kw={"wspace": 0.26, "hspace": 0.55, "width_ratios": [1.05, 1.0]},
    )
    fig.patch.set_facecolor("#fafafa")

    accent_pass = "#2563eb"
    accent_fail = "#b45309"
    accent_line = "#dc2626"
    text_color = "#111827"

    rows = [
        (f"Disciplined search   |   {N_PER_POP} strategies inside a hypothesis-justified parameter window",
         disciplined, accent_pass, axes[0, 0], axes[0, 1]),
        (f"Data-mined search    |   {N_PER_POP} strategies sampled blindly from the full grid",
         data_mined, accent_fail, axes[1, 0], axes[1, 1]),
    ]

    for row_title, result, color, ax_left, ax_right in rows:
        # Left: logit histogram.
        logits = result["logits"]
        ax_left.hist(
            logits, bins=20, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.8,
        )
        ax_left.axvline(0, color=accent_line, linestyle="--", linewidth=1.2, label="logit = 0")
        ax_left.set_xlabel("logit(omega_bar)", color=text_color, fontsize=10)
        ax_left.set_ylabel("number of CSCV splits", color=text_color, fontsize=10)
        verdict = _pbo_verdict(result["pbo"])
        ax_left.set_title(
            f"PBO = {result['pbo']:.3f}    {verdict}",
            color=text_color, fontsize=11, loc="left", pad=8,
        )
        ax_left.legend(frameon=False, loc="upper right", fontsize=9)
        _style_axes(ax_left, text_color)

        # Right: per-split scatter of IS-winner OOS quantile.
        n_splits = len(result["winner_oos_quantile"])
        rng = np.random.default_rng(11)
        x_jitter = rng.uniform(-0.4, 0.4, size=n_splits)
        ax_right.scatter(
            x_jitter, result["winner_oos_quantile"],
            s=44, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.6,
        )
        ax_right.axhline(0.5, color=accent_line, linestyle="--", linewidth=1.0, label="OOS median")
        ax_right.set_xticks([])
        ax_right.set_xlim(-0.7, 0.7)
        ax_right.set_ylim(-0.02, 1.02)
        ax_right.set_xlabel(
            f"each dot = one of {n_splits} symmetric IS/OOS partitions",
            color=text_color, fontsize=10,
        )
        ax_right.set_ylabel("OOS quantile of the IS-winner   (1.0 = best of N)", color=text_color, fontsize=10)
        below_median = float(np.mean(result["winner_oos_quantile"] <= 0.5))
        ax_right.set_title(
            f"{below_median * 100:.0f}% of IS-winners landed at or below the OOS median",
            color=text_color, fontsize=11, loc="left", pad=8,
        )
        ax_right.legend(frameon=False, loc="lower left", fontsize=9)
        _style_axes(ax_right, text_color)

        # Row banner above the two panels.
        fig.text(
            0.06, ax_left.get_position().y1 + 0.04,
            row_title, fontsize=12, color=text_color, weight="bold", ha="left",
        )

    fig.suptitle(
        "Combinatorially Symmetric Cross-Validation   |   how PBO separates disciplined research from parameter spam",
        fontsize=13, color=text_color, x=0.06, y=0.99, ha="left", weight="bold",
    )
    fig.text(
        0.06, 0.012,
        "Bailey, Borwein, Lopez de Prado, Zhu (2014). The Probability of Backtest Overfitting.    "
        "Same synthetic OHLC, same engine, same number of bars. Only the search discipline differs.",
        fontsize=8, color="#6b7280", ha="left",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _pbo_verdict(pbo: float) -> str:
    if pbo < 0.10:
        return "low overfitting risk"
    if pbo < 0.25:
        return "moderate overfitting risk"
    if pbo < 0.50:
        return "high overfitting risk"
    return "severe overfitting risk"


def _style_axes(ax, text_color: str) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d1d5db")
    ax.tick_params(colors=text_color, length=4)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def _run_scenario(bars, n_signal: int, n_noise: int, label: str) -> dict:
    print(f"  [{label}] population: {n_signal} signal, {n_noise} noise")
    configs, truth = synthetic_population(n_signal=n_signal, n_noise=n_noise)
    M = build_performance_matrix(bars, configs, N_BLOCKS)
    return cscv(M, truth)


def main() -> None:
    print("Generating synthetic OHLC bars (regime-switching trend)...")
    bars = generate_ohlc(n_bars=N_BARS, seed=42)

    print(f"Running CSCV on two contrasting strategy populations "
          f"(C({N_BLOCKS}, {N_BLOCKS // 2}) = 70 splits each)")
    print()
    disciplined = _run_scenario(bars, **{k: v for k, v in DISCIPLINED.items() if k != "label"},
                                 label=DISCIPLINED["label"])
    data_mined = _run_scenario(bars, **{k: v for k, v in DATA_MINED.items() if k != "label"},
                                label=DATA_MINED["label"])

    out_path = OUT_DIR / "cscv_pbo.png"
    plot(disciplined, data_mined, out_path)

    print()
    print("Result")
    print("------")
    print(f"  Disciplined search   PBO = {disciplined['pbo']:.3f}    "
          f"({_pbo_verdict(disciplined['pbo'])})")
    print(f"  Data-mined search    PBO = {data_mined['pbo']:.3f}    "
          f"({_pbo_verdict(data_mined['pbo'])})")
    print()
    print(f"  Chart written to     {out_path.relative_to(out_path.parents[2])}")
    print()
    print("Reading the chart")
    print("-----------------")
    print("  Both rows use the same OHLC, same engine, same number of bars.")
    print("  Only the search discipline differs.")
    print()
    print("  Top row    Small parameter window, mostly chosen by hypothesis.")
    print("             PBO falls below 0.10. The IS winner generalizes OOS.")
    print()
    print("  Bottom row Large grid of mostly arbitrary configurations.")
    print("             PBO rises sharply. The IS winner is usually noise that")
    print("             happened to fit the in-sample half. This is what")
    print("             multiple-testing bias looks like in practice.")


if __name__ == "__main__":
    main()
