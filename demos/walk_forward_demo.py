"""
Walk-forward validation demo.

What this shows
---------------
A strategy that looks profitable when you tune its parameters on the same
data you score it on, and what happens when you score it on data that did
not exist when the parameters were chosen. The gap between the two is the
in-sample-to-out-of-sample (IS-to-OOS) degradation that walk-forward exists
to expose.

Run
---
    python demos/walk_forward_demo.py

Output
------
demos/output/walk_forward.png
Console: per-fold IS and OOS Sharpe, walk-forward efficiency per fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _synthetic import StrategyConfig, generate_ohlc

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BARS = 8_000
N_FOLDS = 6
IS_FRACTION = 0.6  # IS window is this fraction of each fold's bars.

FAST_GRID = list(range(3, 13))
SLOW_GRID = list(range(15, 51, 5))


@dataclass(frozen=True)
class FoldResult:
    fold: int
    best_fast: int
    best_slow: int
    is_sharpe: float
    oos_sharpe: float
    is_pnl_total: float
    oos_pnl_total: float

    @property
    def wfe(self) -> float:
        # WFE = OOS / IS only makes sense when IS is materially nonzero.
        # When the in-sample edge collapses to near zero, the ratio becomes
        # uninformative; report NaN and exclude it from the mean.
        if abs(self.is_sharpe) < 0.1:
            return float("nan")
        return self.oos_sharpe / self.is_sharpe


def per_bar_sharpe(per_bar_pnl: np.ndarray) -> float:
    """Per-bar Sharpe (mean / std). No artificial annualization on synthetic
    data, since bars-per-day is undefined here. The point of the demo is the
    IS-vs-OOS gap, not the absolute Sharpe scale."""
    if per_bar_pnl.std() < 1e-12:
        return 0.0
    return float(per_bar_pnl.mean() / per_bar_pnl.std())


def best_config_in_window(close: np.ndarray) -> tuple[int, int, float]:
    best = (FAST_GRID[0], SLOW_GRID[0], -np.inf)
    for fast in FAST_GRID:
        for slow in SLOW_GRID:
            if fast >= slow:
                continue
            cfg = StrategyConfig(fast=fast, slow=slow, kind="sma")
            pnl, _ = cfg.trades(close)
            sr = per_bar_sharpe(pnl)
            if sr > best[2]:
                best = (fast, slow, sr)
    return best


def run() -> tuple[list[FoldResult], dict[str, np.ndarray]]:
    bars = generate_ohlc(n_bars=N_BARS, seed=42)
    close = bars["close"]

    fold_size = len(close) // N_FOLDS
    is_size = int(fold_size * IS_FRACTION)
    results: list[FoldResult] = []

    for f in range(N_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < N_FOLDS - 1 else len(close)
        is_close = close[start : start + is_size]
        oos_close = close[start + is_size : end]
        if len(oos_close) < 50:
            continue

        best_fast, best_slow, is_sr = best_config_in_window(is_close)
        cfg = StrategyConfig(fast=best_fast, slow=best_slow, kind="sma")
        oos_pnl, _ = cfg.trades(oos_close)
        is_pnl, _ = cfg.trades(is_close)
        oos_sr = per_bar_sharpe(oos_pnl)
        results.append(
            FoldResult(
                fold=f + 1,
                best_fast=best_fast,
                best_slow=best_slow,
                is_sharpe=is_sr,
                oos_sharpe=oos_sr,
                is_pnl_total=float(is_pnl.sum()),
                oos_pnl_total=float(oos_pnl.sum()),
            )
        )

    return results, bars


def plot(results: list[FoldResult], out_path: Path) -> None:
    plt.style.use("default")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"hspace": 0.4})
    fig.patch.set_facecolor("#fafafa")

    text_color = "#111827"
    accent_is = "#9ca3af"
    accent_oos = "#2563eb"
    accent_line = "#dc2626"

    fold_labels = [f"fold {r.fold}\nfast={r.best_fast}, slow={r.best_slow}" for r in results]
    x = np.arange(len(results))
    width = 0.36
    is_sharpes = [r.is_sharpe for r in results]
    oos_sharpes = [r.oos_sharpe for r in results]

    ax_top.bar(x - width / 2, is_sharpes, width, color=accent_is, label="in-sample Sharpe", edgecolor="white")
    ax_top.bar(x + width / 2, oos_sharpes, width, color=accent_oos, label="out-of-sample Sharpe", edgecolor="white")
    ax_top.axhline(0, color="#d1d5db", linewidth=1)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(fold_labels, fontsize=9, color=text_color)
    ax_top.set_ylabel("per-bar Sharpe (mean / std)", color=text_color)
    ax_top.set_title(
        "Per-fold IS vs OOS Sharpe   |   the gap is the optimism penalty",
        loc="left", color=text_color, fontsize=12, weight="bold", pad=8,
    )
    ax_top.legend(frameon=False, loc="upper right")
    _style(ax_top, text_color)

    wfes = [r.wfe for r in results]
    ax_bot.bar(x, wfes, color=accent_oos, edgecolor="white", alpha=0.85)
    ax_bot.axhline(0.5, color=accent_line, linestyle="--", linewidth=1.0, label="WFE = 0.50 target")
    ax_bot.axhline(0, color="#d1d5db", linewidth=1)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"fold {r.fold}" for r in results], fontsize=9, color=text_color)
    ax_bot.set_ylabel("walk-forward efficiency", color=text_color)
    mean_wfe = float(np.nanmean(wfes))
    ax_bot.set_title(
        f"Walk-forward efficiency by fold   |   mean WFE = {mean_wfe:.2f}",
        loc="left", color=text_color, fontsize=12, weight="bold", pad=8,
    )
    ax_bot.legend(frameon=False, loc="upper right")
    _style(ax_bot, text_color)

    fig.suptitle(
        "Walk-Forward Validation   |   anchored folds, no re-optimization on the OOS half",
        x=0.06, y=0.99, ha="left", color=text_color, fontsize=13, weight="bold",
    )
    fig.text(
        0.06, 0.012,
        "Pardo (2008). The Evaluation and Optimization of Trading Strategies.    "
        "Synthetic OHLC, 8,000 bars, 6 folds, IS = 60% / OOS = 40% per fold.",
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
    print("Generating synthetic OHLC bars...")
    results, _ = run()
    out_path = OUT_DIR / "walk_forward.png"
    plot(results, out_path)

    print()
    print("Per-fold result")
    print("---------------")
    print(f"  {'fold':<6}{'fast':<6}{'slow':<6}{'IS Sharpe':>12}{'OOS Sharpe':>14}{'WFE':>8}")
    for r in results:
        print(f"  {r.fold:<6}{r.best_fast:<6}{r.best_slow:<6}"
              f"{r.is_sharpe:>12.2f}{r.oos_sharpe:>14.2f}{r.wfe:>8.2f}")
    mean_wfe = float(np.nanmean([r.wfe for r in results]))
    print()
    print(f"  Mean WFE       {mean_wfe:.2f}    (target > 0.50)")
    print()
    print(f"  Chart written to {out_path.relative_to(out_path.parents[2])}")


if __name__ == "__main__":
    main()
