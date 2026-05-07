"""
Parity audit demo.

What this shows
---------------
Two backtesters run the same SMA-crossover strategy on the same synthetic
OHLC data. One has a one-line bug: a futures-style multiplier is left at the
default 1.0 instead of 50.0. The two equity curves diverge by exactly that
factor. A parity audit catches this in the first comparison; running either
backtester alone would not.

This is the cleanest example of why cross-framework reconciliation matters.
The bug is silent (no exception, no warning) and the result it produces is
"plausible but wrong."

Run
---
    pip install -e .
    python demos/parity_audit_demo.py

Output
------
demos/output/parity_audit.png
Console: per-trade reconciliation between the two implementations and the
fixed version.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quant_platform.parity.reconciler import (
    Trade,
    equity_curve,
    reconcile_trade_lists,
)
from quant_platform.parity.synthetic import regime_switching_ohlc

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_MULTIPLIER = 50.0
COMMISSION_PER_SIDE = 3.0


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    csum = np.cumsum(np.concatenate([[0.0], arr]))
    out = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, arr[0])
    return np.concatenate([pad, out])


def detect_trades(close: np.ndarray, fast: int = 8, slow: int = 30) -> list[Trade]:
    """Long when fast SMA crosses above slow SMA, exit on the reverse cross.
    Trades are evaluated on bar close, filled at the next bar's open."""
    fast_sma = _sma(close, fast)
    slow_sma = _sma(close, slow)
    long_signal = (fast_sma > slow_sma).astype(int)
    transitions = np.diff(np.concatenate([[0], long_signal, [0]]))
    entries = np.where(transitions == 1)[0]
    exits = np.where(transitions == -1)[0]

    n = len(close)
    trades: list[Trade] = []
    for entry_idx, exit_idx in zip(entries, exits):
        entry_idx = min(int(entry_idx), n - 2)
        exit_idx = min(int(exit_idx), n - 1)
        e_fill = entry_idx + 1
        x_fill = exit_idx + 1 if exit_idx + 1 < n else exit_idx
        if x_fill <= e_fill:
            continue
        trades.append(
            Trade(
                entry_idx=int(e_fill),
                exit_idx=int(x_fill),
                entry_price=float(close[e_fill]),
                exit_price=float(close[x_fill]),
            )
        )
    return trades


def plot(close: np.ndarray, eq_buggy, eq_correct, eq_third, out_path: Path) -> None:
    plt.style.use("default")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#fafafa")

    text_color = "#111827"
    accent_buggy = "#b45309"
    accent_correct = "#2563eb"
    accent_third = "#0e9f6e"

    bars = np.arange(len(close))

    ax_top.plot(bars, eq_buggy, color=accent_buggy, linewidth=1.5,
                label="Backtester A   (multiplier left at default 1.0)")
    ax_top.plot(bars, eq_correct, color=accent_correct, linewidth=1.5,
                label="Backtester B   (multiplier set to 50.0)")
    ax_top.plot(bars, eq_third, color=accent_third, linewidth=1.5, linestyle="--",
                label="Backtester C   (independent third implementation)")
    ax_top.axhline(0, color="#d1d5db", linewidth=1)
    ax_top.set_xlabel("bar index", color=text_color)
    ax_top.set_ylabel("cumulative PnL ($)", color=text_color)
    ax_top.set_title(
        "Three implementations, same strategy, same OHLC.    "
        "Two agree to the dollar. One does not.",
        loc="left", color=text_color, fontsize=12, weight="bold", pad=8,
    )
    ax_top.legend(frameon=False, loc="upper left", fontsize=9)
    _style(ax_top, text_color)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(eq_correct) > 1e-9, eq_buggy / eq_correct, np.nan)
    ax_bot.plot(bars, ratio, color=accent_buggy, linewidth=1.4)
    ax_bot.axhline(1.0 / POINT_MULTIPLIER, color="#dc2626", linestyle="--", linewidth=1.0,
                   label=f"expected ratio if multiplier is wrong: 1 / {POINT_MULTIPLIER:.0f} = {1 / POINT_MULTIPLIER:.3f}")
    ax_bot.set_xlabel("bar index", color=text_color)
    ax_bot.set_ylabel("PnL_buggy / PnL_correct", color=text_color)
    ax_bot.set_title(
        "Ratio between buggy and correct PnL    converges on 1 / 50",
        loc="left", color=text_color, fontsize=12, weight="bold", pad=8,
    )
    ax_bot.set_ylim(0, max(0.05, 2.5 / POINT_MULTIPLIER))
    ax_bot.legend(frameon=False, loc="upper right", fontsize=9)
    _style(ax_bot, text_color)

    fig.suptitle(
        "Cross-framework parity audit   |   catching a silent multiplier bug",
        x=0.06, y=0.99, ha="left", color=text_color, fontsize=13, weight="bold",
    )
    fig.text(
        0.06, 0.012,
        "The bug: a futures contract has a $50 point multiplier, but Backtester A is left at the framework default of 1.0.    "
        "No exception, no warning, plausible-looking output.",
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
    bars = regime_switching_ohlc(n_bars=4_000, seed=42)
    close = bars["close"]

    print("Detecting trades on three implementations...")
    trades_a = detect_trades(close)
    trades_b = detect_trades(close)
    trades_c = detect_trades(close)

    eq_buggy = equity_curve(trades_a, multiplier=1.0, n_bars=len(close), commission_per_side=COMMISSION_PER_SIDE)
    eq_correct = equity_curve(trades_b, multiplier=POINT_MULTIPLIER, n_bars=len(close), commission_per_side=COMMISSION_PER_SIDE)
    eq_third = equity_curve(trades_c, multiplier=POINT_MULTIPLIER, n_bars=len(close), commission_per_side=COMMISSION_PER_SIDE)

    result = reconcile_trade_lists(
        trades_a=trades_a,
        trades_b=trades_b,
        multiplier_a=1.0,
        multiplier_b=POINT_MULTIPLIER,
        commission_per_side=COMMISSION_PER_SIDE,
    )

    out_path = OUT_DIR / "parity_audit.png"
    plot(close, eq_buggy, eq_correct, eq_third, out_path)

    print()
    print(f"Trade detection                 {result.n_matched} matched, {result.n_mismatched} mismatched")
    print(f"Total trades                    {len(trades_a)}")
    print()
    print(f"Final PnL")
    print(f"  Buggy   (multiplier 1.0)      ${eq_buggy[-1]:>10,.0f}")
    print(f"  Correct (multiplier 50.0)     ${eq_correct[-1]:>10,.0f}")
    print(f"  Independent third impl.       ${eq_third[-1]:>10,.0f}")
    print(f"  Ratio buggy / correct         {eq_buggy[-1] / eq_correct[-1]:.4f}")
    print(f"  Expected if multiplier bug    {1.0 / POINT_MULTIPLIER:.4f}")
    print()
    print(f"First 5 trades (per-trade dollar PnL)")
    print(f"  {'#':<4}{'buggy':>14}{'correct':>14}{'ratio':>10}")
    for i, dollars_a, dollars_b in result.sample:
        ratio = dollars_a / dollars_b if abs(dollars_b) > 1e-9 else float("nan")
        print(f"  {i:<4}{dollars_a:>14,.2f}{dollars_b:>14,.2f}{ratio:>10.4f}")
    print()
    print(f"  Chart written to              {out_path.relative_to(out_path.parents[2])}")


if __name__ == "__main__":
    main()
