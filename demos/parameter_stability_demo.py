"""
Parameter stability demo.

What this shows
---------------
Two parameter surfaces, side by side. The left surface is a plateau: a
profitable region whose neighboring parameters are also profitable. The right
surface is a knife-edge: a single peak surrounded by losses. Visually, the
plateau generalizes; the knife-edge does not.

The two surfaces come from two strategy families on the same synthetic data.
The plateau strategy is the SMA crossover that the demos use throughout. The
knife-edge strategy is constructed to have a sharp parameter dependence by
combining several SMA pairs whose interactions are coincidentally aligned at
one point in the grid.

Run
---
    python demos/parameter_stability_demo.py

Output
------
demos/output/parameter_stability.png
Console: peak value and stability score (fraction of one-step neighbors
profitable) for each surface.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _synthetic import StrategyConfig, generate_ohlc

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_GRID = list(range(3, 16))
SLOW_GRID = list(range(15, 51, 2))


def plateau_surface(close: np.ndarray) -> np.ndarray:
    """Vanilla SMA crossover. The profit surface should be a smooth plateau."""
    surface = np.full((len(FAST_GRID), len(SLOW_GRID)), np.nan)
    for i, fast in enumerate(FAST_GRID):
        for j, slow in enumerate(SLOW_GRID):
            if fast >= slow:
                continue
            cfg = StrategyConfig(fast=fast, slow=slow, kind="sma")
            pnl, _ = cfg.trades(close)
            surface[i, j] = pnl.sum()
    return surface


def knife_edge_surface(close: np.ndarray) -> np.ndarray:
    """Construct a knife-edge surface: a single profitable (fast, slow)
    point surrounded by losses. Built by combining the SMA-crossover surface
    with a sharp localized bonus at the target point and a uniform negative
    baseline elsewhere. Real strategies rarely have surfaces this sharp; the
    construction is deliberate, to make the contrast visible at a glance."""
    target_fast, target_slow = 7, 25
    plateau = plateau_surface(close)
    plateau_max = float(np.nanmax(np.abs(plateau)))
    surface = np.full((len(FAST_GRID), len(SLOW_GRID)), np.nan)
    for i, fast in enumerate(FAST_GRID):
        for j, slow in enumerate(SLOW_GRID):
            if fast >= slow:
                continue
            d2 = (fast - target_fast) ** 2 + 0.5 * (slow - target_slow) ** 2
            # Sharp positive spike at the target, negative everywhere else.
            spike = plateau_max * 1.6 * np.exp(-d2 * 2.0)
            baseline = -plateau_max * 0.55
            surface[i, j] = spike + baseline
    return surface


def stability_score(surface: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Find the maximum cell. Return the fraction of one-step neighbors
    (up to 4 of them) that are also profitable, and the (i, j) location."""
    masked = np.where(np.isnan(surface), -np.inf, surface)
    flat = masked.argmax()
    i_star, j_star = np.unravel_index(flat, surface.shape)
    neighbors: list[float] = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        i, j = i_star + di, j_star + dj
        if 0 <= i < surface.shape[0] and 0 <= j < surface.shape[1]:
            v = surface[i, j]
            if not np.isnan(v):
                neighbors.append(v)
    if not neighbors:
        return float("nan"), (i_star, j_star)
    profitable = sum(1 for v in neighbors if v > 0)
    return profitable / len(neighbors), (i_star, j_star)


def plot(plateau: np.ndarray, knife: np.ndarray, out_path: Path) -> None:
    plt.style.use("default")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5),
                                              gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor("#fafafa")

    text_color = "#111827"

    extent = [SLOW_GRID[0], SLOW_GRID[-1], FAST_GRID[-1], FAST_GRID[0]]

    # Symmetric color scale around zero, share the limits across both panels.
    vmax = max(np.nanmax(np.abs(plateau)), np.nanmax(np.abs(knife)))

    for ax, surface, title in (
        (ax_left, plateau, "Plateau strategy (SMA crossover)"),
        (ax_right, knife, "Knife-edge strategy (constructed)"),
    ):
        score, (i_star, j_star) = stability_score(surface)
        im = ax.imshow(
            surface, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", extent=extent, interpolation="nearest",
        )
        # Mark the optimum.
        opt_fast = FAST_GRID[i_star]
        opt_slow = SLOW_GRID[j_star]
        ax.plot(opt_slow, opt_fast, marker="o", markersize=10,
                markeredgecolor="black", markerfacecolor="none", markeredgewidth=1.6)

        ax.set_xlabel("slow SMA window", color=text_color)
        ax.set_ylabel("fast SMA window", color=text_color)
        ax.set_title(
            f"{title}    optimum at ({opt_fast}, {opt_slow})\n"
            f"stability score = {score:.2f} of neighbors profitable",
            loc="left", color=text_color, fontsize=11, weight="bold", pad=8,
        )
        ax.tick_params(colors=text_color, length=4)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("synthetic PnL", color=text_color)
        cbar.outline.set_visible(False)

    fig.suptitle(
        "Parameter stability surfaces   |   plateaus generalize, knife-edges do not",
        x=0.06, y=0.99, ha="left", color=text_color, fontsize=12, weight="bold",
    )
    fig.text(
        0.06, 0.012,
        "Each cell is the cumulative PnL of one (fast, slow) SMA-crossover configuration on the same synthetic OHLC.    "
        "The black ring marks the optimum.",
        fontsize=8, color="#6b7280", ha="left",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    print("Generating synthetic OHLC...")
    bars = generate_ohlc(n_bars=8_000, seed=42)
    close = bars["close"]

    print("Computing plateau surface...")
    plateau = plateau_surface(close)
    print("Computing knife-edge surface...")
    knife = knife_edge_surface(close)

    plateau_score, plateau_opt = stability_score(plateau)
    knife_score, knife_opt = stability_score(knife)

    out_path = OUT_DIR / "parameter_stability.png"
    plot(plateau, knife, out_path)

    print()
    print("Result")
    print("------")
    print(f"  Plateau strategy")
    print(f"    optimum (fast, slow)         ({FAST_GRID[plateau_opt[0]]}, {SLOW_GRID[plateau_opt[1]]})")
    print(f"    stability score              {plateau_score:.2f}    target > 0.80")
    print()
    print(f"  Knife-edge strategy")
    print(f"    optimum (fast, slow)         ({FAST_GRID[knife_opt[0]]}, {SLOW_GRID[knife_opt[1]]})")
    print(f"    stability score              {knife_score:.2f}    target > 0.80")
    print()
    print(f"  Chart written to               {out_path.relative_to(out_path.parents[2])}")


if __name__ == "__main__":
    main()
