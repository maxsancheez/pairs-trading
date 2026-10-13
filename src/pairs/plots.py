"""Charts for a single pair."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_pair(
    prices: pd.DataFrame,
    y: str,
    x: str,
    z: pd.Series,
    entry: float = 2.0,
    out_path: Path | None = None,
):
    """Three panels: normalized prices, the spread's z-score with entry
    bands, and the z-score distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    normalized = prices[[y, x]] / prices[[y, x]].iloc[0] * 100
    normalized.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title(f"{y} vs {x} (normalized to 100)")
    axes[0].set_ylabel("Price index")

    axes[1].plot(z.index, z, color="#444", linewidth=1)
    for level, color in [(entry, "#c0392b"), (-entry, "#27ae60"), (0, "#888")]:
        axes[1].axhline(level, color=color, linestyle="--", linewidth=1)
    axes[1].fill_between(z.index, entry, z, where=z > entry, color="#c0392b", alpha=0.3)
    axes[1].fill_between(z.index, -entry, z, where=z < -entry, color="#27ae60", alpha=0.3)
    axes[1].set_title(f"Spread z-score (red: short {y}/long {x}, green: long {y}/short {x})")
    axes[1].set_ylabel("Z-score")

    z.dropna().hist(ax=axes[2], bins=60, color="#5b7db1", edgecolor="white")
    axes[2].set_title("Z-score distribution")
    axes[2].set_xlabel("Z-score")

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig
