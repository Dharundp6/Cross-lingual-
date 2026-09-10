"""Create a reader-first router comparison figure from the recorded values.

This is deliberately a deterministic vector chart, rather than a generated image:
the labels, intervals and values are evidence in the dissertation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#263746"
MUTED = "#607080"
GRID = "#d7dde3"
CLAUDE = "#c34d2f"
RESPLIT = "#2c73b9"
CONTROL = "#6e7b86"


def draw_interval(ax, y, value, low, high, colour, marker, annotation):
    ax.hlines(y, low, high, color=colour, linewidth=2.0, zorder=3)
    ax.vlines([low, high], y - 0.12, y + 0.12, color=colour, linewidth=1.5, zorder=3)
    ax.scatter(value, y, s=54, marker=marker, color=colour, zorder=4)
    ax.text(0.0315, y, annotation, color=colour, fontsize=9, va="center", ha="left",
            fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(8.1, 2.55))
    fig.subplots_adjust(left=0.32, right=0.86, top=0.94, bottom=0.27)

    rows = [
        (2, "Claude: pre-declared 497/496 split", 0.02111, 0.01175, 0.03074, CLAUDE, "D",
         "+0.02111  [0.01175, 0.03074]"),
        (1, "Claude: mean across 100 re-splits", 0.01523, 0.01139, 0.01932, RESPLIT, "o",
         "+0.01523  [0.01139, 0.01932]"),
        (0, "Keyword control: pre-declared split", 0.00357, 0.00020, 0.00712, CONTROL, "s",
         "+0.00357  [0.00020, 0.00712]"),
    ]
    for row in rows:
        draw_interval(ax, row[0], *row[2:])

    ax.axvline(0, color=INK, linewidth=1.1, zorder=1)
    ax.set_xlim(-0.0015, 0.0405)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([row[1] for row in rows], fontsize=9, color=INK)
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xticklabels(["0", "+0.01", "+0.02", "+0.03"], fontsize=8.5, color=MUTED)
    ax.set_xlabel("Improvement in Macro-F1", fontsize=9.3, color=INK, labelpad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig-router-comparison-v8.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig-router-comparison-v8.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
