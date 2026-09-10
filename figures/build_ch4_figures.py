# -*- coding: utf-8 -*-
"""Chapter 4 admission figure v7 -- chart only, explanation in the caption.

v6 carried a kicker, a two-line headline, a standfirst and a footnote block.
That is web-article furniture: in a dissertation the caption is where the
qualifications belong, typeset in the document's own font, and repeating
them inside the artwork just makes the figure a slide. v7 keeps the chart
and moves every sentence to \\caption{}.

Drawn rather than generated: the imagegen route already put a factual error
into the manuscript (fig-graph-expansion-imagegen-v1.png labels a discarded
node "Hub degree >25", but kg_corpus_build.py discards court rulings citing
more than 25 articles, not neighbours by node degree), and this figure is
nothing but exact recorded numbers at exact positions.

Arithmetic in the badges, checked against the recorded values:
  0.970 / 0.27   = 3.59  -> "clears the bar 3.6x"
  0.27  / 0.04   = 6.75  -> "7x short of the bar"
  0.27  / 0.0142 = 19.0  -> "19x short of the bar"

Numbers: 0.970 extractor (training queries, fires on 9.2%), 0.04 commercial
cross-encoder max score gate, 0.0142 keep/drop reranker at dense depth 2,000
-- the last two are ten-query court diagnostics outside the reproducible
score lineage. The 0.27 line is the empirical court-side working figure the
project recorded, never a reconstructed per-query pi*. All of that is in the
caption; none of it is asserted by the drawing.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT = Path("C:/Users/Dharun prasanth/OneDrive/Documents/Projects/"
           "LLm_Agentic/dissertation/overleaf_live/figures")

INK = "#172A3A"
SLATE = "#46586B"
BLUE = "#2364AA"
RED = "#C8553D"
GREY = "#7A848E"
TRACK = "#F2F1EE"
PALE_BLUE = "#E7F0F9"
PALE_RED = "#FAEAE4"
RULE = "#D9DDE1"

W, H = 5.56, 1.98
L, R = .285, .735
TOP, BOT = .855, .210
THRESH = 27.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

fig = plt.figure(figsize=(W, H))
ax = fig.add_subplot(111)
fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT)

rows = [
    (2, "Explicit extractor", "reads the identifier directly",
     97.0, "97 in 100", BLUE, PALE_BLUE, "Clears the bar 3.6$\\times$"),
    (1, "Cross-encoder gate", "scores candidate matches",
     4.0, "4 in 100", RED, PALE_RED, "7$\\times$ short of the bar"),
    (0, "Keep/drop reranker", "decides what stays",
     1.4, "1.4 in 100", RED, PALE_RED, "19$\\times$ short of the bar"),
]

ax.set(xlim=(0, 100), ylim=(-.62, 2.62), yticks=[])

# rounded ends that stay round: rounding_size is in x-units and the box is
# built after y is divided by mutation_aspect, so A equalises the two scales
xu = 100 / ((R - L) * W)
yu = 3.24 / ((TOP - BOT) * H)
ASPECT = yu / xu
RAD = 1.8 / 72 * xu
BH = .56


def bar(x0, x1, y, colour, z):
    r = min(RAD, max((x1 - x0) / 2.4, 1e-6))
    ax.add_patch(FancyBboxPatch(
        (x0, y - BH / 2), x1 - x0, BH, mutation_aspect=ASPECT, zorder=z,
        boxstyle="round,pad=0,rounding_size=%.4f" % r,
        linewidth=0, facecolor=colour))


gt = ax.get_yaxis_transform()          # x axes-fraction, y data

for y, title, sub, v, vtxt, colour, pale, badge in rows:
    bar(0, 100, y, TRACK, 1)
    bar(0, v, y, colour, 2)
    ax.text(-.032, y + .17, title, transform=gt, clip_on=False, zorder=4,
            fontsize=8.2, fontweight="bold", color=INK, ha="right",
            va="center")
    ax.text(-.032, y - .25, sub, transform=gt, clip_on=False, zorder=4,
            fontsize=6.6, color=GREY, ha="right", va="center")
    inside = v > 40
    ax.text(v - 2.4 if inside else v + 3.0, y, vtxt, zorder=8,
            fontsize=8.4, family="DejaVu Sans Mono", fontweight="bold",
            color="white" if inside else colour,
            ha="right" if inside else "left", va="center",
            path_effects=[] if inside else
            [pe.withStroke(linewidth=2.6, foreground="white")])
    ax.text(1.050, y, badge, transform=gt, clip_on=False, zorder=4,
            fontsize=6.3, fontweight="bold", color=colour, ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=.42", facecolor=pale,
                      edgecolor="none"))

ax.vlines(THRESH, -.62, 2.62, color=INK, lw=1.0, ls=(0, (3.2, 2.2)),
          zorder=6)
ax.text(THRESH, 2.96, "Working threshold  $\\mathtt{27\\ in\\ 100}$",
        zorder=7, clip_on=False, fontsize=6.9, color="white", ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=.42", facecolor=INK, edgecolor="none"))

ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Correct citations out of every 100 added", fontsize=7.4,
              color=SLATE, labelpad=5)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(RULE)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0, colors=GREY, labelsize=6.9, pad=3)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig-admission-bar-v7.pdf")
fig.savefig(OUT / "fig-admission-bar-v7.png", dpi=340)
plt.close(fig)
print("wrote fig-admission-bar-v7.pdf / .png")
