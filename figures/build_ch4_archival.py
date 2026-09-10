# -*- coding: utf-8 -*-
"""Chapter 4 figures, archival house style.

Rebuilt after checking the conventions actually used in IEEE Transactions and
Springer LNCS papers in IR and data science. Four findings changed the design,
and all four are applied here:

1. TYPE SIZE. Both venues target 9-10 pt inside figures at final printed size;
   6 pt is described as an emergency floor for a nonessential secondary tick
   label. The previous versions ran 6.2-6.9 pt at a 5.56 in final width with no
   scaling, i.e. below the floor for every label in the figure. Nothing here is
   under 8.5 pt and the substantive labels are 9-9.5 pt.

2. NO MONOSPACE NUMERALS. A monospaced font for plotted values reads as console
   output; it is reserved for code, identifiers and run tags. All values are
   proportional now.

3. NO INFOGRAPHIC FURNITURE. Rounded pill badges, colour-filled number chips
   and tinted panel grounds are slide-deck styling in archival IR work. White
   ground, one accent colour per series, plain unfilled annotations.

4. NAME THE RANGE TYPE. Two overlapping intervals could be min-max, a
   percentile band, a bootstrap CI or fold variation, and those are different
   claims. Figure C plots observed minimum and maximum; the caption says so.

Type scale, fixed once and shared:
    9.5 pt  series and row names, plotted values
    9.0 pt  axis labels, panel letters
    8.5 pt  tick labels, annotations

Explanatory sublabels were dropped rather than shrunk. At 9 pt they no longer
fit the label column, and they were caption material in the first place --
which is where the archival split puts them: caption decodes the figure, body
argues from it.

Every number is quoted from the chapter or the archive. Nothing is computed,
rounded or inferred here.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT = Path("C:/Users/Dharun prasanth/OneDrive/Documents/Projects/"
           "LLm_Agentic/dissertation/overleaf_live/figures")

INK = "#161F27"
GREY = "#5F6B76"
BLUE = "#2364AA"
RED = "#C8553D"
TRACK = "#ECECE9"
RULE = "#C9CFD5"
W = 5.56

FS_NAME = 9.5
FS_VAL = 9.5
FS_AXIS = 9.0
FS_TICK = 8.5
FS_ANN = 8.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


def bare(ax, keep_bottom=True):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)
    if keep_bottom:
        ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, colors=GREY, labelsize=FS_TICK, pad=3)


# ------------------------------------------------------- 4.1 admission -----
def fig_admission():
    H = 1.80
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.300, right=.755, top=.815, bottom=.300)

    rows = [(2, "Explicit extractor", 97.0, "97", BLUE, "3.6$\\times$ above"),
            (1, "Cross-encoder gate", 4.0, "4.0", RED, "7$\\times$ below"),
            (0, "Keep/drop reranker", 1.4, "1.4", RED, "19$\\times$ below")]

    ax.set(xlim=(0, 100), ylim=(-.62, 2.62), yticks=[])
    gt = ax.get_yaxis_transform()

    for y, name, v, vtxt, colour, ann in rows:
        ax.barh(y, 100, height=.52, color=TRACK, zorder=1)
        ax.barh(y, v, height=.52, color=colour, zorder=2)
        ax.text(-.030, y, name, transform=gt, clip_on=False, zorder=4,
                fontsize=FS_NAME, color=INK, ha="right", va="center")
        inside = v > 40
        ax.text(v - 2.0 if inside else v + 2.0, y, vtxt, zorder=5,
                fontsize=FS_VAL, color="white" if inside else colour,
                ha="right" if inside else "left", va="center")
        ax.text(1.035, y, ann, transform=gt, clip_on=False, zorder=4,
                fontsize=FS_ANN, color=GREY, ha="left", va="center")

    ax.axvline(27, color=INK, lw=1.0, ls=(0, (3.2, 2.4)), zorder=6)
    ax.text(27, 2.86, "working threshold 27", fontsize=FS_ANN, color=INK,
            ha="center", va="center", clip_on=False)

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Correct citations out of every 100 added",
                  fontsize=FS_AXIS, color=INK, labelpad=5)
    bare(ax)
    save(fig, "fig-admission-bar-v8")
    plt.close(fig)
    print("4.1  fig-admission-bar-v8")


# ------------------------------------------------------- B reach/answer ----
def fig_reach():
    H = 1.82
    fig = plt.figure(figsize=(W, H))
    spec = [
        (.115, "Fine-tuning the\ndense retriever", "Statute Recall@500",
         "ten-query set", .34, .54, "0.34", "0.54", "before", "after",
         BLUE, (.26, .62), [.3, .4, .5, .6]),
        (.620, "Removing the\ndense retriever", "Macro-F1",
         "development benchmark", .12093, .15887, "0.12093", "0.15887",
         "dense in", "dense out", RED, (.108, .172), [.12, .14, .16]),
    ]
    for (x0, title, ylab, setlab, v0, v1, t0, t1, xl0, xl1, colour,
         ylim, yticks) in spec:
        ax = fig.add_axes([x0, .200, .325, .465])
        ax.plot([0, 1], [v0, v1], color=colour, lw=1.8, zorder=3)
        ax.scatter([0, 1], [v0, v1], s=34, color=colour, zorder=4)
        ax.annotate(t0, (0, v0), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=FS_VAL, color=colour)
        ax.annotate(t1, (1, v1), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=FS_VAL, color=colour)
        ax.set(xlim=(-.42, 1.42), ylim=ylim, xticks=[0, 1], yticks=yticks)
        ax.set_xticklabels([xl0, xl1], fontsize=FS_TICK, color=GREY)
        ax.set_ylabel(ylab, fontsize=FS_AXIS, color=INK, labelpad=7)
        ax.tick_params(axis="y", labelsize=FS_TICK, colors=GREY, length=2)
        bare(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(RULE)
        fig.text(x0, .960, title, fontsize=FS_NAME, fontweight="bold",
                 color=INK, ha="left", va="top", linespacing=1.35)
        fig.text(x0, .742, setlab, fontsize=FS_ANN, color=GREY, ha="left",
                 va="center")
    save(fig, "fig-reach-vs-answer-v2")
    plt.close(fig)
    print("B    fig-reach-vs-answer-v2")


# ------------------------------------------------------- C failure walls ---
def fig_walls():
    H = 2.05
    fig = plt.figure(figsize=(W, H))

    axa = fig.add_axes([.105, .755, .700, .105])
    axa.set(xlim=(0, 100), ylim=(0, 1))
    axa.barh(.5, 95, height=1.0, color=BLUE)
    axa.barh(.5, 5, left=95, height=1.0, color=TRACK)
    axa.text(2.2, .5, "Parent article absent   95%", fontsize=FS_NAME,
             color="white", ha="left", va="center", zorder=3)
    axa.text(102, .5, "present  5%", fontsize=FS_ANN, color=GREY, ha="left",
             va="center", clip_on=False)
    axa.axis("off")
    fig.text(.105, .930, "(a)  Where the missed statute gold went",
             fontsize=FS_AXIS, fontweight="bold", color=INK, ha="left",
             va="center")

    axb = fig.add_axes([.275, .215, .665, .285])
    axb.fill_betweenx([-.55, 1.55], .55, .69, color=TRACK, zorder=0)
    for y, lo, hi, lab, colour in [(1, .51, .69, "Correct paragraph", BLUE),
                                   (0, .55, .71, "Wrong sibling", RED)]:
        axb.plot([lo, hi], [y, y], color=colour, lw=1.8, zorder=3)
        for x in (lo, hi):
            axb.plot([x, x], [y - .17, y + .17], color=colour, lw=1.6,
                     zorder=3)
        axb.text(-.030, y, lab, transform=axb.get_yaxis_transform(),
                 fontsize=FS_NAME, color=INK, ha="right", va="center",
                 clip_on=False)
        axb.text(hi + .004, y, "%.2f\u2013%.2f" % (lo, hi), fontsize=FS_ANN,
                 color=colour, ha="left", va="center", zorder=4)
    axb.text(.62, -.66, "overlap 0.55\u20130.69", fontsize=FS_ANN, color=GREY,
             ha="center", va="center")
    axb.set(xlim=(.495, .755), ylim=(-1.05, 1.60), yticks=[])
    axb.set_xticks([.50, .55, .60, .65, .70])
    axb.set_xlabel("Observed cosine similarity to the question",
                   fontsize=FS_AXIS, color=INK, labelpad=4)
    bare(axb)
    fig.text(.105, .585, "(b)  Where the article was reached",
             fontsize=FS_AXIS, fontweight="bold", color=INK, ha="left",
             va="center")

    save(fig, "fig-failure-walls-v6")
    plt.close(fig)
    print("C    fig-failure-walls-v6")


# ------------------------------------------------------- D router forest ---
def fig_router():
    H = 1.70
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.325, .245, .530, .665])

    rows = [(2, "Pre-declared split", .02111, .01175, .03074, RED, "D"),
            (1, "100 post-hoc re-splits", .01523, .01139, .01932, BLUE, "o"),
            (0, "Keyword control", .00357, .00020, .00712, GREY, "s")]

    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    gt = ax.get_yaxis_transform()
    for y, name, v, lo, hi, colour, mk in rows:
        ax.plot([lo, hi], [y, y], color=colour, lw=1.5, zorder=3)
        for x in (lo, hi):
            ax.plot([x, x], [y - .12, y + .12], color=colour, lw=1.5,
                    zorder=3)
        ax.scatter([v], [y], marker=mk, s=40, color=colour, zorder=4)
        ax.text(-.032, y, name, transform=gt, clip_on=False, fontsize=FS_NAME,
                color=INK, ha="right", va="center")
        ax.text(hi + .0011, y, "%+.5f" % v, fontsize=FS_ANN, color=colour,
                ha="left", va="center")

    ax.set(xlim=(-.0018, .0372), ylim=(-.60, 2.60), yticks=[])
    ax.set_xticks([0, .01, .02, .03])
    ax.set_xticklabels(["0", "+0.01", "+0.02", "+0.03"])
    ax.set_xlabel("Change in Macro-F1 on the scored half", fontsize=FS_AXIS,
                  color=INK, labelpad=5)
    bare(ax)
    save(fig, "fig-router-sensitivity-v5")
    plt.close(fig)
    print("D    fig-router-sensitivity-v5")


PREV = Path(__file__).resolve().parent


def save(fig, stem):
    """Deliverable PDF at print quality, plus a small PNG for the canvas."""
    fig.savefig(OUT / (stem + ".pdf"))
    fig.savefig(OUT / (stem + ".png"), dpi=340)
    fig.savefig(PREV / (stem + "-preview.png"), dpi=150)


OUT.mkdir(parents=True, exist_ok=True)
fig_admission()
fig_reach()
fig_walls()
fig_router()
