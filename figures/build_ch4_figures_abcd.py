# -*- coding: utf-8 -*-
"""Chapter 4 figures A-D. House style, drawn not generated.

Shared rules, established on Figure 4.1 and applied to all four:
  * 5.56 in house width, vector PDF, Chapter 3's #2364AA / #C8553D pair
  * the artwork carries numbers; the caption carries every qualification
  * no headline, standfirst or footnote block inside the drawing

Every number below is quoted from the chapter or the archive. Nothing is
computed, rounded or inferred here.

A  Four kinds of evidence (sec 4.1.3). A crosswalk, not a ranking: sec 4.1.3
   lists the four tiers without ordering them by strength, and any ordinal
   encoding here would invent a claim the text does not make. One accent
   colour throughout, deliberately, for that reason.

B  Fine-tuning the dense retriever vs removing it (sec 4.5). Two panels, each
   on its OWN axis with its own evaluation set named, because Table 4.2's
   caption states the columns are not comparable across rows. Panels are
   titled by the INTERVENTION, not the metric: both slopes rise, and the
   finding is that improving the component and deleting the component both
   helped, measured on different things.

C  Failure walls (sec 4.7), replacing fig-failure-walls-imagegen-v3.png. That
   image draws the wrong-sibling range ending near 0.725 against a recorded
   0.71 -- the one quantity the panel exists to show. 95% parent-absent is
   correct and is recorded in analysis/DIAGNOSTIC_FINDINGS_REPORT.md line 24.

D  Router sensitivity (sec 4.8), replacing fig-router-sensitivity-v3.pdf and
   moving into the body. Adds the keyword control the old figure omitted.
   The re-split row's interval is an EMPIRICAL RANGE across re-splits, not a
   bootstrap CI like the other two, so it is drawn with a different mark.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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
W = 5.56
MONO = "DejaVu Sans Mono"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


def rounded(ax, x0, y0, w, h, aspect, colour, edge="none", lw=0, z=1,
            rad=2.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, mutation_aspect=aspect, zorder=z,
        boxstyle="round,pad=0,rounding_size=%.5f" % rad,
        linewidth=lw, edgecolor=edge, facecolor=colour))


# ---------------------------------------------------------------- A --------
def fig_a():
    H = 2.42
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    rows = [
        ("Development benchmark", "~280 submissions; returns\ninformed later design",
         ["0.16898", "0.18386"], "a held-out estimate"),
        ("Ten-query diagnostics", "n = 10",
         ["0.34\u21920.54", "0.0142"], "resolving small differences"),
        ("One pre-declared split", "497 select / 496 score",
         ["+0.02111"], "a magnitude"),
        ("Post-hoc sensitivity", "100 re-splits of the\nsame predictions",
         ["+0.01523"], "any independent claim"),
    ]

    top, gap = .945, .012
    hrow = (top - .055 - 3 * gap) / 4

    ax.text(.040, .988, "EVIDENCE", fontsize=5.8, fontweight="bold",
            color=GREY, ha="left", va="center")
    ax.text(.410, .988, "WHAT RESTS ON IT", fontsize=5.8, fontweight="bold",
            color=GREY, ha="left", va="center")
    ax.text(.590, .988, "WHAT IT CANNOT SUPPORT", fontsize=5.8,
            fontweight="bold", color=GREY, ha="left", va="center")

    for i, (name, scale, chips, cannot) in enumerate(rows):
        y1 = top - i * (hrow + gap)
        y0 = y1 - hrow
        mid = (y0 + y1) / 2
        ax.add_patch(plt.Rectangle((.026, y0), .952, hrow, facecolor=TRACK,
                                   edgecolor="none", zorder=0))
        ax.add_patch(plt.Rectangle((.026, y0), .0075, hrow, facecolor=BLUE,
                                   edgecolor="none", zorder=1))
        ax.text(.050, mid + .045, name, fontsize=7.9, fontweight="bold",
                color=INK, ha="left", va="center")
        ax.text(.050, mid - .052, scale, fontsize=6.2, color=GREY,
                ha="left", va="center", linespacing=1.3)
        cx = .410
        for c in chips:
            t = ax.text(cx, mid, c, fontsize=7.2, family=MONO,
                        fontweight="bold", color=BLUE, ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=.34",
                                  facecolor=PALE_BLUE, edgecolor="none"))
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(fig.transFigure.inverted())
            cx = bb.x1 + .017
        ax.text(.590, mid, cannot, fontsize=6.9, color=RED, style="italic",
                ha="left", va="center")

    fig.savefig(OUT / "fig-evidence-kinds-v1.pdf")
    fig.savefig(OUT / "fig-evidence-kinds-v1.png", dpi=340)
    plt.close(fig)
    print("A  fig-evidence-kinds-v1")


# ---------------------------------------------------------------- B --------
def fig_b():
    H = 2.16
    fig = plt.figure(figsize=(W, H))
    panels = [
        (fig.add_axes([.085, .215, .375, .565]),
         "Fine-tuning the dense retriever",
         "Statute Recall@500, ten queries",
         0.34, 0.54, "0.34", "0.54", "before", "after", BLUE, PALE_BLUE,
         (.24, .64)),
        (fig.add_axes([.605, .215, .375, .565]),
         "Removing the dense retriever",
         "Development benchmark Macro-F1",
         0.12093, 0.15887, "0.12093", "0.15887", "with", "removed",
         RED, PALE_RED, (.105, .175)),
    ]
    for (ax, title, ylab, v0, v1, t0, t1, x0lab, x1lab, colour, pale,
         ylim) in panels:
        ax.plot([0, 1], [v0, v1], color=colour, lw=2.0, zorder=3,
                solid_capstyle="round")
        ax.scatter([0, 1], [v0, v1], s=46, color=colour, zorder=4,
                   edgecolors="white", linewidths=1.4)
        ax.text(0, v0, "  " + t0, fontsize=7.8, family=MONO,
                fontweight="bold", color=colour, ha="left", va="top")
        ax.text(1, v1, t1 + "  ", fontsize=7.8, family=MONO,
                fontweight="bold", color=colour, ha="right", va="bottom")
        ax.set(xlim=(-.30, 1.30), ylim=ylim, xticks=[0, 1])
        ax.set_xticklabels([x0lab, x1lab], fontsize=7.0, color=SLATE)
        ax.set_title(title, fontsize=8.0, fontweight="bold", color=INK,
                     pad=8)
        ax.set_ylabel(ylab, fontsize=6.8, color=SLATE, labelpad=4)
        ax.tick_params(axis="y", labelsize=6.4, colors=GREY, length=2)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(RULE)
        ax.set_facecolor(pale)
        ax.patch.set_alpha(.35)

    fig.text(.5, .050, "Different interventions, different measurements: the "
                       "two panels do not share a scale.",
             fontsize=6.8, color=GREY, ha="center", va="center")
    fig.savefig(OUT / "fig-reach-vs-answer-v1.pdf")
    fig.savefig(OUT / "fig-reach-vs-answer-v1.png", dpi=340)
    plt.close(fig)
    print("B  fig-reach-vs-answer-v1")


# ---------------------------------------------------------------- C --------
def fig_c():
    H = 1.92
    fig = plt.figure(figsize=(W, H))

    axa = fig.add_axes([.085, .585, .875, .150])
    axa.set(xlim=(0, 100), ylim=(0, 1), yticks=[], xticks=[])
    asp_a = (1 / (.150 * H)) / (100 / (.875 * W))
    rounded(axa, 0, .12, 95, .76, asp_a, BLUE, z=2, rad=1.1)
    rounded(axa, 95, .12, 5, .76, asp_a, RULE, z=2, rad=1.1)
    axa.text(2.4, .5, "Parent article absent  95%", fontsize=7.6,
             fontweight="bold", color="white", ha="left", va="center",
             zorder=3)
    axa.text(101.5, .5, "present  5%", fontsize=6.8, color=GREY, ha="left",
             va="center", zorder=3, clip_on=False)
    axa.axis("off")
    fig.text(.085, .800, "a   WHERE THE MISSED STATUTE GOLD WENT",
             fontsize=6.2, fontweight="bold", color=GREY, ha="left",
             va="center")
    fig.text(.085, .530, "Share of missed statute gold, ten-query audit",
             fontsize=6.5, color=GREY, ha="left", va="top")

    axb = fig.add_axes([.235, .175, .620, .215])
    axb.fill_betweenx([-.6, 1.6], .55, .69, color=TRACK, zorder=0)
    for y, lo, hi, lab, colour in [(1, .51, .69, "Correct paragraph", BLUE),
                                   (0, .55, .71, "Wrong sibling", RED)]:
        axb.plot([lo, hi], [y, y], color=colour, lw=2.4, zorder=3,
                 solid_capstyle="butt")
        for x in (lo, hi):
            axb.plot([x, x], [y - .19, y + .19], color=colour, lw=1.6,
                     zorder=3)
        axb.text(-.055, y, lab, transform=axb.get_yaxis_transform(),
                 fontsize=7.3, fontweight="bold", color=INK, ha="right",
                 va="center", clip_on=False)
        axb.text((lo + hi) / 2, y + .34, "%.2f\u2013%.2f" % (lo, hi),
                 fontsize=6.8, family=MONO, color=colour, ha="center",
                 va="center", zorder=4)
    axb.text(.62, -.72, "overlap 0.55\u20130.69", fontsize=6.6, color=SLATE,
             ha="center", va="center")
    axb.set(xlim=(.495, .735), ylim=(-.95, 1.62), yticks=[])
    axb.set_xticks([.50, .55, .60, .65, .70])
    axb.set_xlabel("Observed cosine similarity to the question", fontsize=6.9,
                   color=SLATE, labelpad=3)
    axb.spines[["top", "right", "left"]].set_visible(False)
    axb.spines["bottom"].set_color(RULE)
    axb.tick_params(axis="y", length=0)
    axb.tick_params(axis="x", length=0, colors=GREY, labelsize=6.5, pad=2)
    fig.text(.085, .445, "b   WHERE THE ARTICLE WAS REACHED",
             fontsize=6.2, fontweight="bold", color=GREY, ha="left",
             va="center")

    fig.savefig(OUT / "fig-failure-walls-v5.pdf")
    fig.savefig(OUT / "fig-failure-walls-v5.png", dpi=340)
    plt.close(fig)
    print("C  fig-failure-walls-v5")


# ---------------------------------------------------------------- D --------
def fig_d():
    H = 1.86
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.335, .275, .625, .615])

    rows = [
        (2, "Pre-declared split", "497 select / 496 score", .02111,
         .01175, .03074, RED, "D", "paired bootstrap 95% CI"),
        (1, "100 post-hoc re-splits", "same archived predictions", .01523,
         .01139, .01932, BLUE, "o", "empirical 95% range"),
        (0, "Keyword control", "identical code path", .00357,
         .00020, .00712, GREY, "s", "paired bootstrap 95% CI"),
    ]
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    for y, name, sub, v, lo, hi, colour, mk, _ in rows:
        ax.plot([lo, hi], [y, y], color=colour, lw=1.5, zorder=3,
                solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [y - .13, y + .13], color=colour, lw=1.5,
                    zorder=3)
        ax.scatter([v], [y], marker=mk, s=52, color=colour, zorder=4,
                   edgecolors="white", linewidths=1.2)
        gt = ax.get_yaxis_transform()
        ax.text(-.035, y + .13, name, transform=gt, clip_on=False,
                fontsize=7.7, fontweight="bold", color=INK, ha="right",
                va="center")
        ax.text(-.035, y - .17, sub, transform=gt, clip_on=False,
                fontsize=6.2, color=GREY, ha="right", va="center")
        ax.text(hi + .0009, y, "%+.5f" % v, fontsize=7.1, family=MONO,
                fontweight="bold", color=colour, ha="left", va="center")

    ax.set(xlim=(-.0016, .0375), ylim=(-.62, 2.62), yticks=[])
    ax.set_xticks([0, .01, .02, .03])
    ax.set_xticklabels(["0", "+0.01", "+0.02", "+0.03"])
    ax.set_xlabel("Change in Macro-F1 on the scored half", fontsize=7.2,
                  color=SLATE, labelpad=4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, colors=GREY, labelsize=6.9, pad=3)

    fig.savefig(OUT / "fig-router-sensitivity-v5.pdf")
    fig.savefig(OUT / "fig-router-sensitivity-v5.png", dpi=340)
    plt.close(fig)
    print("D  fig-router-sensitivity-v5")


OUT.mkdir(parents=True, exist_ok=True)
fig_a()
fig_b()
fig_c()
fig_d()
