# -*- coding: utf-8 -*-
"""Chapter 4 figures, narrated layout supplied by the author.

Each figure gains a plain-language reading under a rule: a bold takeaway line
and a grey qualifying note. Some carry a standfirst above the chart. The LaTeX
caption is trimmed to the formal decoding so the two do not restate each other.

The in-figure title from the supplied design is NOT reproduced: \\caption{}
already emits "Figure 4.N:", so an artwork title numbers the figure twice on
the page.

FIVE CLAIMS FROM THE SUPPLIED DESIGN WERE CORRECTED. Each was checked against
the recorded values before being redrawn:

D  "Whiskers show the observed spread" -- false for two of the three rows.
   The pre-declared and keyword rows are paired percentile-bootstrap 95% CIs
   (Table 4.5); the re-split row is an empirical 95% range across re-splits.
   Collapsing them into one phrase asserts they are the same quantity.

D  "recovers a fifth of what remains" -- 0.00357 / 0.01523 = 0.234, a quarter.

C  "where the two distributions overlap" -- no distributions were retained.
   Section 4.7: the audit kept the two ranges but not the number of paragraph
   pairs compared. "Ranges" is the most the archive supports.

C  "Almost every miss happens because the parent article is absent" -- causal
   phrasing on a descriptive audit. Redrawn as "is one where".

B  "better by a third" -- 0.15887 / 0.12093 = 1.314, so "about a third".

Checked and kept unchanged from the supplied design:
   +0.20   = 0.54 - 0.34
   +0.038  = 0.15887 - 0.12093 = 0.03794
   28%     = (0.02111 - 0.01523) / 0.02111 = 0.279
   "one in five more gold statutes" -- Recall@500 rises 0.20 of the gold set

Type stays at 8.5 pt and above regardless of styling: that came out of the
venue guidance as a legibility floor, not a matter of taste.
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT = Path("C:/Users/Dharun prasanth/OneDrive/Documents/Projects/"
           "LLm_Agentic/dissertation/overleaf_live/figures")
PREV = Path(__file__).resolve().parent

INK = "#161F27"
SLATE = "#3E5060"
GREY = "#5F6B76"
MUTE = "#8A95A0"
BLUE = "#2364AA"
RED = "#C8553D"
TRACK = "#ECECE9"
BAND = "#E8E8E5"
RULE = "#C9CFD5"
W = 5.56
MONO = "DejaVu Sans Mono"

FS_NAME, FS_VAL, FS_AXIS, FS_TICK = 9.5, 9.5, 9.0, 8.5
FS_LEAD, FS_TAKE, FS_NOTE = 9.0, 9.0, 8.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


def bare(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, colors=GREY, labelsize=FS_TICK, pad=3)


def footer(fig, H, rule_in, take, note):
    """Bottom rule, bold takeaway, grey note. Text wraps to the figure width.

    Wrap widths are measured, not guessed: DejaVu Sans averages about 0.60 em
    bold and 0.55 em regular, so at 5.56 in (400 pt) a 9 pt bold line holds
    roughly 78 characters and an 8 pt regular line roughly 94.
    """
    take = textwrap.fill(take, 78)
    note = textwrap.fill(note, 94)
    y = 1 - rule_in / H
    fig.add_artist(plt.Line2D([.012, .988], [y, y], color=RULE, lw=.9,
                              transform=fig.transFigure))
    fig.text(.012, 1 - (rule_in + .12) / H, take, fontsize=FS_TAKE,
             fontweight="bold", color=INK, ha="left", va="top",
             linespacing=1.45)
    n = take.count(chr(10)) + 1
    fig.text(.012, 1 - (rule_in + .24 + n * .17) / H, note, fontsize=FS_NOTE,
             color=MUTE, ha="left", va="top", linespacing=1.45)


def save(fig, stem):
    fig.savefig(OUT / (stem + ".pdf"))
    fig.savefig(OUT / (stem + ".png"), dpi=340)
    fig.savefig(PREV / (stem + "-preview.png"), dpi=150)
    plt.close(fig)
    print("  " + stem)


# --------------------------------------------------------------- 4.1 -------
def fig_admission():
    H = 2.72
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.300, 1 - 1.02 / H, .455, .70 / H])

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
        ax.text(v - 2.0 if inside else v + 2.4, y, vtxt, zorder=5,
                fontsize=FS_VAL, family=MONO, fontweight="bold",
                color="white" if inside else colour,
                ha="right" if inside else "left", va="center")
        ax.text(1.035, y, ann, transform=gt, clip_on=False, zorder=4,
                fontsize=FS_TICK, color=GREY, ha="left", va="center")
    ax.axvline(27, color=INK, lw=1.0, ls=(0, (3.2, 2.4)), zorder=6)
    ax.text(27, 2.88, "working threshold 27", fontsize=FS_TICK, color=INK,
            ha="center", va="center", clip_on=False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Correct citations out of every 100 added",
                  fontsize=FS_AXIS, color=INK, labelpad=5)
    bare(ax)

    # The earlier line read "only explicit extraction clears it", which invites
    # the reading that the extractor was measured against this threshold. It
    # was not: the 27-in-100 figure is court-side, and the extractor is a
    # statute-side component scored on training queries. The claim is now
    # confined to the two selectors the threshold actually governs, and the
    # extractor stays as a labelled high-precision contrast.
    footer(fig, H, 1.58,
           "Neither tested court-side selector clears the recorded working "
           "threshold.",
           "Figures are pooled across components and query sets, so the "
           "per-query bar $\\pi^{*}=F_1/2$ of Equation 4.4 is not evaluable "
           "against them. The extractor is a statute-side component measured "
           "on training queries, shown here as a high-precision contrast "
           "rather than as a candidate for court-side selection.")
    save(fig, "fig-admission-bar-v10")


# ----------------------------------------------------------------- B -------
def fig_reach():
    H = 3.30
    fig = plt.figure(figsize=(W, H))
    fig.text(.012, 1 - .12 / H, textwrap.fill(
        "Finding more of the right statutes did not make the system answer "
        "better. The two panels measure different things on different sets.",
        82), fontsize=FS_LEAD, color=SLATE, ha="left", va="top",
        linespacing=1.5)

    spec = [
        (.115, "1 · Fine-tuning the retriever", "ten-query set",
         "+0.20", "Statute Recall@500", .34, .54, "0.34", "0.54",
         "before", "after", BLUE, (.26, .62), [.3, .4, .5, .6]),
        (.620, "2 · Removing the retriever", "development benchmark",
         "+0.038", "Macro-F1", .12093, .15887, "0.12093", "0.15887",
         "dense in", "dense out", RED, (.108, .172), [.12, .14, .16]),
    ]
    for (x0, title, sub, delta, ylab, v0, v1, t0, t1, xl0, xl1, colour, ylim,
         yt) in spec:
        ax = fig.add_axes([x0, 1 - 1.98 / H, .325, .90 / H])
        ax.plot([0, 1], [v0, v1], color=colour, lw=1.8, zorder=3)
        ax.scatter([0, 1], [v0, v1], s=34, color=colour, zorder=4)
        for xv, vv, tt in ((0, v0, t0), (1, v1, t1)):
            ax.annotate(tt, (xv, vv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=FS_VAL,
                        family=MONO, fontweight="bold", color=colour)
        ax.set(xlim=(-.42, 1.42), ylim=ylim, xticks=[0, 1], yticks=yt)
        ax.set_xticklabels([xl0, xl1], fontsize=FS_TICK, color=GREY)
        ax.set_ylabel(ylab, fontsize=FS_AXIS, color=INK, labelpad=7)
        ax.tick_params(axis="y", labelsize=FS_TICK, colors=GREY, length=2)
        bare(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(RULE)

        fig.text(x0 - .075, 1 - .70 / H, title, fontsize=FS_NAME,
                 fontweight="bold", color=INK, ha="left", va="center")
        fig.text(x0 - .075, 1 - .90 / H, sub, fontsize=FS_NOTE, color=MUTE,
                 ha="left", va="center")
        fig.text(x0 + .345, 1 - .90 / H, delta, fontsize=FS_NOTE,
                 family=MONO, fontweight="bold", color="white", ha="right",
                 va="center",
                 bbox=dict(boxstyle="round,pad=.38", facecolor=colour,
                           edgecolor="none"))

    footer(fig, H, 2.30,
           "Better retrieval did not translate into better answers — the "
           "component that improved is the one the system was better off "
           "without.",
           "Coverage rose by one in five of the gold statutes; removing the "
           "retriever raised the benchmark score by about a third. Different "
           "evaluation sets and metrics: two separate observations, not one "
           "before/after chain.")
    save(fig, "fig-reach-vs-answer-v3")


# ----------------------------------------------------------------- C -------
def fig_walls():
    H = 3.49
    fig = plt.figure(figsize=(W, H))

    fig.text(.012, 1 - .16 / H, "(a)  Where the missed statute gold went",
             fontsize=FS_AXIS, fontweight="bold", color=INK, ha="left",
             va="center")
    axa = fig.add_axes([.012, 1 - .60 / H, .800, .26 / H])
    axa.set(xlim=(0, 100), ylim=(0, 1))
    axa.barh(.5, 95, height=1.0, color=BLUE)
    axa.barh(.5, 5, left=95, height=1.0, color=TRACK)
    axa.text(1.8, .5, "Parent article absent   95%", fontsize=FS_TICK,
             fontweight="bold", color="white", ha="left", va="center")
    axa.text(102.5, .5, "present 5%", fontsize=FS_TICK, color=GREY,
             ha="left", va="center", clip_on=False)
    axa.axis("off")

    fig.text(.012, 1 - .92 / H, "(b)  Where the article was reached",
             fontsize=FS_AXIS, fontweight="bold", color=INK, ha="left",
             va="center")
    axb = fig.add_axes([.235, 1 - 1.86 / H, .615, .72 / H])
    axb.fill_betweenx([-.62, 1.58], .55, .69, color=BAND, zorder=0)
    for y, lo, hi, lab, colour in [(1, .51, .69, "Correct paragraph", BLUE),
                                   (0, .55, .71, "Wrong sibling", RED)]:
        axb.plot([lo, hi], [y, y], color=colour, lw=1.6, zorder=3)
        for x in (lo, hi):
            axb.plot([x, x], [y - .16, y + .16], color=colour, lw=1.5,
                     zorder=3)
        axb.text(-.028, y, lab, transform=axb.get_yaxis_transform(),
                 fontsize=FS_NAME, color=INK, ha="right", va="center",
                 clip_on=False)
        axb.text(1.028, y, "%.2f\u2013%.2f" % (lo, hi),
                 transform=axb.get_yaxis_transform(), clip_on=False,
                 fontsize=FS_TICK, family=MONO, fontweight="bold",
                 color=colour, ha="left", va="center")
    axb.text(.62, -.60, "overlap 0.55\u20130.69", fontsize=FS_NOTE,
             color=MUTE, ha="center", va="center")
    axb.set(xlim=(.487, .733), ylim=(-1.05, 1.62), yticks=[])
    axb.set_xticks([.50, .55, .60, .65, .70])
    axb.set_xlabel("Observed cosine similarity to the question",
                   fontsize=FS_AXIS, color=INK, labelpad=4)
    bare(axb)

    footer(fig, H, 2.36,
           "Almost every miss is one where the parent article is absent "
           "\u2014 and when it is present, similarity cannot separate the "
           "correct paragraph from its siblings.",
           "Bars are the observed minimum and maximum cosine similarity, not "
           "confidence intervals; the shaded band marks where the two ranges "
           "overlap. The audit did not retain the number of paragraph pairs "
           "compared.")
    save(fig, "fig-failure-walls-v7")


# ----------------------------------------------------------------- D -------
def fig_router():
    H = 2.95
    fig = plt.figure(figsize=(W, H))
    fig.text(.012, 1 - .12 / H, textwrap.fill(
        "How much the router adds to Macro-F1, measured three ways. The "
        "further right, the bigger the claimed gain.", 82),
        fontsize=FS_LEAD, color=SLATE, ha="left", va="top", linespacing=1.5)

    ax = fig.add_axes([.325, 1 - 1.42 / H, .530, .78 / H])
    rows = [(2, "Pre-declared split", .02111, .01175, .03074, RED, "D"),
            (1, "100 post-hoc re-splits", .01523, .01139, .01932, BLUE, "o"),
            (0, "Keyword control", .00357, .00020, .00712, GREY, "s")]

    ax.axvspan(0, .00712, color=BAND, zorder=0)
    ax.text(.00712 / 2, 2.90, "keyword-control range", fontsize=FS_NOTE,
            color=MUTE, ha="center", va="center", clip_on=False)
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    gt = ax.get_yaxis_transform()
    for y, name, v, lo, hi, colour, mk in rows:
        ax.plot([lo, hi], [y, y], color=colour, lw=1.5, zorder=3)
        for x in (lo, hi):
            ax.plot([x, x], [y - .12, y + .12], color=colour, lw=1.5,
                    zorder=3)
        ax.scatter([v], [y], marker=mk, s=40, color=colour, zorder=4)
        ax.text(-.032, y, name, transform=gt, clip_on=False,
                fontsize=FS_NAME, color=INK, ha="right", va="center")
        ax.text(1.030, y, "%+.5f" % v, transform=gt, clip_on=False,
                fontsize=FS_TICK, family=MONO, fontweight="bold",
                color=colour, ha="left", va="center")
    ax.set(xlim=(-.0018, .0352), ylim=(-.62, 2.62), yticks=[])
    ax.set_xticks([0, .01, .02, .03])
    ax.set_xticklabels(["0", "+0.01", "+0.02", "+0.03"])
    ax.set_xlabel("Change in Macro-F1 on the scored half", fontsize=FS_AXIS,
                  color=INK, labelpad=5)
    bare(ax)

    footer(fig, H, 1.82,
           "The headline gain shrinks by 28% once the split is redrawn, and a "
           "keyword baseline with no router recovers a quarter of what "
           "remains.",
           "The outer two rows show paired percentile-bootstrap 95% "
           "confidence intervals; the middle row shows the empirical 95% "
           "range across re-splits. The vertical rule marks no change; "
           "shading marks the range the keyword control alone reaches.")
    save(fig, "fig-router-sensitivity-v6")


OUT.mkdir(parents=True, exist_ok=True)
print("writing:")
fig_admission()
fig_reach()
fig_walls()
fig_router()
