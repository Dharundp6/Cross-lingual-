# -*- coding: utf-8 -*-
"""Insert fig-admission-bar-v7 into Section 4.6, after the paragraph whose
numbers it plots.

The artwork carries no prose, so every qualification lives in the caption:
 - the 27-in-100 line is the recorded court-side working figure, not pi*
 - the extractor's 0.970 is on training queries, firing for 9.2% of them
 - the two selectors are ten-query diagnostics outside the score lineage
 - no per-addition precision exists for anchored co-citation, so its
   absence from the chart is a gap in the archive, not a failed result
"""
import io
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = ("C:/Users/Dharun prasanth/OneDrive/Documents/Projects/LLm_Agentic/"
     "dissertation/overleaf_live/chapters/experiments.tex")

ANCHOR = ("Both fall below the working threshold of approximately 0.27 the "
          "project had recorded for court-side additions, by factors of "
          "nineteen and about seven.\n")

FIG = """
\\begin{figure}[ht]
\\centering
\\includegraphics[width=\\textwidth]{fig-admission-bar-v7.pdf}
\\caption{Recorded probability that an added citation is correct, set against
the empirical court-side working threshold. These are the only three
components for which a per-addition precision was retained; none exists for
anchored co-citation, which is therefore absent here even though
Section~\\ref{sec:experiments-proposed-comparison} records it improving the
baseline. The two selectors are ten-query court diagnostics run outside the
reproducible score lineage, while the extractor's 0.970 is measured on the
training queries, where it returns anything for only 9.2\\% of them, and is
shown for contrast rather than because the court-side line governs it. That
line is the figure the project recorded at the operating point then in use,
not a reconstructed per-query \\(\\pi^{*}\\).}
\\label{fig:admission-bar}
\\end{figure}
"""

t = io.open(P, encoding='utf-8', newline='').read()
assert t.count(ANCHOR) == 1, 'anchor x%d' % t.count(ANCHOR)
assert 'fig:admission-bar' not in t, 'already inserted'
t = t.replace(ANCHOR, ANCHOR + FIG)

# reference it from the sentence it illustrates
OLD = ("by factors of nineteen and about seven.")
NEW = ("by factors of nineteen and about seven "
       "(Figure~\\ref{fig:admission-bar}).")
assert t.count(OLD) == 1
t = t.replace(OLD, NEW)

BANNED = ['0.29494', 'held-out', 'reconstructed \\(\\pi^{*}\\) threshold']
for s in BANNED:
    assert s not in FIG, 'BANNED: ' + s
for s in ['0.970', '9.2', 'ten-query', 'not a reconstructed per-query',
          'anchored co-citation']:
    assert s in FIG, 'LOST: ' + s

io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('inserted fig:admission-bar; file now %d chars' % len(t))
