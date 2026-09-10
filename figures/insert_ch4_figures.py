# -*- coding: utf-8 -*-
"""Wire figures B, C, D into Chapter 4 and swap 4.1 to v9. Anchored exact-string
edits: every anchor must appear exactly once or the script refuses."""
import io
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = ("C:/Users/Dharun prasanth/OneDrive/Documents/Projects/LLm_Agentic/"
        "dissertation/overleaf_live/")


def edit(path, pairs):
    t = io.open(ROOT + path, encoding='utf-8', newline='').read()
    for a, b in pairs:
        assert t.count(a) == 1, '%s: anchor x%d: %r' % (path, t.count(a), a[:60])
        t = t.replace(a, b)
    io.open(ROOT + path, 'w', encoding='utf-8', newline='').write(t)
    print('%-40s %d edits, %d chars' % (path, len(pairs), len(t)))


# ---------------------------------------------------------------- 4.6 / 4.5
E = 'chapters/experiments.tex'
OLD_DENSE = (
    "\\paragraph{Dense fine-tuning.} Domain adaptation and synthetic contrastive fine-tuning improve statute Recall@500 on the ten-query set from 0.34 to 0.54. That gain is a precondition and not a score: a candidate that is retained but wrong enlarges the denominator, so improved reachability lowers the score unless selection converts it. Separately, removing the dense contribution improves the benchmark score from 0.12093 to 0.15887. The two measurements move in opposite directions, and that contrast is the reason the row is reported; it is not a controlled comparison, since neither the query set nor the scored quantity is held fixed between them, and together they identify a mismatch between reach and selection without isolating its source. Section~\\ref{sec:results-learned-transfer} describes the kind of confusion involved, where a required \\texttt{Art. 41 Abs. 1 OR} is returned alongside its neighbouring \\texttt{Art. 41 Abs. 2 OR}, as Figure~\\ref{fig:worked-example} illustrates.\n")
NEW_DENSE = (
    "\\paragraph{Dense fine-tuning.} Domain adaptation and synthetic contrastive fine-tuning improve statute Recall@500 on the ten-query set from 0.34 to 0.54; removing the dense contribution altogether improves the benchmark score from 0.12093 to 0.15887 (Figure~\\ref{fig:reach-vs-answer}). Improving the component and deleting it both helped, measured on different things: a candidate that is retained but wrong enlarges the denominator, so reachability lowers the score unless selection converts it. This is not a controlled comparison, since neither the query set nor the scored quantity is held fixed, and together the two identify a mismatch between reach and selection without isolating its source. Section~\\ref{sec:results-learned-transfer} describes the confusion involved, where a required \\texttt{Art. 41 Abs. 1 OR} is returned alongside its neighbouring \\texttt{Art. 41 Abs. 2 OR} (Figure~\\ref{fig:worked-example}).\n"
    "\n\\begin{figure}[ht]\n\\centering\n"
    "\\includegraphics[width=\\textwidth]{fig-reach-vs-answer-v3.pdf}\n"
    "\\caption{Fine-tuning the dense retriever and removing it, on different evaluation sets. Left: statute Recall@500 on the ten validation queries before and after fine-tuning. Right: development benchmark Macro-F1 with the dense contribution present and removed. The panels share neither query set nor metric, so they are two observations rather than one before-and-after chain.}\n"
    "\\label{fig:reach-vs-answer}\n\\end{figure}\n")

edit(E, [
    ("{fig-admission-bar-v7.pdf}", "{fig-admission-bar-v9.pdf}"),
    (OLD_DENSE, NEW_DENSE),
    ("\\includegraphics[width=.74\\textwidth]{fig-failure-walls-imagegen-v3.png}",
     "\\includegraphics[width=\\textwidth]{fig-failure-walls-v7.pdf}"),
    ("First, nearly all missed statute gold are parent-article misses,",
     "First, 95\\% of missed statute gold are parent-article misses,"),
    ("\\caption{Descriptive failure decomposition on the ten-query diagnostic audit. Most missed statute gold are parent-article misses; where the article is reachable, the correct-paragraph and wrong-sibling cosine ranges overlap.}",
     "\\caption{Descriptive failure decomposition on the ten-query diagnostic audit. (a) 95\\% of missed statute gold are parent-article misses. (b) Where the article was reached, the observed minimum and maximum cosine similarity of correct paragraphs and of wrong siblings overlap over 0.55--0.69; these are observed ranges rather than confidence intervals, and the audit did not retain the number of paragraph pairs compared.}"),
])

# ---------------------------------------------------------------- 4.8 -----
R = 'chapters/routing.tex'
OLD_RES = (
    "\\paragraph{Result and its sensitivity.} The primary comparison is language-model routing against keyword routing through identical code, reported in Table~\\ref{tab:router-downstream}: \\(+0.02111\\) on the pre-declared split against \\(+0.00357\\) for the keyword control. A post-hoc sensitivity analysis re-splits the same archived predictions 100 times and has mean \\(+0.01523\\); the pre-declared split lies above that range (Appendix~\\ref{app:appendix-methodnotes}). The plain reading is that the pre-declared split was a favourable draw. It is reported as the protocol result because it was fixed before the comparison was run, and reporting only the more flattering of two figures chosen afterwards would defeat the purpose of declaring it. Both point the same way, and the disagreement between them is about magnitude rather than direction.\n")
NEW_RES = (
    "\\paragraph{Result and its sensitivity.} The primary comparison is language-model routing against keyword routing through identical code (Table~\\ref{tab:router-downstream}, Figure~\\ref{fig:router-sensitivity}): \\(+0.02111\\) on the pre-declared split against \\(+0.00357\\) for the keyword control, and a mean of \\(+0.01523\\) across 100 post-hoc re-splits of the same archived predictions, which the pre-declared split lies above. The plain reading is that the pre-declared split was a favourable draw. It is reported as the protocol result because it was fixed before the comparison was run, and reporting only the more flattering figure chosen afterwards would defeat the purpose of declaring it. Both point the same way; the disagreement is about magnitude rather than direction.\n"
    "\n\\begin{figure}[ht]\n\\centering\n"
    "\\includegraphics[width=\\textwidth]{fig-router-sensitivity-v6.pdf}\n"
    "\\caption{The router's effect on Macro-F1 measured three ways: the pre-declared 497/496 split, 100 post-hoc re-splits of the same archived predictions, and the keyword control through identical code. The outer rows carry paired percentile-bootstrap 95\\% confidence intervals; the middle row carries the empirical 95\\% range across re-splits, a different quantity. Shading marks the interval the keyword control alone reaches.}\n"
    "\\label{fig:router-sensitivity}\n\\end{figure}\n")
edit(R, [(OLD_RES, NEW_RES)])

# ---------------------------------------------------------------- appendix -
A = 'appendices/appendix-methodnotes.tex'
OLD_APP = (
    "\\section{Router split sensitivity}\n\\begin{figure}[ht]\n\\centering\n"
    "\\includegraphics[width=.74\\textwidth]{fig-router-sensitivity-v3.pdf}\n"
    "\\caption{The pre-declared routing delta against 100 post-hoc random 497/496 splits, recomputed over the archived router predictions without further model calls. The pre-declared split lies above the empirical range of the re-splits.}\n"
    "\\label{fig:router-sensitivity}\n\\end{figure}\n\nThe empirical")
NEW_APP = (
    "\\section{Router split sensitivity}\nFigure~\\ref{fig:router-sensitivity} in Section~\\ref{sec:routing-study} plots the pre-declared delta against 100 post-hoc random 497/496 splits, recomputed over the archived router predictions without further model calls. The empirical")
edit(A, [(OLD_APP, NEW_APP)])
print('done')
