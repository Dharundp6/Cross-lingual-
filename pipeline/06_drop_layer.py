# -*- coding: utf-8 -*-
"""
CELL 6 — Drop layer: operative-vs-dispositif content classification.

Replicates v305/v307/v311/v312 mechanism. Drop court refs whose paragraph text
is a NON-DOCTRINAL content type:
  1. Issue-framing transition ("Streitig und zu prüfen ist", "Zu prüfen bleibt")
  2. Party-argument summary ("Beschwerdeführer macht geltend", "rügt", "se plaint")
  3. Pure fact narrative (dates + amounts only, no doctrinal vocabulary)
  4. Case-specific procedural admissibility (Eventualbegründung, "nicht einzugehen")
  5. Gehör digression with different sub-grievances

All rules grounded in val analysis. No LLM API needed — pure regex over corpus text.
"""

# --- Drop pattern regexes (operative-vs-dispositif content class) ---
DROP_PAT_ISSUE_FRAMING = re.compile(
    r'(Streitig\s+(und\s+zu\s+prüfen\s+ist|ist)|'
    r'Zu\s+prüfen\s+bleibt|'
    r'Es\s+bleibt\s+zu\s+untersuchen|'
    r'vorab\s+die\s+von\s+Amtes\s+wegen\s+zu\s+prüfende\s+Rechtsfrage|'
    r'^[0-9]+\.[0-9]?\.?\s*Streitig)',
    re.IGNORECASE
)

DROP_PAT_PARTY_SUMMARY = re.compile(
    r'(Der\s+Beschwerdeführer\s+(macht\s+geltend|rügt|bemängelt|kritisiert|beanstandet)|'
    r'Sie\s+macht\s+geltend|'
    r'Die\s+Beschwerdeführerin\s+(macht\s+geltend|rügt|bemängelt)|'
    r'Le\s+recourant\s+(se\s+plaint|critique|invoque|fait\s+valoir)|'
    r'La\s+recourante\s+(critique|se\s+plaint|fait\s+valoir)|'
    r'Zur\s+Hauptsache\s+erblickt\s+der\s+Kläger)',
    re.IGNORECASE
)

DROP_PAT_CASE_PROC = re.compile(
    r'(in\s+der\s+Eventualbegründung|'
    r'mangels\s+Tatsachenbehauptungen|'
    r'ist\s+nicht\s+einzugehen|'
    r'liess\s+die\s+Vorinstanz\s+offen|'
    r'Die\s+Kritik\s+der\s+Vorinstanz\s+am\s+rückweisenden\s+Urteil)',
    re.IGNORECASE
)

# Pure fact narrative: short text with dates+amounts but no doctrinal "Art." references
DATE_AMT_PAT = re.compile(r'(\d{1,2}\.\s*[A-Z][a-zü]+\s+\d{4}|CHF\s*[\d\']+|€\s*[\d\']+|\d+[%‰])', re.IGNORECASE)
ART_PAT_IN_TEXT = re.compile(r'\bArt\.\s*\d+', re.IGNORECASE)

def is_dispositif_class(text):
    """Return classification name if text matches a drop class, else None."""
    if not text or len(text) > 700:  # Long substantive paragraphs are safe
        return None
    if DROP_PAT_ISSUE_FRAMING.search(text):
        return 'issue_framing'
    if DROP_PAT_PARTY_SUMMARY.search(text):
        return 'party_summary'
    if DROP_PAT_CASE_PROC.search(text):
        return 'case_procedural'
    # Pure fact narrative: many dates/amounts, no Art. references
    n_dates = len(DATE_AMT_PAT.findall(text))
    n_arts = len(ART_PAT_IN_TEXT.findall(text))
    if n_dates >= 3 and n_arts == 0 and len(text) < 500:
        return 'fact_narrative'
    return None

# --- Apply drops to reranked per_q ---
n_drops_total = 0
drop_breakdown = Counter()
ranked_per_q_dropped = {}

for qid, ranked in ranked_per_q.items():
    new_ranked = []
    for cit, score in ranked:
        if court_pat.match(cit):
            text = court_text_concat(cit)
            cls = is_dispositif_class(text)
            if cls is not None:
                drop_breakdown[cls] += 1
                n_drops_total += 1
                continue
        new_ranked.append((cit, score))
    ranked_per_q_dropped[qid] = new_ranked

print(f'\nDrop layer applied:')
print(f'  total drops: {n_drops_total}')
for cls, n in drop_breakdown.most_common():
    print(f'    {cls:<20} {n}')

ranked_per_q = ranked_per_q_dropped
