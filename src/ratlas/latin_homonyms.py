"""Latin place names that name two or more places (v2.4; pure functions, no I/O).

Why. The independent check of v2.3 found 290 mentions of ``Lugduni`` / ``Lugdunum`` / ``Lugdunenses`` / ``Lugduno`` / ``Lugdunensi`` in the letters of the Dutch Republic
(ckcc_epistolarium) placed on Lyon, a majority of them meaning Leiden ('Lugduni Batavorum', 'Academia Lugdunensi', 'Lugdunenses burgemeesters'). The exonym table had given the
bare Latin forms to Lyon, and the Latin path's guard (``LATIN_HOMONYM_CTX``) let Lyon through because the letters name other French places too. The rule is now one for every
Latin name that is known to belong to several places (``FAMILIES``: Lugdunum = Lyon or Leiden, Traiectum = Utrecht, Maastricht or Frankfurt (Oder), Augusta = Augsburg, Turin or
Trier, Hadria = the Adriatic Sea or the town of Adria):

* each candidate has a **cue**, a regular expression on the words around the mention ('Batavorum', 'Rhodani', 'Mosam' ...): explicit support;
* a candidate that **needs a cue** (Lyon in a Dutch letter, the town of Adria) is placed only with explicit support and without explicit support for another one;
* a candidate that does not need one (Leiden, Utrecht, Augsburg, the sea) is placed when the passage names it, or, with no cue at all, when the **places of the same documents**
  support it (``DOC_CTX_MIN``) clearly more than the other candidates (``DOC_CTX_MARGIN``);
* cues for two candidates in the mentions of one place, or no support: the place stays unplaced (``why`` = ``latin_homonym``).

The decision is per local place (a surface form such as 'Lugduni' in one edition): the mentions of one place all get the one answer, so a form that is Leiden in 17 letters
and Lyon in 5 (cues for both) is not placed at all. Precision first: a wrong point is worse than a missing one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from . import latin_names as LN

#: the documents' places must support a candidate with this much (the context terms of the GeoNames score, see ``Geocoder.doc_ctx_at``) ...
DOC_CTX_MIN = 0.35
#: ... and more than every other candidate by this margin
DOC_CTX_MARGIN = 0.2
#: a candidate named by the passage in at least this share of the mentions is placed on its cue alone (below it, the documents must support it as well)
CUE_SHARE_MIN = 0.15
#: a candidate named in half of the passages or more is placed even when the documents' places point elsewhere
CUE_SHARE_STRONG = 0.5
#: a candidate that needs a cue is placed on the cue alone only when this share of the passages names it
CUE_NEEDS_SHARE = 0.8
#: ... and an explicit candidate that the documents contradict (another one is supported by this much more) is not placed
DOC_CONTRADICTION = 0.6


@dataclass(frozen=True)
class Cand:
    label: str
    lat: float
    lon: float
    radius: float
    kind: str                 # 'city' | 'area' | 'sea'
    cue: Optional[re.Pattern]
    needs_cue: bool = False   # placed only with explicit support in the passage
    ctx_free: bool = False    # the documents' places say nothing about it (a sea): placed when it is the only candidate without a need for a cue


@dataclass(frozen=True)
class Family:
    name: str
    pattern: re.Pattern      # on the folded key of the whole name (``latin_names.fold_key``)
    cands: tuple


def _c(rx):
    return re.compile(rx, re.I)


# fmt: off
FAMILIES = [
    Family('lugdunum', _c(r'^lugdun(um|i|o|ensis|ensi|enses|ensium|ensibus|ensem|ensia|ensibus)?$'), (
        Cand('Leiden', 52.160, 4.494, 8.0, 'city', _c(r'batav|leid|leyd|holland|nederl|hogeschool|rijnsburg|rhenoburg|ordines (generales|hollandiae)|\bL\.?\s?B\.?\b|bibliotheca lugd|academi\w* lugdun')),
        Cand('Lyon', 45.764, 4.836, 12.0, 'city', _c(r'rhod|rhon[ea]\b|saone|sauconn|\barar\b|ararim|segusi|\bgall|francia|franc[eo]\b|lyon|lione|gratianop|massili|vienn|delphin|narbon|savoi|sabaud|archiepisc'), needs_cue=True),
    )),
    Family('traiectum', _c(r'^(traiect|trajec)(um|i|o|a|ensis|ensi|enses)?$'), (
        Cand('Utrecht', 52.091, 5.122, 8.0, 'city', _c(r'rhen|ultra[ij]ect|batav|utrecht|vtrecht|voet|acad|univers|trajectin|ultraiectin')),
        Cand('Maastricht', 50.851, 5.690, 8.0, 'city', _c(r'\bmos[ae]\b|mosam|maas|tungr|limburg|maestricht'), needs_cue=True),
        Cand('Frankfurt (Oder)', 52.347, 14.550, 8.0, 'city', _c(r'viadr|\boder\b|brandenb|marchi'), needs_cue=True),
    )),
    Family('augusta', _c(r'^augusta$'), (
        Cand('Augsburg', 48.371, 10.898, 12.0, 'city', _c(r'vindel|suev|bavar|\bboi|\blyc|augspurg|augsburg')),
        Cand('Turin', 45.070, 7.687, 10.0, 'city', _c(r'taurin|pedemont|sabaud|torino'), needs_cue=True),
        Cand('Trier', 49.750, 6.637, 10.0, 'city', _c(r'trever|trevir|mosell|treves'), needs_cue=True),
    )),
    Family('hadria', _c(r'^hadri(a|e|am|as|ae|aticum)$'), (
        Cand('Adriatic Sea', 42.8, 15.5, 250.0, 'sea', None, ctx_free=True),
        Cand('Adria', 45.050, 12.050, 10.0, 'city', _c(r'rhodig|polesin|atriae|atriani|rovig|episcop\w* (atri|hadri)|ostig'), needs_cue=True),
    )),
]
# fmt: on


def family_of(name) -> Optional[Family]:
    """The family of a place name (the whole name, folded), or None. 'Lugduni' -> lugdunum; 'Lugduni Batavorum' -> None (explicit: the exonym table)."""
    if not isinstance(name, str):
        return None
    k = LN.fold_key(name.replace('.', ' '))
    k = ' '.join(k.split())
    for f in FAMILIES:
        if f.pattern.match(k):
            return f
    return None


@dataclass
class Decision:
    cand: Optional[Cand]
    reason: str                 # why: 'cue' | 'documents' | 'free' | a refusal 'mixed_cues' | 'no_support' | 'contradicted' | 'needs_cue'
    n_cue: dict                 # {label: number of mentions whose passage matches its cue}
    n_windows: int
    ctx: dict                   # {label: support from the documents' places}


def decide(family: Family, windows: Sequence[str], ctx: dict) -> Decision:
    """Choose the candidate of ``family`` for a place whose mentions have the passages ``windows`` and whose documents support each candidate with ``ctx[label]``.
    ``ctx`` may lack a label (no cloud: 0.0)."""
    wins = [w for w in windows if isinstance(w, str) and w]
    n_cue = {c.label: sum(1 for w in wins if c.cue is not None and c.cue.search(w)) for c in family.cands}
    cx = {c.label: float(ctx.get(c.label, 0.0)) for c in family.cands}
    explicit = [c for c in family.cands if n_cue[c.label] > 0]
    n = len(wins)

    def out(c, why):
        return Decision(c, why, n_cue, n, cx)
    if len(explicit) >= 2:
        return out(None, 'mixed_cues')
    if len(explicit) == 1:
        c = explicit[0]
        other_best = max([cx[o.label] for o in family.cands if o is not c] + [0.0])
        share = n_cue[c.label] / n if n else 1.0
        if cx[c.label] + DOC_CONTRADICTION <= other_best and share < CUE_SHARE_STRONG:
            return out(None, 'contradicted')
        if c.needs_cue:
            # a candidate that is placed on its cue alone (Lyon in a Dutch letter, Turin for 'Augusta') must be named in nearly every passage: 'Augusta' is also the adjective of
            # Majesty and the History, and the other twenty passages of a form are rarely all the one place
            ok = share >= CUE_NEEDS_SHARE or (share >= CUE_SHARE_STRONG and cx[c.label] >= DOC_CTX_MIN)
        else:
            ok = share >= CUE_SHARE_MIN or c.ctx_free or cx[c.label] >= DOC_CTX_MIN
        if ok:
            return out(c, 'cue')
        return out(None, 'no_support')
    # no passage names a candidate
    free = [c for c in family.cands if not c.needs_cue]
    ctxfree = [c for c in free if c.ctx_free]
    if ctxfree and len(free) == len(ctxfree) == 1:
        return out(ctxfree[0], 'free')                      # the sea: the only candidate that does not need a cue
    ranked = sorted((c for c in free if not c.ctx_free), key=lambda c: -cx[c.label])
    if ranked:
        best = ranked[0]
        rest = max([cx[o.label] for o in family.cands if o is not best] + [0.0])
        if cx[best.label] >= DOC_CTX_MIN and cx[best.label] - rest >= DOC_CTX_MARGIN:
            return out(best, 'documents')
    return out(None, 'needs_cue' if any(c.needs_cue for c in family.cands) else 'no_support')
