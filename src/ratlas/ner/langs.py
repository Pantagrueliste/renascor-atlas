"""Language codes (ISO 639-1 when the language has one) and a cheap stop-word language guess.

The guess is only a fallback (and a quality check of the declared language): it counts the closed-class words of
each language in a passage and answers when one language is clearly ahead. Deterministic, no model, no I/O.
"""
from __future__ import annotations

import re
from typing import Optional

_ISO3 = {
    'lat': 'la', 'latin': 'la', 'la': 'la', 'neo': 'la', 'lat-x-neo': 'la',
    'deu': 'de', 'ger': 'de', 'german': 'de', 'de': 'de', 'gmh': 'de', 'gml': 'de', 'goh': 'de', 'nds': 'nds', 'gsw': 'de',
    'fra': 'fr', 'fre': 'fr', 'french': 'fr', 'fr': 'fr', 'frm': 'fr', 'fro': 'fr',
    'eng': 'en', 'english': 'en', 'en': 'en', 'enm': 'en', 'ang': 'en',
    'ita': 'it', 'italian': 'it', 'it': 'it',
    'spa': 'es', 'esp': 'es', 'spanish': 'es', 'es': 'es', 'osp': 'es',
    'por': 'pt', 'portuguese': 'pt', 'pt': 'pt',
    'nld': 'nl', 'dut': 'nl', 'dutch': 'nl', 'nl': 'nl', 'dum': 'nl',
    'pol': 'pl', 'pl': 'pl', 'hrv': 'hr', 'hr': 'hr', 'ces': 'cs', 'cze': 'cs', 'cs': 'cs',
    'grc': 'grc', 'gre': 'el', 'ell': 'el', 'el': 'el', 'heb': 'he', 'he': 'he',
    'ota': 'ota', 'gez': 'gez', 'cat': 'ca', 'ca': 'ca', 'swe': 'sv', 'dan': 'da', 'hun': 'hu', 'ron': 'ro',
}
_UNDETERMINED = {'und', 'mul', 'zxx', 'unk', 'trad', 'x', '', 'none'}


def norm_lang(code: Optional[str]) -> Optional[str]:
    """'de-Latf' -> 'de', 'lat' -> 'la', 'frm' -> 'fr', 'und' -> None. Unknown codes are returned lower-cased (primary subtag)."""
    if not code:
        return None
    c = str(code).strip().lower().replace('_', '-')
    if not c:
        return None
    if c in _ISO3:
        return _ISO3[c]
    head = c.split('-')[0]
    if head in _UNDETERMINED:
        return None
    return _ISO3.get(head, head)


_SW = {
    'la': 'et in est non ad cum ut quod qui quae sed ab ex de per ne si nec aut sunt esse eius hoc haec ille ipse tibi mihi nos vos quam quo etiam enim autem tamen sit erat pro sub inter quia atque sicut',
    'de': 'und der die das den dem des ein eine einem einer ist nicht mit von zu auf für fuer auch sich als wie aber dass daß wir ich sie er es im an bei nach aus nur noch wenn oder hat haben sind wird werden war vnd dem dess jhr seyn sein',
    'fr': 'et le la les des du de un une est dans que qui pour par sur avec ne pas se ce cette ses son sa au aux il elle ils nous vous mais ou où comme plus tout fait être estre ont ont',
    'en': 'the and of to in is that it was for with as his he her she be by on at this which but not are from or had have they were their been will would so all my your our thy thou hath doth',
    'it': 'il lo la le gli di del della dei delle che e un una per con non si da in su come ma più piu sono è era hanno suo sua suoi alla nel nella al ai questo quello io tu lui noi voi essi sia anche',
    'es': 'el la los las de del un una y en que es por con no se su al lo como para más mas pero sus le ya o fue este sí si me muy también esta ese',
    'pt': 'o a os as de do da dos das um uma e em que é por com não se seu sua ao como para mais mas foi ele ela nos já também',
    'nl': 'de het een en van in is dat op te voor met zijn niet aan er maar om ook als dan bij of uit naar over nog wel hij zij ik wij',
}
_SWSETS = {k: frozenset(w for w in v.split() if len(w) > 1) for k, v in _SW.items()}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def guess_lang(text: str, min_hits: int = 8, margin: float = 1.6, min_share: float = 0.10) -> Optional[str]:
    """Language of a passage by closed-class words, or None when no language is clearly ahead (also None for languages not in the lists:
    Irish, Ge'ez, Ottoman... give few hits)."""
    ws = [w.lower() for w in _WORD.findall(text)]
    if len(ws) < 12:
        return None
    scores = {k: sum(1 for w in ws if w in s) for k, s in _SWSETS.items()}
    best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    (l1, s1), (l2, s2) = best[0], best[1]
    if s1 < min_hits or s1 < margin * s2 or s1 < min_share * len(ws):
        return None
    return l1
