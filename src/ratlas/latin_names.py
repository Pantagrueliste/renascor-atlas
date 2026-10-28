"""Latin place-name lemmatiser: from an inflected Latin toponym to ranked nominative candidates (v2.3 helper).

Why. About 4,100 unplaced mentions are Latin case forms of places the atlas already knows under their nominative or under
a vernacular name: Leidam (Leiden, 676), Hagam (The Hague, 461), Rhenum / Rheni / Rheno (the Rhine), Budam, Mosellam, Colmariam,
Transilvaniam, Scotiam ... A naive de-inflection (-am -> -a, -um -> -us ...) finds them, but also matches French or Spanish
words (Moree, Delfos), so the rules are only to be applied to Latin context (is_latin_context) and their output is only a
*list of candidates*: the caller keeps a candidate only when it is found in a lexicon of known Latin names (Wikidata 'la'
labels, atlas names; see LatinLexicon and resolve). Precision first: nothing here decides a placement.

Public API (all pure Python, no dependency; the optional Collatinus path is lazy and off by default)::

    lemmatize(form, *, max_candidates=6, min_score=0.4, include_self=True, adjective_bases=True, collatinus=False)
        -> list[Candidate]            # ranked, Candidate(lemma, score, rule); score in [0, 1]
    nominatives(form, **kw) -> list[str]
    lemmatize_in_context(form, latin_context, **kw) -> list[Candidate]   # [] when latin_context is false
    is_latin_context(coll_slug=None, lang=None, doc_language=None, *, edition_share=None, min_share=0.5) -> bool
    latin_context_level(coll_slug=None, lang=None, doc_language=None, *, min_share=0.5) -> 'latin' | 'mixed' | 'non_latin' | 'unknown'
    resolve_in_context(form, lexicon, coll_slug=None, lang=None, doc_language=None) -> list[Resolved]   # the one-call entry point
    edition_latin_share(coll_slug) -> Optional[float]
    latin_share_from_languages(languages) -> float                 # from an entry's 'languages' list
    is_latin_language(code) -> Optional[bool]                      # True / False, None when unknown ('?', 'und', '')
    fold_key(text) -> str                                          # accent / spelling-variant folding for lexicon keys
    LatinLexicon(names_or_mapping); resolve(form, lexicon, ...) -> list[Resolved]   # candidates found in a lexicon
    collatinus_available() -> bool; collatinus_lemmas(form) -> list[str]            # optional, GPL data, see below

Rules (Candidate.rule), by declension. Stems shorter than 2 letters are never stripped.

* 1st (-a, f.): -am / -iam (acc.) -> -a / -ia; -ae / -iae (gen., dat., locative: Romae, Lutetiae) -> -a / -ia; medieval -e / -ie
  (Colmarie) -> -a / -ia; plural -as, -is, -arum -> -ae (Athenas, Athenis, Athenarum); -ae also kept as is (plural-only names:
  Athenae, Aquae).
* 2nd (-us, -er, -um): -um / -i / -ii / -o / -io -> -us or -um (Rhenum, Rheni, Rheno -> Rhenus; Lugduno -> Lugdunum), the neuter
  being preferred for stems of towns in -dunum, -burgum, -acum, -furt-, -mont-, -dur-, -ent-, -orat-, -lan-, -in-; -ri -> -er
  (Alexandri); plural -os, -orum, -is -> -i (Delphos, Batavorum; the weak -os -> -i is scored 0.45).
* 3rd: -ntem / -ntis / -nte -> -ns (Montem -> Mons); -onem / -onis / -one and -inem / -inis / -ine -> -o (Narbonem -> Narbo,
  Carthaginem -> Carthago); -im / -em / -e / -is -> -is (Neapolim -> Neapolis, Tiberim -> Tiberis); -atem -> -as; -cem / -gem ->
  -x; -rem -> -r; -ibus -> -es; low-scored extras (-is -> -s, -is -> -a, -um -> -es).
* Adjectives: -ensis / -ensem / -ensi / -ense / -ensium -> -ensis, plus the base place the adjective is built on (Hafniensis ->
  Hafnia, Londinensis -> Londinium; adjective_bases); -anus / -inus / -icus ... decline as 1st/2nd and are covered above.
* Multiword (up to the full string): leading prepositions are dropped (in, ad, apud ...); the head noun is inflected and the
  modifiers follow it: a modifier is de-inflected when its ending is 1st / 2nd declension (Vallis Telinae -> Vallis Telina,
  Montem Pessulanum -> Mons Pessulanus, Sancti Galli -> Sanctus Gallus, with gender agreement with the head), but a genitive
  in -is, -orum, -arum, -ium is kept (Hagae Comitis -> Haga Comitis, Lugduno Batavorum -> Lugdunum Batavorum).

Not handled: Greek forms in -en / -on / -es, consonant stems whose nominative is not recoverable (Paris from Parisios), and any
non-Latin form (Moree, Delfos): call it only when is_latin_context is true.

Optional: Collatinus (pycollatinus, lexicon data under GPL; PyPI says GPL v2; unmaintained and broken on Python >= 3.10
unless collections.Callable is shimmed, which collatinus_lemmas does). It is not a dependency of the repo; install it on
Hypatia only if wanted, in a scratch venv. It knows classical names (Rhenus, Lutetia, Mosella, Londinium) but none of the
medieval / early-modern forms this module targets (Leida, Haga, Colmaria, Transilvania), so it is a cross-check, not a source.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    'Candidate', 'Resolved', 'LatinLexicon', 'lemmatize', 'nominatives', 'lemmatize_in_context', 'resolve', 'fold_key',
    'is_latin_context', 'latin_context_level', 'resolve_in_context', 'CONTEXT_MIN_SCORE', 'is_latin_language', 'edition_latin_share', 'latin_share_from_languages',
    'collatinus_available', 'collatinus_lemmas',
]


@dataclass(frozen=True)
class Candidate:
    lemma: str
    score: float
    rule: str

    def __str__(self) -> str:
        return self.lemma


# ---------------------------------------------------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------------------------------------------------

_LIGATURES = str.maketrans({'æ': 'ae', 'Æ': 'Ae', 'œ': 'oe', 'Œ': 'Oe', 'ę': 'ae', 'Ę': 'Ae', 'ǽ': 'ae', 'ſ': 's'})


def _clean(s: str) -> str:
    """NFC, ligatures spelled out, combining marks (macrons, accents) removed."""
    s = unicodedata.normalize('NFC', s).translate(_LIGATURES)
    d = unicodedata.normalize('NFD', s)
    return unicodedata.normalize('NFC', ''.join(c for c in d if unicodedata.category(c) != 'Mn'))


def fold_key(s: str) -> str:
    """Lexicon key: lower case, no diacritics, ae / oe -> e, j -> i, v / w -> u, y -> i, punctuation -> space."""
    s = _clean(s).lower()
    s = s.replace('ae', 'e').replace('oe', 'e').replace('j', 'i').replace('v', 'u').replace('w', 'u').replace('y', 'i')
    s = re.sub(r"[^a-z0-9]+", ' ', s)
    return ' '.join(s.split())


# ---------------------------------------------------------------------------------------------------------------------
# single-word rules
# ---------------------------------------------------------------------------------------------------------------------

# (ending, replacement, score, rule, gender of the lemma)
_RULES: List[Tuple[str, str, float, str, str]] = [
    # 1st declension
    ('iam', 'ia', 0.95, 'a1_acc', 'f'), ('am', 'a', 0.92, 'a1_acc', 'f'),
    ('iae', 'ia', 0.93, 'a1_gen_dat_loc', 'f'), ('ae', 'a', 0.90, 'a1_gen_dat_loc', 'f'),
    ('ie', 'ia', 0.62, 'a1_medieval_e', 'f'), ('e', 'a', 0.45, 'a1_medieval_e', 'f'),
    ('as', 'ae', 0.55, 'a1_acc_pl', 'f'), ('is', 'ae', 0.42, 'a1_dat_abl_pl', 'f'), ('arum', 'ae', 0.80, 'a1_gen_pl', 'f'),
    ('is', 'a', 0.25, 'a1_weak_is', 'f'),
    # 2nd declension (masculine -us / -er and neuter -um; the neuter/masculine order is set in _expand)
    ('ium', 'ius', 0.60, 'o2_acc', 'm'), ('um', 'us', 0.80, 'o2_acc', 'm'),
    ('ii', 'ius', 0.80, 'o2_gen', 'm'), ('ii', 'ium', 0.74, 'o2_gen', 'n'),
    ('ri', 'er', 0.60, 'o2_gen_er', 'm'),
    ('i', 'us', 0.80, 'o2_gen', 'm'), ('i', 'um', 0.72, 'o2_gen', 'n'),
    ('io', 'ium', 0.70, 'o2_dat_abl', 'n'), ('io', 'ius', 0.55, 'o2_dat_abl', 'm'),
    ('o', 'us', 0.78, 'o2_dat_abl', 'm'), ('o', 'um', 0.70, 'o2_dat_abl', 'n'),
    ('os', 'i', 0.45, 'o2_acc_pl', 'm'), ('orum', 'i', 0.80, 'o2_gen_pl', 'm'), ('orum', 'um', 0.50, 'o2_gen_pl', 'n'),
    ('is', 'i', 0.40, 'o2_dat_abl_pl', 'm'),
    # 3rd declension
    ('ntem', 'ns', 0.85, 'c3_acc', 'm'), ('ntis', 'ns', 0.85, 'c3_gen', 'm'), ('nte', 'ns', 0.80, 'c3_abl', 'm'),
    ('nti', 'ns', 0.78, 'c3_dat', 'm'),
    ('onem', 'o', 0.86, 'c3_acc', 'f'), ('onis', 'o', 0.86, 'c3_gen', 'f'), ('one', 'o', 0.80, 'c3_abl', 'f'),
    ('oni', 'o', 0.74, 'c3_dat', 'f'),
    ('inem', 'o', 0.83, 'c3_acc', 'f'), ('inis', 'o', 0.80, 'c3_gen', 'f'), ('ine', 'o', 0.74, 'c3_abl', 'f'),
    ('im', 'is', 0.80, 'i3_acc', 'f'), ('em', 'is', 0.55, 'i3_acc', 'f'), ('e', 'is', 0.50, 'i3_abl', 'f'),
    ('atem', 'as', 0.50, 'c3_acc', 'f'), ('ate', 'as', 0.45, 'c3_abl', 'f'),
    ('cem', 'x', 0.45, 'c3_acc', 'm'), ('gem', 'x', 0.45, 'c3_acc', 'm'), ('cis', 'x', 0.40, 'c3_gen', 'm'),
    ('gis', 'x', 0.40, 'c3_gen', 'm'),
    ('rem', 'r', 0.50, 'c3_acc', 'm'), ('ris', 'r', 0.40, 'c3_gen', 'm'),
    ('ibus', 'es', 0.60, 'c3_dat_abl_pl', 'm'),
    ('is', 's', 0.35, 'c3_gen', 'm'),
    ('um', 'es', 0.35, 'c3_gen_pl', 'm'),
]

# adjectives in -ensis (two terminations, 3rd declension): all these forms go to -ensis
_ENSIS = re.compile(r'^(.+?)ens(is|em|i|e|es|ium|ibus)$', re.I)

# stems of 2nd-declension neuter place names (towns in -dunum, -burgum, -acum ...): -um / -ium preferred over -us / -ius
_NEUTER_STEM = re.compile(r'(burg|dun|dur|ac|ent|orat|furt|mont|lan|in|isc|oc|ec)$')
_NEUTER_STEM_NOT = re.compile(r'^(lac|dominic|pontific|rhen)$')
# 1st-declension plural-only place names (nominative -ae)
_PLURAL_A1 = re.compile(r'^(athen|theb|cum|syracus|mycen|cann|aqu|thermopyl|sard|trebi|ferr|gad|bai|vei)ae$')

# nominative-looking endings: score of the form itself as a candidate
_SELF = [('ia', 0.88), ('a', 0.85), ('us', 0.85), ('er', 0.80), ('ns', 0.80), ('is', 0.60), ('es', 0.55), ('um', 0.5), ('ae', 0.5),
         ('i', 0.5), ('os', 0.45), ('on', 0.5), ('o', 0.40), ('e', 0.45)]


def _self_score(lw: str, neuter_bias: bool = True) -> float:
    if lw.endswith('um'):
        stem = lw[:-2]
        if not neuter_bias:
            return 0.5
        if _NEUTER_STEM.search(stem) and not _NEUTER_STEM_NOT.match(stem):
            return 0.90
        if stem.endswith('i'):
            return 0.85          # -ium (n.) is the commonest -um toponym
        return 0.5
    if _PLURAL_A1.match(lw):
        return 0.90
    for end, sc in _SELF:
        if lw.endswith(end):
            return sc
    return 0.70


@dataclass
class _C:
    lemma: str
    score: float
    rule: str
    gender: str = ''


def _expand(word: str, *, adjective_bases: bool = True, neuter_bias: bool = True) -> List[_C]:
    """All de-inflection candidates of one cleaned word (no ranking cut)."""
    w = word
    lw = w.lower()
    out: Dict[str, _C] = {}

    def add(lemma: str, score: float, rule: str, gender: str = '') -> None:
        cur = out.get(lemma)
        if cur is None or score > cur.score:
            out[lemma] = _C(lemma, round(score, 3), rule, gender)

    if len(lw) < 2 or not lw.isalpha():
        return []
    m = _ENSIS.match(lw)
    if m and len(m.group(1)) >= 2:
        stem = w[:len(m.group(1))]
        add(stem + 'ensis', 0.90 if m.group(2) != 'is' else 0.92, 'adj_ensis', '')
        if adjective_bases:
            sl = stem.lower()
            bases = [(stem + 'a', 0.45), (stem + 'um', 0.42)] if sl.endswith('i') else                     [(stem + 'a', 0.45), (stem + 'ia', 0.44), (stem + 'um', 0.42), (stem + 'ium', 0.41)]
            for lem, sc in bases:
                add(lem, sc, 'adj_base', 'f' if lem.endswith('a') else 'n')
        return list(out.values())

    add(w, _self_score(lw, neuter_bias), 'as_is', _gender_of(lw))
    for end, rep, score, rule, gender in _RULES:
        if not lw.endswith(end):
            continue
        stem = lw[:len(lw) - len(end)]
        if len(stem) < 2:
            continue
        if end in ('e', 'ie') and stem[-1] in 'ae':               # Colmariae is not Colmaria + e
            continue
        if end in ('um', 'is', 'ium') and lw.endswith(('orum', 'arum', 'ibus')) and rep != 'i':
            continue
        lemma = w[:len(w) - len(end)] + rep
        if rule.startswith('o2') and rep in ('um', 'ium', 'us', 'ius'):
            neuter = neuter_bias and bool(_NEUTER_STEM.search(stem)) and not _NEUTER_STEM_NOT.match(stem)
            if rep in ('um', 'ium'):
                score = max(score, 0.88) if neuter else score
            elif neuter:
                score = min(score, 0.42)
        if rule == 'a1_gen_dat_loc' and _PLURAL_A1.match(lw):
            add(w, 0.90, 'a1_pl_nom', 'f')
        add(lemma, score, rule, gender)
    # Greek / Latin -os: Rhodos, Delos are their own nominative
    if lw.endswith('os') and len(lw) > 4:
        add(w, 0.5, 'as_is_greek', 'm')
    return list(out.values())


def _gender_of(lw: str) -> str:
    if lw.endswith(('a', 'ia')):
        return 'f'
    if lw.endswith('um'):
        return 'n'
    if lw.endswith(('us', 'er', 'ns')):
        return 'm'
    return ''


def _fits(lemma: str, gender: str) -> bool:
    l = lemma.lower()
    if gender == 'f':
        return l.endswith(('a', 'is', 'ensis', 'ae'))
    if gender == 'm':
        return l.endswith(('us', 'er', 'is', 'ensis', 'ns', 'o', 'es', 'i'))
    if gender == 'n':
        return l.endswith(('um', 'e', 'is', 'ensis', 'on'))
    return True


# ---------------------------------------------------------------------------------------------------------------------
# multiword
# ---------------------------------------------------------------------------------------------------------------------

_PREPS = frozenset('in ad apud ex e de a ab cum prope per inter circa sub iuxta juxta ultra trans intra ante post super supra usque '
                   'versus contra pro propter'.split())
# head nouns of compound toponyms: nominative -> gender
_HEADS = {'vallis': 'f', 'mons': 'm', 'pons': 'm', 'castrum': 'n', 'oppidum': 'n', 'urbs': 'f', 'civitas': 'f', 'insula': 'f',
          'portus': 'm', 'lacus': 'm', 'fluvius': 'm', 'flumen': 'n', 'fretum': 'n', 'promontorium': 'n', 'sinus': 'm',
          'regnum': 'n', 'ducatus': 'm', 'comitatus': 'm', 'marchia': 'f', 'terra': 'f', 'provincia': 'f', 'dioecesis': 'f',
          'ecclesia': 'f', 'abbatia': 'f', 'monasterium': 'n', 'forum': 'n', 'vicus': 'm', 'pagus': 'm', 'ager': 'm',
          'silva': 'f', 'via': 'f', 'campus': 'm', 'aquae': 'f', 'burgus': 'm', 'villa': 'f', 'turris': 'f', 'arx': 'f',
          'collis': 'm', 'fons': 'm', 'mare': 'n', 'oceanus': 'm', 'stagnum': 'n', 'palus': 'f', 'amnis': 'm', 'peninsula': 'f',
          'thermae': 'f', 'palatinatus': 'm', 'landgraviatus': 'm', 'archiepiscopatus': 'm', 'episcopatus': 'm', 'portus': 'm'}
_HEADS = {fold_key(k): v for k, v in _HEADS.items()}
# adjectives placed before their noun: they agree with the NEXT word, which is then the head
_PRE_ADJ = {fold_key(x) for x in ('nova nove novum novus novi novae novam novo magna magnum magnus magni magnae magnam parva '
                                    'parvum parvus parvi parvae parvam sancta sanctum sanctus sancti sanctae sanctam sancto '
                                    'superior inferior superioris inferioris superiorem inferiorem alta altum altus alti altae '
                                    'altam antiqua antiquum antiquus antiqui antiquae antiquam').split()}


def _tokenise(form: str) -> Optional[List[str]]:
    s = _clean(form)
    if re.search(r'[()\[\]{}<>/|]', s):                     # 'Löbau (Lubawa)': a vernacular gloss, not Latin inflection
        return None
    s = re.sub(r"[,;:.!?\"'’«»]+", ' ', s)
    toks = s.split()
    while toks and toks[0].lower() in _PREPS and len(toks) > 1:
        toks = toks[1:]
    if not toks or any(not re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", t) for t in toks):
        return None
    return toks


def _top(cs: List[_C], n: int, gender: str = '', mod: bool = False, keep: bool = False) -> List[_C]:
    res: List[_C] = []
    for c in cs:
        sc = c.score
        if mod:
            lw = c.lemma.lower()
            # a genitive plural / 3rd-declension genitive qualifier is part of the name as is: Comitis, Batavorum
            if c.rule == 'as_is' and lw.endswith(('orum', 'arum', 'ium', 'is', 'um', 'es', 'ibus')):
                sc = max(sc, 0.95 if lw.endswith(('orum', 'arum', 'is', 'ibus')) else 0.7)
        if gender:
            sc += 0.20 if _fits(c.lemma, gender) else -0.10
        if keep and c.rule == 'as_is' and c.lemma.lower().endswith('ae'):
            sc = 1.3
        res.append(_C(c.lemma, sc, c.rule, c.gender))
    res.sort(key=lambda c: -c.score)
    return res[:n]


def _fold_head(lemma: str) -> str:
    return fold_key(lemma)


def _lemmatize_words(toks: List[str], adjective_bases: bool) -> List[Candidate]:
    n = len(toks)
    # head position: the first word, unless it is an adjective placed before its noun
    h = 0
    if n > 1 and fold_key(toks[0]) in _PRE_ADJ:
        h = 1
    head_cands = _expand(toks[h], adjective_bases=adjective_bases)
    known = [c for c in head_cands if fold_key(c.lemma) in _HEADS]
    if known:
        head_cands = known                                             # Vallis, Vallem, Vallis: the head noun is Vallis
    for c in head_cands:                                    # known head nouns are certain in their nominative
        if fold_key(c.lemma) in _HEADS:
            c.score = 0.97 if c.rule == 'as_is' else max(c.score, 0.9)
            c.gender = _HEADS[fold_key(c.lemma)]
    head_cands = _top(head_cands, 8)
    if n > 1 and head_cands:
        head_cands = [c for c in head_cands if c.score >= head_cands[0].score - 0.25]
    if n == 1:
        return [Candidate(c.lemma, c.score, c.rule) for c in head_cands]
    # beam over (partial words, mean score)
    beams: List[Tuple[List[str], float, str]] = []
    for hc in head_cands:
        g = hc.gender or _gender_of(hc.lemma.lower())
        plural_head = hc.lemma.lower().endswith('ae') and fold_key(hc.lemma) in _HEADS      # Aquae Sextiae: both plural
        words = list(toks)
        total = hc.score
        ok = True
        for i, t in enumerate(toks):
            if i == h:
                words[i] = hc.lemma
                continue
            gg = g if (i > h or fold_key(t) in _PRE_ADJ) else ''
            mc = _top(_expand(t, adjective_bases=False, neuter_bias=not gg), 3, gender=gg, mod=True, keep=plural_head)
            if not mc:
                ok = False
                break
            words[i] = mc[0].lemma
            total += mc[0].score
        if ok:
            beams.append((words, total / n, hc.rule))
    # the other readings of the first modifier give the alternatives (e.g. identity vs de-inflected)
    alts: List[Tuple[List[str], float, str]] = list(beams)
    for hc in head_cands[:2]:
        g = hc.gender or _gender_of(hc.lemma.lower())
        plural_head = hc.lemma.lower().endswith('ae') and fold_key(hc.lemma) in _HEADS
        for i, t in enumerate(toks):
            if i == h:
                continue
            mcs = _top(_expand(t, adjective_bases=False, neuter_bias=not g), 3, gender=g, mod=True, keep=plural_head)
            for mc in mcs[1:]:
                words = [hc.lemma if j == h else toks[j] for j in range(n)]
                tot = hc.score
                for j, tj in enumerate(toks):
                    if j == h:
                        continue
                    if j == i:
                        words[j] = mc.lemma
                        tot += mc.score
                    else:
                        best = _top(_expand(tj, adjective_bases=False, neuter_bias=not g), 1, gender=g, mod=True, keep=plural_head)
                        words[j] = best[0].lemma if best else tj
                        tot += best[0].score if best else 0.5
                alts.append((words, tot / n, 'multiword'))
    seen: Dict[str, Candidate] = {}
    for words, sc, rule in alts:
        s = ' '.join(words)
        c = Candidate(s, round(sc, 4), 'multiword')
        if s not in seen or c.score > seen[s].score:
            seen[s] = c
    return list(seen.values())


def lemmatize(form: str, *, max_candidates: int = 6, min_score: float = 0.4, include_self: bool = True,
              adjective_bases: bool = True, collatinus: bool = False) -> List[Candidate]:
    """Ranked nominative candidates of a Latin place-name form (single word or multiword).

    The first candidate is the best guess without a lexicon; use resolve to keep only the candidates that are known names.
    include_self keeps the form itself in the list when it looks like a nominative (Roma, Lugdunum).
    """
    if not form or not isinstance(form, str):
        return []
    toks = _tokenise(form)
    if toks is None:
        return []
    upper = len(toks) == 1 and len(toks[0]) > 1 and toks[0].isupper()
    if upper:
        toks = [toks[0].capitalize()]
    cands = _lemmatize_words(toks, adjective_bases)
    if collatinus and len(toks) == 1:
        for lem in collatinus_lemmas(toks[0]):
            cands.append(Candidate(lem, 0.90, 'collatinus'))
    best: Dict[str, Candidate] = {}
    for c in cands:
        lemma = c.lemma.upper() if upper else c.lemma
        c = Candidate(lemma, c.score, c.rule)
        if c.score < min_score:
            continue
        if not include_self and c.lemma == form:
            continue
        if c.lemma not in best or c.score > best[c.lemma].score:
            best[c.lemma] = c
    ranked = sorted(best.values(), key=lambda c: -c.score)[:max_candidates]
    return [Candidate(c.lemma, min(round(c.score, 3), 1.0), c.rule) for c in ranked]       # scores are ranks, not probabilities


def nominatives(form: str, **kw) -> List[str]:
    return [c.lemma for c in lemmatize(form, **kw)]


def lemmatize_in_context(form: str, latin_context: bool, **kw) -> List[Candidate]:
    """lemmatize when the context is Latin, otherwise nothing (Moree, Delfos are French / Spanish words)."""
    return lemmatize(form, **kw) if latin_context else []


# ---------------------------------------------------------------------------------------------------------------------
# lexicon matching
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolved:
    lemma: str            # the candidate, as generated
    matched: str          # the lexicon name it matched (original spelling)
    score: float
    rule: str
    payload: object = None


class LatinLexicon:
    """Known Latin names, indexed on fold_key. names is an iterable of strings or a mapping name -> payload."""

    def __init__(self, names: 'Iterable[str] | Mapping[str, object]'):
        self._idx: Dict[str, List[Tuple[str, object]]] = {}
        items = names.items() if isinstance(names, Mapping) else ((n, None) for n in names)
        for name, payload in items:
            if not name:
                continue
            self._idx.setdefault(fold_key(name), []).append((name, payload))

    def __len__(self) -> int:
        return len(self._idx)

    def __contains__(self, name: str) -> bool:
        return fold_key(name) in self._idx

    def get(self, name: str) -> List[Tuple[str, object]]:
        return self._idx.get(fold_key(name), [])


def resolve(form: str, lexicon: LatinLexicon, *, min_score: float = 0.6, max_candidates: int = 10, **kw) -> List[Resolved]:
    """Candidates of ``form`` found in ``lexicon``, best first. The form itself is never a result (nothing to gain).

    ``min_score`` 0.6 drops the weak rules (medieval -e -> -a, accusative plural -os -> -i, -cis / -gis -> -x ...): against a lexicon
    of tens of thousands of names they give false matches (Delos -> Deli, Lippe -> Lippa, Syene -> Siena).
    """
    out: List[Resolved] = []
    seen = set()
    f = fold_key(form)
    for c in lemmatize(form, max_candidates=max_candidates, min_score=min_score, **kw):
        k = fold_key(c.lemma)
        if k == f or k in seen:
            continue
        hits = lexicon.get(c.lemma)
        if hits:
            seen.add(k)
            name, payload = hits[0]
            out.append(Resolved(c.lemma, name, c.score, c.rule, payload))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Latin context
# ---------------------------------------------------------------------------------------------------------------------

_LATIN_CODES = {'la', 'lat', 'latin', 'latina', 'latein', 'lateinisch', 'latim', 'latino', 'latijn', 'neo-latin'}
_UNKNOWN_CODES = {'', '?', 'und', 'mul', 'zxx', 'trad', 'none', 'nan', 'null'}


def is_latin_language(code: Optional[str]) -> Optional[bool]:
    """True for a Latin language code ('la', 'lat', 'la-x-medieval', 'Latin'), False for another language, None if unknown.

    The codes present in mentions.parquet (lang) and documents.parquet (language) are a mix of ISO 639-1 / 639-2 and
    free text: la, lat, de, deu, ger, fr, fra, fre, frm, en, eng, nl, nld, dut, es, spa, esp, it, ita, pt, ...
    """
    if code is None:
        return None
    s = str(code).strip().lower()
    if s in _UNKNOWN_CODES:
        return None
    base = s.split('-')[0].split('_')[0]
    if s in _LATIN_CODES or base in _LATIN_CODES:
        return True
    return False


def latin_share_from_languages(languages: Optional[Sequence[str]]) -> float:
    """Rough Latin share of an edition from the languages list of data/entries/*.json: 0 without Latin; 1/n if Latin comes
    first among n languages (1.0 for Latin alone), 0.5/n otherwise (Latin only a secondary language)."""
    langs = [l for l in (languages or []) if l]
    if not langs or not any(is_latin_language(l) for l in langs):
        return 0.0
    n = len(langs)
    return round((1.0 if is_latin_language(langs[0]) else 0.5) / n, 2)


# coll_slug -> (Latin share, source). Source 'm': measured on the place mentions of mentions.parquet v2.2 that carry a known
# lang (>= 50 mentions and >= 50 % of the place mentions of the collection); share = lang in {la, lat} / mentions with a
# known lang. Source 'e': estimated from the languages of the edition's entry in Renascor data/entries (latin_share_from_languages).
_EDITIONS: Dict[str, Tuple[float, str]] = {
    'abacus': (0.01, 'm'),
    'bdpn': (0.02, 'm'),
    'betamasaheft_geez': (0.0, 'm'),
    'biblioteca_italiana': (0.25, 'e'),
    'bullinger': (0.62, 'm'),
    'camena_cera': (1.0, 'e'),
    'camena_poemata': (1.0, 'e'),
    'camena_thesaurus': (1.0, 'e'),
    'casebooks': (0.0, 'm'),
    'cavriana': (0.0, 'e'),
    'celt': (0.17, 'e'),
    'cidtc': (0.68, 'm'),
    'ckcc_epistolarium': (0.07, 'e'),
    'controversia_confessio': (0.25, 'e'),
    'corpus17_french': (0.0, 'e'),
    'corpus_corporum': (1.0, 'e'),
    'cosuiza': (0.0, 'e'),
    'croala': (0.95, 'm'),
    'dasypodius_dictionarium_latinogermanicum': (0.5, 'e'),
    'dipko_schwarzenhorn_1649': (0.01, 'm'),
    'disco_sonnets': (0.0, 'm'),
    'dta_1600': (0.02, 'm'),
    'elan_crenum': (0.0, 'e'),
    'emmo_folger': (0.0, 'e'),
    'epistemon_bvh': (0.0, 'm'),
    'ethica_complementoria': (0.0, 'e'),
    'ferdinand_i': (0.12, 'e'),
    'freem_bvh': (0.0, 'm'),
    'freem_other': (0.0, 'e'),
    'fruchtbringende_briefe': (0.0, 'm'),
    'germanc': (0.0, 'e'),
    'gongora_epistolario': (0.0, 'm'),
    'gongora_soledades': (0.0, 'm'),
    'grand_tour': (0.0, 'e'),
    'guicciardini_si': (0.0, 'm'),
    'h15corpus': (0.0, 'e'),
    'hainhofer_reisen': (0.0, 'e'),
    'hainhofer_stammbuch': (0.08, 'e'),
    'hearth_tax': (0.25, 'e'),
    'kaiserin_eleonora_magdalena': (0.0, 'm'),
    'karlstadt_kritische_gesamtausgabe': (0.5, 'e'),
    'koenigsfelden': (0.25, 'e'),
    'leibniz_reihe_viii': (0.33, 'e'),
    'letradas': (0.0, 'm'),
    'lettres_calvin': (0.0, 'm'),
    'luene_nonnenbriefe': (0.25, 'e'),
    'map_of_early_modern_london': (0.0, 'e'),
    'old_bailey': (0.0, 'e'),
    'paesi_novamente': (0.0, 'e'),
    'poliziano_sappho': (0.17, 'e'),
    'postscriptum': (0.0, 'm'),
    'qhod_sema': (0.01, 'm'),
    'reed': (0.08, 'm'),
    'rejestr_poznan': (1.0, 'e'),
    'religionsfrieden_quellenedition': (0.04, 'm'),
    'rubens_online': (0.06, 'm'),
    'rup': (0.0, 'e'),
    'sandrart': (0.01, 'm'),
    'selbstzeugnisse_hab_diarium': (0.25, 'e'),
    'selbstzeugnisse_hab_sz2': (0.25, 'e'),
    'sozinianische_briefwechsel': (0.85, 'm'),
    'ssrq_swisslaw': (0.0, 'm'),
    'st_galler_missiven': (0.0, 'm'),
    'suriano': (0.0, 'e'),
    'tagebuecher_christian_ii_anhalt': (0.17, 'e'),
    'urfehdebuecher_basel_x': (0.0, 'e'),
}


def edition_latin_share(coll_slug: Optional[str]) -> Optional[float]:
    """Latin share of the collection (0-1), None if the collection is not in the v2.2 table."""
    v = _EDITIONS.get(coll_slug or '')
    return None if v is None else v[0]


def is_latin_context(coll_slug: Optional[str] = None, lang: Optional[str] = None, doc_language: Optional[str] = None, *,
                     edition_share: Optional[float] = None, min_share: float = 0.5) -> bool:
    """Is a place mention to be read as Latin? Decided in this order, first answer wins:

    1. lang: the mention's own language (mentions.parquet lang; set on 60 % of the place mentions);
    2. doc_language: the document's language (documents.parquet language; 'header', 'catalogue' ... sources);
    3. the edition: edition_share if given, else edition_latin_share(coll_slug), true when >= min_share
       (corpus_corporum, croala, camena_*, sozinianische_briefwechsel, cidtc, bullinger, karlstadt ...).
    Unknown collection and no language: False (precision first).
    """
    for code in (lang, doc_language):
        r = is_latin_language(code)
        if r is not None:
            return r
    share = edition_share if edition_share is not None else edition_latin_share(coll_slug)
    return share is not None and share >= min_share


def latin_context_level(coll_slug: Optional[str] = None, lang: Optional[str] = None, doc_language: Optional[str] = None, *,
                        min_share: float = 0.5) -> str:
    """'latin' (the mention, its document or its edition is Latin: ``is_latin_context``), 'non_latin' (the mention or document
    says another language, or the edition has no Latin), 'mixed' (language unknown, edition lists Latin among other languages
    without being mostly Latin: ckcc_epistolarium, hearth_tax, reed ...), 'unknown' (collection not in the table)."""
    for code in (lang, doc_language):
        r = is_latin_language(code)
        if r is not None:
            return 'latin' if r else 'non_latin'
    share = edition_latin_share(coll_slug)
    if share is None:
        return 'unknown'
    if share >= min_share:
        return 'latin'
    return 'mixed' if share > 0 else 'non_latin'


# lowest rule score accepted per context level; 'mixed' keeps the strong endings (-am, -iam, -ae, -iae, -um/-us, -onem ...)
CONTEXT_MIN_SCORE = {'latin': 0.6, 'mixed': 0.78}


def resolve_in_context(form: str, lexicon: LatinLexicon, coll_slug: Optional[str] = None, lang: Optional[str] = None,
                       doc_language: Optional[str] = None, **kw) -> List[Resolved]:
    """``resolve`` gated by the Latin context of the mention: nothing for 'non_latin' / 'unknown', strong rules only for 'mixed'."""
    lvl = latin_context_level(coll_slug, lang, doc_language)
    if lvl not in CONTEXT_MIN_SCORE:
        return []
    kw.setdefault('min_score', CONTEXT_MIN_SCORE[lvl])
    return resolve(form, lexicon, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# optional: Collatinus
# ---------------------------------------------------------------------------------------------------------------------

_COLL = None


def _collatinus():
    global _COLL
    if _COLL is None:
        try:
            import collections
            import collections.abc
            if not hasattr(collections, 'Callable'):          # pycollatinus 0.1.6 predates Python 3.10
                collections.Callable = collections.abc.Callable
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from pycollatinus import Lemmatiseur
                _COLL = Lemmatiseur()                         # compiles from the bundled data, ~0.7 s
        except Exception:                                      # not installed / broken: the rule path does not need it
            _COLL = False
    return _COLL


def collatinus_available() -> bool:
    return bool(_collatinus())


def collatinus_lemmas(form: str) -> List[str]:
    """Lemmas of one word according to Collatinus ([] if pycollatinus is not installed). Capitalised like the input when the
    lexicon has it as a common noun (buda -> Buda)."""
    lem = _collatinus()
    if not lem or not form or ' ' in form.strip():
        return []
    try:
        res = list(lem.lemmatise(_clean(form)))
    except Exception:
        return []
    out: List[str] = []
    for r in res:
        k = r.get('lemma') if isinstance(r, dict) else None      # pycollatinus yields dicts whose 'lemma' is a str
        if not isinstance(k, str):
            continue
        k = re.sub(r'\d+$', '', k)
        if form[:1].isupper() and k[:1].islower():
            k = k[:1].upper() + k[1:]
        if k and k not in out:
            out.append(k)
    return out
