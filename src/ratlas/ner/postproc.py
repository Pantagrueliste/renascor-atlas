"""RenNER v2.1 (b): boundary post-processing rules that do not depend on the open conventions of the owner.

They never decide: the final abbreviation period, peoples and adjectives, a place inside a title, regnal names, deities and
personifications, titles and kinship inside PER, generic buildings. A title is never removed from a name; a final '.' is never touched.

Rules (each can be turned on alone; ``apply(tokens, lang, spans, rules)``; spans are {"LOC": [(s, e)], "PER": [(s, e)]}, end exclusive):
  trim_punct      drop leading / trailing punctuation tokens of a span (',', ';', ':', '|', '/', dashes, quotes and brackets left
                  unpaired inside the span; never '.') and
                  trailing particles ("conde de" -> "conde", "Terram de" -> "Terram"): a name never ends with "de", "di", "of", "von"...
  (particles count only when written in lowercase: a capital "E" is an abbreviation, e.g. "E F G", not the conjunction)
  drop_generic    drop a span made only of particles, punctuation, digits and (PER only) title words: "don", "señora doña", "conde de".
                  A title alone is not a name under any convention (the conventions only say whether a title next to a name is inside it).
  split_title     (not Latin) split a PER span before an inner honorific (don, doña, signor, messer, monsieur, sier...) that follows a capitalised
                  name and is directly followed by another capitalised name: "Aminta doña Vitoria" -> "Aminta" + "doña Vitoria" (two persons;
                  the honorific stays with its name). Rank titles after a name ("Fridericus Dux Saxoniae", "Christian Fürst zu Anhalt")
                  are left alone: whether they are inside PER is an open convention.
  split_punct     split a span at an inner ',', ';', ':' or '|' (the punctuation leaves the spans).
  split_coord     split a span at an inner coordinator ("e", "et", "y", "und", "and", "&", "ac", "atque") between capitalised words.
  merge_loc_of_loc  two LOC spans "X di Y" / "X de Y" (one particle between, both capitalised, nothing else between): one span
                  ("Nizza di Provenza", "Bocha di Cataro"). Tested on dev3; not adopted unless it helps there.
  extend_loc_adj  a LOC span followed by a size / position adjective of the name ("Armenia maggiore", "Asia minor") takes it in.
  la_place_pair   Latin date lines ("Hamburgo Chilonium", "Neoburgo Ratisbonam"): a LOC span of two capitalised words, the first with an
                  ablative ending (-o, -a, -is, -ae, -i), the second with an accusative ending (-um, -am, -em, -im, -as, -os, -es), is split.
"""
from __future__ import annotations

import re
import unicodedata

PARTICLES = {
    'de', 'di', 'da', 'del', 'dei', 'degli', 'della', 'dello', 'delle', 'dal', 'des', 'du', 'd', "d'", 'd’', 'le', 'la', 'lo', 'los', 'las',
    'il', 'el', 'of', 'the', 'von', 'vom', 'van', 'vander', 'vanden', 'zu', 'zum', 'zur', 'ze', 'zue', 'zu̇', 'ab', 'a', 'à', 'al', 'y', 'e',
    'et', 'and', 'und', 'ex', 'in', 'à', 'ten', 'ter', 'op', "'", '’', 'dalla', 'dalle', 'do', 'dos', 'das',
}
TITLES = {
    'don', 'doña', 'dona', 'donna', 'señor', 'señora', 'senor', 'senora', 'sr', 'sra', 'signor', 'signore', 'signora', 'sig', 'monsignor',
    'monsignore', 'messer', 'messere', 'madonna', 'madama', 'madame', 'monsieur', 'mons', 'monsr', 'monseigneur', 'sier', 'ser', 'fra', 'frate',
    'fray', 'padre', 'dom', 'dominus', 'domini', 'domino', 'dominum', 'domine', 'domina', 'dominae', 'dn', 'rex', 'regis', 'regi', 'regem',
    'rege', 'rey', 'roi', 'roy', 'king', 'queen', 'regina', 'reina', 'reyna', 'papa', 'pope', 'pontifex', 'cardinal', 'cardinale',
    'cardenal', 'conte', 'conde', 'comte', 'comes', 'duca', 'duque', 'duc', 'duke', 'dux', 'marchese', 'marqués', 'marques', 'marquis',
    'principe', 'príncipe', 'prince', 'princeps', 'vescovo', 'obispo', 'bishop', 'episcopus', 'arzobispo', 'archbishop', 'imperator',
    'imperatore', 'emperador', 'empereur', 'emperor', 'kaiser', 'könig', 'herzog', 'graf', 'fürst', 'herr', 'frau', 'sir', 'lord', 'lady',
    'mr', 'mrs', 'capitano', 'capitán', 'captain', 'virrey', 'viceré', 'viceroi', 'almirante', 'contador', 'licenciado', 'doctor',
    'dottor', 'dottore', 'dr', 'magister', 'frater', 'sor', 'madre', 'mastro', 'maestro', 'reverendo', 'reverendus', 'baili', 'bailo',
}
# honorifics put in front of a given name, never after a name as an apposition (unlike rank titles: "Fridericus Dux Saxoniae")
HONORIFICS = {'don', 'doña', 'dona', 'señor', 'señora', 'senor', 'senora', 'signor', 'signora', 'signore', 'messer', 'messere', 'monsieur',
              'madame', 'madama', 'madonna', 'monsignor', 'monsignore', 'sier', 'ser', 'mr', 'mrs', 'sir', 'fray'}
COORD = {'e', 'et', 'y', 'und', 'and', '&', 'ac', 'atque', 'ed'}
SPLIT_PUNCT = {',', ';', ':', '|'}
ABL = ('o', 'a', 'is', 'ae', 'i', 'ā', 'ō')
ACC = ('um', 'am', 'em', 'im', 'as', 'os', 'es')
LOC_ADJ = {'maggiore', 'minore', 'major', 'minor', 'maior', 'superiore', 'inferiore', 'superior', 'inferior', 'mayor', 'menor', 'magna',
           'parva', 'citeriore', 'ulteriore', 'citerior', 'ulterior', 'alta', 'bassa', 'vecchia', 'nuova', 'vieja', 'nueva', 'grande'}
RULES = ('trim_punct', 'drop_generic', 'split_title', 'split_punct', 'split_coord', 'la_place_pair', 'merge_loc_of_loc', 'extend_loc_adj')


def _norm(w: str) -> str:
    return unicodedata.normalize('NFC', w).lower().replace('ſ', 's')


def is_punct(w: str) -> bool:
    return bool(w) and all(unicodedata.category(c)[0] in 'PSZ' for c in w)


def is_cap(w: str) -> bool:
    for c in w:
        if c.isalpha():
            return c.isupper()
    return False


TRIM = {',', ';', ':', '|', '/', '-', '–', '—', "'", '’', '‘', '"', '“', '”', '«', '»', '(', ')', '[', ']'}
PAIRS = {')': '(', ']': '[', '’': '‘', '”': '“', '»': '«'}
OPEN = {v: k for k, v in PAIRS.items()}


def _trimmable(tokens, k, s, e, end):
    w = tokens[k]
    if w not in TRIM:
        return False
    inner = tokens[s:e]
    if end and w in PAIRS:              # a closing bracket stays when its opening one is inside the span ("August [ inus ]")
        return PAIRS[w] not in inner[:-1]
    if not end and w in OPEN:
        return OPEN[w] not in inner[1:]
    if w in ("'", '"'):                 # a straight quote stays when the span holds another one
        return inner.count(w) == 1
    return True


def _trim(tokens, s, e):
    while s < e and _trimmable(tokens, s, s, e, False):
        s += 1
    while e > s and (_trimmable(tokens, e - 1, s, e, True) or (_norm(tokens[e - 1]) in PARTICLES and not is_cap(tokens[e - 1]))):
        e -= 1
    return s, e


def _generic(tokens, s, e, lab):
    for w in tokens[s:e]:
        n = _norm(w).rstrip('.')
        if is_punct(w) or w.isdigit() or (n in PARTICLES and not is_cap(w)) or n == '':
            continue
        if lab == 'PER' and n in TITLES:
            continue
        return False
    return True


def _split_at(tokens, s, e, cut):
    """cut: positions k (s < k < e) where token k is dropped (punctuation / coordinator)."""
    out, a = [], s
    for k in cut:
        if k > a:
            out.append((a, k))
        a = k + 1
    if e > a:
        out.append((a, e))
    return out


def apply(tokens, lang, spans, rules):
    out = {}
    for lab, sps in spans.items():
        res = []
        for s, e in sorted(sps):
            pieces = [(s, e)]
            if 'split_punct' in rules:
                nxt = []
                for a, b in pieces:
                    cut = [k for k in range(a + 1, b - 1) if tokens[k] in SPLIT_PUNCT]
                    nxt += _split_at(tokens, a, b, cut) if cut else [(a, b)]
                pieces = nxt
            if 'split_coord' in rules:
                nxt = []
                for a, b in pieces:
                    cut = [k for k in range(a + 1, b - 1) if _norm(tokens[k]) in COORD and is_cap(tokens[k - 1]) and is_cap(tokens[k + 1])]
                    nxt += _split_at(tokens, a, b, cut) if cut else [(a, b)]
                pieces = nxt
            if 'split_title' in rules and lab == 'PER' and lang != 'la':
                nxt = []
                for a, b in pieces:
                    cuts = [k for k in range(a + 1, b - 1) if _norm(tokens[k]).rstrip('.') in HONORIFICS and is_cap(tokens[k - 1])
                            and _norm(tokens[k - 1]).rstrip('.') not in TITLES and _norm(tokens[k - 1]) not in PARTICLES
                            and is_cap(tokens[k + 1]) and _norm(tokens[k + 1]).rstrip('.') not in TITLES]
                    st = a
                    for k in cuts:
                        nxt.append((st, k))
                        st = k
                    nxt.append((st, b))
                pieces = nxt
            if 'la_place_pair' in rules and lab == 'LOC' and lang == 'la':
                nxt = []
                for a, b in pieces:
                    if b - a == 2 and all(is_cap(w) and w.isalpha() for w in tokens[a:b]):
                        w1, w2 = _norm(tokens[a]), _norm(tokens[a + 1])
                        if w1.endswith(ABL) and w2.endswith(ACC) and len(w1) > 3 and len(w2) > 3:
                            nxt += [(a, a + 1), (a + 1, b)]
                            continue
                    nxt.append((a, b))
                pieces = nxt
            if 'trim_punct' in rules:
                pieces = [_trim(tokens, a, b) for a, b in pieces]
            pieces = [(a, b) for a, b in pieces if b > a]
            if 'drop_generic' in rules:
                pieces = [(a, b) for a, b in pieces if not _generic(tokens, a, b, lab)]
            res += pieces
        res = sorted(set(res))
        if lab == 'LOC' and 'merge_loc_of_loc' in rules:
            m = []
            for sp in res:
                if m and sp[0] == m[-1][1] + 1 and _norm(tokens[m[-1][1]]) in {'di', 'de', 'del', 'della', 'du', 'von', 'of'} \
                        and is_cap(tokens[sp[0]]) and is_cap(tokens[m[-1][0]]):
                    m[-1] = (m[-1][0], sp[1])
                else:
                    m.append(sp)
            res = m
        if lab == 'LOC' and 'extend_loc_adj' in rules:
            starts = {a for a, _ in res}
            res = [(a, b + 1) if b < len(tokens) and _norm(tokens[b]) in LOC_ADJ and not is_cap(tokens[b]) and b not in starts else (a, b)
                   for a, b in res]
        out[lab] = res
    return out


def apply_pred_line(d: dict, tokens, lang, rules) -> dict:
    """One prediction line of ner_audit_predict.py ({"id", "LOC": [{s, e, ...}], "PER": [...]}) -> the same with new spans (s, e only)."""
    sp = {lab: [(x['s'], x['e']) for x in d.get(lab) or []] for lab in ('LOC', 'PER')}
    new = apply(tokens, lang, sp, rules)
    return {'id': d['id'], **{lab: [{'s': s, 'e': e} for s, e in new[lab]] for lab in ('LOC', 'PER')}}
