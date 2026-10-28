"""Candidate passages for reference set v2 (gold2): text clean-up and passage drawing.

Functions only, no I/O of corpora (the corpus readers live in ``scripts/ner2_build_passages.py``). All of it is deterministic given a ``random.Random``.

* ``rejoin_eol_hyphens``  end-of-line hyphens are rejoined ("Pon-\\ntefract" -> "Pontefract").
* ``repair_dropcaps``     detached drop capitals ("L ondon", "f rance", "s aine") are glued back to their word; a letter that is a real one-letter word of
                           the language (a, e, i, o, y, à, ...) is never touched.
* ``stopword_share`` / ``guess_lang``  a cheap check that a passage is written in the expected language.
* ``draw_passage``        a 120-200 word passage that starts at a sentence start and ends at a sentence end, with the v1 filter (>= 8 capitalised
                           words that are not sentence-initial, digits <= 10 % of the characters) plus a few more (verse, markup residue, lists).
* ``tokenize``            the tokenisation of ``gold_seed_v1.jsonl`` (``\\w+|[^\\w\\s]``).
"""
from __future__ import annotations

import random
import re
import unicodedata

WORD = re.compile(r'\w+|[^\w\s]', re.U)
SENT = re.compile(r'(?<=[.!?:;])\s+(?=[A-ZÀ-ÝÆŒ"“«(¿¡])')
CAP_RE = re.compile(r'(?<![.!?:;"\n])\s([A-ZÀ-Ý][a-zà-ÿſ]{2,})')
ALPHA = 'A-Za-zÀ-ÖØ-öø-ÿſÆŒæœ'

# one-letter words that exist in the language: never glued to the next word
LEGIT_SINGLE = {
    'en': set('aAIOoy'),
    'la': set('aAeEoOiI&'),
    'it': set('aAeEoOiIéÉèÈàÀ'),
    'es': set('aAeEoOyYuU'),
    'fr': set('aAàÀyYôÔ'),
    'de': set('aAiIoO'),
}

STOP = {
    'en': 'the of and to in that is was for with his he as by on be at it this which from but not are or have had they their who all his her were been'.split(),
    'la': 'et in est non ad cum quod qui ut ex de sed quae si sunt quam per ab sit esse vel enim autem nec etiam hoc eius eorum quo seu atque aut'.split(),
    'it': 'di e il la che in a un per non con le si del della da i è dei sono come più lo gli una al alla nel ma ha se questo anche'.split(),
    'es': 'de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este si porque esta'.split(),
    'fr': 'de la le et les des en un du que qui il est à pour dans une ne se pas sur au par plus ce avec son sa ses mais ou nous vous'.split(),
    'de': 'der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat daß dass sie nach bey'.split(),
}
STOP = {k: set(v) for k, v in STOP.items()}


def tokenize(text: str):
    ms = list(WORD.finditer(text))
    return [m.group() for m in ms], [m.start() for m in ms]


def fold(s: str) -> str:
    """Lower case, long s -> s, accents removed (used for comparisons only, never for the stored text)."""
    s = s.replace('ſ', 's').replace('ſ', 's')
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').lower()


STOP_FOLDED = {k: {fold(w) for w in v} | {'auec', 'auoir', 'ſont', 'sont', 'dans', 'vne', 'vn', 'qui', 'que'} for k, v in STOP.items()}


def rejoin_eol_hyphens(text: str) -> str:
    """'Pon-\\ntefract' / 'Pon- tefract' at a line end -> 'Pontefract' (only when the next piece starts in lower case); '¬' line-end marks too."""
    text = re.sub(rf'([{ALPHA}])[¬⁃]\s*\n\s*([a-zà-ÿſ])', r'\1\2', text)
    text = re.sub(rf'([{ALPHA}])[-‐‑–]\s*\n\s*([a-zà-ÿſ])', r'\1\2', text)
    return text


def doc_vocab(text: str, lang: str):
    """Word counts of a document, with the detached-initial pattern blanked out (so that the broken form does not count as a word)."""
    from collections import Counter
    t = DROPCAP_PAT.sub(' ', text)
    return Counter(fold(w) for w in re.findall(rf'[{ALPHA}]+', t))


DROPCAP_PAT = re.compile(rf'(?<![{ALPHA}\'’\-])([{ALPHA}]) ([a-zà-ÿſ]{{2,}})(?![{ALPHA}]*[\'’])')


def repair_dropcaps(text: str, lang: str, vocab=None):
    """Glue detached drop capitals to their word. Returns (new text, [(before, after), ...]).

    Pattern: a lone, unaccented letter that is not a one-letter word of ``lang``, followed by one space and a lower-case word of >= 2 letters, the lone
    letter standing at a word boundary (never after an apostrophe or inside a word). When ``vocab`` (``doc_vocab`` of the whole document) is given the
    repair is made only if the joined word occurs elsewhere in the document, or if the second half is not a word of the document on its own
    ("s iudex" stays: "iudex" is a word; "f rance" becomes "france": "rance" is not)."""
    legit = LEGIT_SINGLE.get(lang, set('aAeEoOiI'))
    out = []
    reps = []
    pos = 0
    for m in DROPCAP_PAT.finditer(text):
        letter, rest = m.group(1), m.group(2)
        if letter in legit or ord(letter) > 0x7f and letter not in 'ſ':
            continue
        if fold(rest) in STOP_FOLDED.get(lang, ()):
            continue
        if vocab is not None:
            joined = fold(letter + rest)
            if not (vocab.get(joined, 0) > 0 or vocab.get(fold(rest), 0) == 0):
                continue
        out.append(text[pos:m.start()])
        out.append(letter + rest)
        reps.append((m.group(0), letter + rest))
        pos = m.end()
    out.append(text[pos:])
    return ''.join(out), reps


def stopword_share(text: str, lang: str) -> float:
    ws = [fold(w) for w in re.findall(rf'[{ALPHA}]+', text)]
    if not ws:
        return 0.0
    st = {fold(x) for x in STOP[lang]}
    return sum(w in st for w in ws) / len(ws)


def guess_lang(text: str) -> str:
    return max(STOP, key=lambda k: stopword_share(text, k))


def is_verse(raw_seg: str) -> bool:
    """Short lines (median < 55 characters) in a text that kept its line breaks."""
    lines = [l.strip() for l in raw_seg.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    lens = sorted(len(l) for l in lines)
    return lens[len(lens) // 2] < 55


JUNK = re.compile(r'\b(Índice (general|auxiliar)|Table des mati[eè]res|Inhaltsverzeichnis|Questo testo fa parte|Indice generale|Capítulo [IVXL]+ ?-? ?Capítulo|Wikisource|Wikipedia|Transkription|überlieferte Text)\b|\[ ?(editar|modifier|bearbeiten|modifica) ?\]')


def bad_markup(seg: str) -> bool:
    return (bool(re.search(r'https?:|www\.|<|>|&#|&[a-z]+;|\{|\}|\\|\||[←→↑↓◄►▲▼]', seg)) or seg.count('[') + seg.count(']') > 6 or bool(JUNK.search(seg))
            or not re.match(r'[A-ZÀ-ÝÆŒ«“"(¿¡\[‹–—\d]', seg))


def noise_share(seg: str, lang: str) -> float:
    """Share of words that are a single letter which is not a one-letter word of the language (OCR / HTR debris, broken initials)."""
    legit = LEGIT_SINGLE.get(lang, set())
    ws = re.findall(rf"[{ALPHA}]+", seg)
    if not ws:
        return 1.0
    return sum(1 for w in ws if len(w) == 1 and w not in legit) / len(ws)


def passage_ok(seg: str, lang: str, lo=120, hi=200, min_caps=8, max_digits=0.10, min_stop=0.12, max_noise=0.04):
    """The v1 filter and a few more. Returns (ok, reason)."""
    n = len(seg.split())
    if n < lo or n > hi:
        return False, 'length'
    digits = sum(c.isdigit() for c in seg) / max(1, len(seg))
    if digits > max_digits:
        return False, 'digits'
    caps = len(CAP_RE.findall(' ' + seg))
    if caps < min_caps:
        return False, 'caps'
    if bad_markup(seg) or re.search('[●◊]', seg):
        return False, 'markup'
    if noise_share(seg, lang) > max_noise:
        return False, 'noise'
    # a heading or a list: too few function words, or mostly very short or all-capital words
    if stopword_share(seg, lang) < min_stop:
        return False, 'stopwords'
    best = guess_lang(seg)
    if best != lang and stopword_share(seg, best) >= stopword_share(seg, lang):
        return False, f'lang:{best}'
    # tables of contents and indexes: many bare numbers or Roman numerals with a full stop
    if sum(bool(re.fullmatch(r'\d{1,4}[.,;]?', w)) for w in seg.split()) > 0.08 * n:
        return False, 'index'
    if len(re.findall(r'(?<![A-Za-z])[IVXL]{1,6}\.', seg)) >= 4:
        return False, 'toc'
    upper = sum(1 for w in seg.split() if len(w) > 3 and w.isupper())
    if upper > 0.15 * n:
        return False, 'allcaps'
    return True, ''


def draw_passage(text: str, rng: random.Random, lang: str, lo=120, hi=200, tries=120, verse_check=True, min_stop=0.12, extra_check=None):
    """Random passage of lo..hi words starting at a sentence start and ending at a sentence end.

    ``text`` may keep its line breaks (they are used only to detect verse); the returned passage has single spaces.
    Returns (offset in the cleaned text, passage) or None."""
    flat = text.replace('\n', ' ')          # same length as `text`, so offsets are shared
    starts = [0] + [m.end() for m in SENT.finditer(flat)]
    if len(starts) < 3:
        return None
    for _ in range(tries):
        j = rng.randrange(len(starts))
        s = starts[j]
        k = j
        end = s
        while k < len(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(flat)
            k += 1
            if len(flat[s:end].split()) >= lo:
                break
        seg_raw = text[s:end]
        seg = re.sub(r'\s+', ' ', flat[s:end]).strip()
        ok, _ = passage_ok(seg, lang, lo, hi, min_stop=min_stop)
        if not ok:
            continue
        if verse_check and is_verse(seg_raw):
            continue
        if extra_check is not None and not extra_check(seg):
            continue
        # must end at a sentence end
        if not re.search(r'[.!?:;»”"\)]\s*$', seg):
            continue
        return s, seg
    return None


def fingerprints(text: str, n=8):
    ws = [fold(w) for w in re.findall(r'\w+', text)]
    return {' '.join(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}
