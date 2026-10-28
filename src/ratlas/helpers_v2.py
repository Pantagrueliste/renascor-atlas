"""Helpers of the v2 rebuild (audit of 2026-10-16, owner decisions of 2026-10-16).

  * ``pick(legacy, new)``       module paths: the re-harvest of 2026-10-17 (default) or the v1 harvest (RATLAS_HARVEST=legacy)
  * ``NEW``                     the dated folder under RQ/data/raw
  * ``load_registers``          read stand-alone register files by path (not through the scan) into ``ctx.shared``
  * ``clean_surface``           line-break hyphens, split initials, trailing punctuation (audit: freem_other, elan_crenum, corpus_corporum)
  * ``entry_dates``             date each mention from the nearest preceding dated entry (diaries, Urfehdebuch)
  * ``range_of``                helper to build a DocDate from years (never a midpoint of a wide range: see model.best_year)
"""
from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Optional

from . import common as C
from .model import Entity, Mention

NEW = 'originals_2026-10-17'


def use_new() -> bool:
    """True = read the re-harvest of 2026-10-17 (v2). RATLAS_HARVEST=legacy keeps the v1 paths available."""
    return os.environ.get('RATLAS_HARVEST', 'new').lower() != 'legacy'


def pick(legacy, new):
    """``legacy`` or ``new`` (lists of selectors, or strings) according to the harvest switch."""
    return new if use_new() else legacy


# ----------------------------------------------------------------------------------------------------------------------
# registers read by path
# ----------------------------------------------------------------------------------------------------------------------
def read_xml(path: str):
    """Parsed root of a raw-relative file, or None when it is missing (a missing register must not stop a run)."""
    try:
        data = C.read_raw(path)
    except (FileNotFoundError, KeyError, OSError):
        return None
    return C.parse_xml(data)


def load_registers(ctx: C.Context, paths: Iterable[str], *, fix: Optional[Callable] = None, doc_id=None) -> list[Entity]:
    """Parse TEI register files (listPerson/listPlace/listOrg) found at raw-relative ``paths`` into ``ctx.shared``.

    ``fix(ent, aliases, el_by_id) -> (ent, aliases) | None`` may edit or drop an entry. Returns the entities."""
    out: list[Entity] = []
    for p in paths:
        root = read_xml(p)
        if root is None:
            continue
        els = {e.get(C.XML_ID): e for e in root.iter() if isinstance(e.tag, str) and e.get(C.XML_ID)}
        for ent, aliases in C.parse_register(root, ctx, p, doc_id=doc_id)[0]:
            if fix is not None:
                r = fix(ent, list(aliases), els)
                if r is None:
                    continue
                ent, aliases = r
            if ent.local_key in ctx.shared.entities:
                continue
            ctx.shared.add(ent, aliases)
            out.append(ent)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# surfaces
# ----------------------------------------------------------------------------------------------------------------------
_HYPH = re.compile(r'(?<=\w)[-­‐‑‒–] (?=[a-zß-ÿ])')     # 'Lon- dres', 'Po- linville'
_LONG_S = str.maketrans({'ſ': 's'})
_TRAIL = re.compile(r'[\s,;:.·]+$')


def clean_surface(s: str, *, hyphens: bool = True, trailing: bool = True) -> str:
    """Normalise a surface for the atlas: line-break hyphens joined, trailing punctuation dropped (kept: a final
    abbreviation dot of a short token such as 'St.' or 'Frf.')."""
    if not s:
        return s
    if hyphens:
        s = _HYPH.sub('', s)
    if trailing:
        t = _TRAIL.sub('', s)
        if t and (len(t) > 3 or not s.endswith('.')):
            s = t
    return s.strip()


# ----------------------------------------------------------------------------------------------------------------------
# dates
# ----------------------------------------------------------------------------------------------------------------------
def set_range(m: Mention, y_from: Optional[int], y_to: Optional[int], source: Optional[str]) -> Mention:
    """Set the dating of a mention from a year range. doc_year follows model.best_year (null for wide ranges)."""
    from .model import best_year, period_overlap
    m.doc_year_from, m.doc_year_to = y_from, y_to
    m.doc_year = best_year(y_from, y_to)
    m.in_period = period_overlap(y_from, y_to)
    m.date_source = source if (y_from is not None or y_to is not None) else None
    return m


def years_in(text: str, lo: int = 1400, hi: int = 1800) -> list[int]:
    return [int(y) for y in C._YEAR_RX.findall(text or '') if lo <= int(y) <= hi]


def median(xs: list[int]) -> Optional[int]:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


# ----------------------------------------------------------------------------------------------------------------------
# stable entity keys
# ----------------------------------------------------------------------------------------------------------------------
_PAREN = re.compile(r'[\(\[][^\)\]]*[\)\]]')
_LIFE = re.compile(r'\b(?:ca\.?\s*|um\s+|\*|†|✝)?\d{3,4}\s*[-‐-―−?]\s*(?:ca\.?\s*)?\d{0,4}\??')
_DASHES = str.maketrans({c: '-' for c in '‐‑‒–—―−'})


def term_key(s: str) -> str:
    """Stable key of a printed index term: parenthetical variants, life dates and dash variants dropped, folded.
    'Huygens (Hugenius), Christiaan 1629-1695' and 'Huygens, Christiaan 1629–1695' -> 'huygens christiaan'."""
    import unicodedata
    s = (s or '').translate(_DASHES)
    s = _PAREN.sub(' ', s)
    s = _LIFE.sub(' ', s)
    s = unicodedata.normalize('NFKD', s.casefold())
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


# ----------------------------------------------------------------------------------------------------------------------
# register entry fix-ups shared by the HAB (Wolfenbuettel) editions
# ----------------------------------------------------------------------------------------------------------------------
_MYTH_NOTE = re.compile(r'mytholog|sagengestalt|g[oö]ttin|gott der|held d\. griech|titan|nymphe|muse\b', re.I)
_BIBLE_NOTE = re.compile(r'bibl|testament|apostel|prophet|evangelist', re.I)


def drop_url_other(ent: Entity) -> None:
    """Remove bare 'url:' entries (bibliography links such as ADB / BBKL pages) from auth.other: they are not authority ids."""
    ent.auth['other'] = [x for x in ent.auth.get('other', []) if not x.startswith('url:')]


def hab_person_fix(ent: Entity, aliases: list, els: dict):
    """Entry fix-up for HAB person registers: note text -> mythological / biblical; no bibliography urls in auth."""
    drop_url_other(ent)
    if ent.kind == 'person':
        el = els.get(aliases[0]) if aliases else None
        if el is not None:
            notes = ' '.join(C.surface_text(n) for n in el if C.ln(n.tag) == 'note')
            if _MYTH_NOTE.search(notes):
                ent.note_type = 'mythological'
            elif _BIBLE_NOTE.search(notes):
                ent.note_type = 'biblical'
    if not ent.note_type:
        ent.note_type = ent.kind
    return ent, aliases


# ----------------------------------------------------------------------------------------------------------------------
# dating by the preceding dated entry (diaries, itineraries, record books)
# ----------------------------------------------------------------------------------------------------------------------
class RunningDate:
    """The year of the last ``<date>`` that precedes an element in reading order (the first date serves before it).

    Audit point 12: a diary of 40 years is not dated by one midpoint. Isolated typos are ignored: a date that differs by more
    than ``gap`` years from the median of its neighbours (+-3 dates) is dropped (Urfehdebuch '1656' among 1566).
    Header dates are never used. Build once per document; ``year_at(el)`` is O(log n)."""

    def __init__(self, root, lo: int = 1400, hi: int = 1800, gap: int = 5, attrs=('when', 'notBefore', 'notAfter', 'from', 'to')):
        from bisect import bisect_right
        self._bisect = bisect_right
        self.pos: dict[int, int] = {}
        # lxml hands out short-lived proxy objects: keep every element alive so that id(el) stays unique and stable
        self._keep = list(root.iter())
        raw: list[tuple[int, int, int]] = []          # (position, year_from, year_to)
        for i, el in enumerate(self._keep):
            n = C.ln(el.tag) if isinstance(el.tag, str) else None
            self.pos[id(el)] = i
            if n != 'date':
                continue
            if any(C.ln(a.tag) == 'teiHeader' for a in el.iterancestors()):
                continue
            ys = [y for a in attrs if (y := C._attr_year(el.get(a))) is not None and lo <= y <= hi]
            if ys:
                raw.append((i, min(ys), max(ys)))
        keep = []
        mids = [(a + b) // 2 for _, a, b in raw]
        for k, (i, a, b) in enumerate(raw):
            nb = sorted(mids[max(0, k - 3):k] + mids[k + 1:k + 4])
            med = nb[len(nb) // 2] if nb else mids[k]
            if abs(mids[k] - med) <= gap:
                keep.append((i, a, b))
        self.dates = keep
        self.idx = [i for i, _, _ in keep]

    def __bool__(self):
        return bool(self.dates)

    def range_at(self, el):
        if not self.dates:
            return None
        i = self.pos.get(id(el))
        if i is None:
            return None
        k = self._bisect(self.idx, i) - 1
        _, a, b = self.dates[max(k, 0)]
        return a, b

    def year_at(self, el):
        r = self.range_at(el)
        return r[0] if r and r[0] == r[1] else (r[0] if r else None)
