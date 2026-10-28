"""Small helpers shared by the batch-5 collection modules (additive; common.py is untouched)."""
from __future__ import annotations

import re
from typing import Iterable, Optional

from . import common as C

_YEAR3 = re.compile(r'(?<![\d.])(-?\d{3,4})(?!\d)')


def first_header(root):
    """The first teiHeader element of the tree (or None)."""
    for el in root.iter():
        if C.ln(el.tag) == 'teiHeader':
            return el
    return None


def own_text(el, skip: Iterable[str] = ('date',)) -> str:
    """Text of an element without the listed child elements (e.g. an <author> without its lifespan <date>)."""
    skip = set(skip)
    out = [el.text or '']
    for c in el:
        if C.ln(c.tag) not in skip:
            out.append(C.surface_text(c))
        out.append(c.tail or '')
    return C.norm(''.join(out))


def make_hit(el, *, kind: str, tag: str, surface: str, zone: str, role: str, refs: dict, lang: Optional[str],
             context: Optional[str] = None):
    """Build a common.Hit for an element the generic scanner does not consider an entity tag."""
    anc = []
    p = el.getparent()
    while p is not None:
        n = C.ln(p.tag)
        if n:
            anc.append(n)
        p = p.getparent()
    return C.Hit(kind=kind, tag=tag, name=C.ln(el.tag), surface=surface, zone=zone, role=role, refs=refs,
                 lang=lang, context=context, el=el, ancestors=tuple(reversed(anc)))


def years_in(text: str, lo: int = -3000, hi: int = 2100) -> list[int]:
    """3-4 digit (optionally negative) numbers of a string, as years."""
    return [int(y) for y in _YEAR3.findall(text or '') if lo <= int(y) <= hi]


def any_year_range(el) -> Optional[tuple[int, int]]:
    """(from, to) of a date-like element from its date attributes, else from 3-4 digit numbers in its text.
    Unlike common.date_range this accepts years before 1000 (antiquity)."""
    ys = []
    for a in C._DATE_ATTRS:
        v = el.get(a)
        if v:
            m = re.match(r'\s*(-?\d{1,4})', v)
            if m:
                ys.append(int(m.group(1)))
    if not ys:
        ys = years_in(C.surface_text(el))
    return (min(ys), max(ys)) if ys else None


VIAF_ANY = re.compile(r'viaf\.org/(?:viaf/)?(\d+)', re.I)


def point_context(el, limit: int = 120, mark: str = ' | ') -> str:
    """Context of a point marker (an empty-ish element such as <index> whose text is not the mention itself):
    the text just before and after it, separated by ``mark``. At most ``limit`` characters."""
    prev = el.getprevious()
    left = (prev.tail if prev is not None else (el.getparent().text if el.getparent() is not None else '')) or ''
    right = el.tail or ''
    half = (limit - len(mark)) // 2
    left, right = C.norm(left), C.norm(right)
    return (left[-half:] + mark + right[:half + (half - min(len(left), half))])[:limit]


def author_ref_hits(root, lang: Optional[str], want_context: bool = False, skip_date: bool = True):
    """Hits for header <author ref="..."> elements that carry an authority / local reference but no persName child
    (the generic scanner only sees persName). Tag 'author', role 'author', zone hdr."""
    hdr = first_header(root)
    out = []
    if hdr is None:
        return out
    for el in hdr.iter():
        if C.ln(el.tag) != 'author' or not (el.get('ref') or el.get('key')):
            continue
        if any(C.ln(c.tag) == 'persName' for c in el.iter()):
            continue
        name = own_text(el, ('date',) if skip_date else ())
        if not name:
            continue
        refs = {a: el.get(a).strip() for a in ('ref', 'key') if el.get(a) and el.get(a).strip()}
        out.append(make_hit(el, kind='person', tag='author', surface=name, zone='hdr', role='author', refs=refs,
                            lang=lang, context=name if want_context else None))
    return out


CORRESP_ROLES = ('sender', 'addressee', 'sent_from', 'sent_to')


def drop_header_context(m):
    """The context of a correspDesc entry would only be neighbouring header fields: no snippet for it."""
    if m.zone == 'hdr' and m.role in CORRESP_ROLES:
        m.context = None
    return m
