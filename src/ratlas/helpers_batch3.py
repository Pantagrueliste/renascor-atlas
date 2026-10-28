"""Helpers shared by the batch-3 collection modules (additive; common.py is not touched)."""
from __future__ import annotations

import re
from typing import Iterable, Optional

from . import common as C

# --------------------------------------------------------------------------------------------------------------
# sender / addressee from a letter heading "<rs>Sender</rs> an <rs>Addressee</rs>"
# --------------------------------------------------------------------------------------------------------------


def _flat_with_offsets(head, hit_els: set) -> tuple[str, dict]:
    """Flattened text of ``head`` with every tagged candidate element blanked (so a name containing the
    separator word cannot split the heading) and the start offset of each element in ``hit_els``."""
    parts: list[str] = []
    pos = 0
    offs: dict = {}

    def rec(e, blank: bool):
        nonlocal pos
        if id(e) in hit_els:
            offs[id(e)] = pos
            blank = True
        if e.text:
            t = e.text
            parts.append(' ' * len(t) if blank else t)
            pos += len(t)
        for c in e:
            if isinstance(c.tag, str):
                rec(c, blank)
            if c.tail:
                parts.append(' ' * len(c.tail) if blank else c.tail)
                pos += len(c.tail)

    rec(head, False)
    return ''.join(parts), offs


def assign_heading_roles(hits: list, sep: str = r'\ban\b', head_names=('head',), only_first_head: bool = True) -> None:
    """Set role 'sender' / 'addressee' on the person hits of a letter heading ``X an Y`` (in place).

    A hit qualifies when one of its ancestors is a ``head`` whose text (names blanked) contains ``sep``; hits
    before the separator are senders, after it addressees. Only the first qualifying heading of the document
    is used (later headings belong to enclosures or sections)."""
    rx = re.compile(sep)
    by_head: dict = {}
    order: list = []
    for h in hits:
        if h.zone != 'txt' or h.kind != 'person':
            continue
        p = h.el.getparent()
        while p is not None and C.ln(p.tag) not in head_names:
            p = p.getparent()
        if p is None:
            continue
        if id(p) not in by_head:
            by_head[id(p)] = (p, [])
            order.append(id(p))
        by_head[id(p)][1].append(h)
    for hid in order:
        head, hs = by_head[hid]
        flat, offs = _flat_with_offsets(head, {id(h.el) for h in hs})
        m = rx.search(flat)
        if not m:
            continue
        for h in hs:
            o = offs.get(id(h.el))
            if o is None:
                continue
            h.role = 'sender' if o < m.start() else 'addressee'
        if only_first_head:
            return


def yymmdd_year(name: str, century: int = 1600) -> Optional[int]:
    """Year of a file name starting with YYMMDD (Fruchtbringende Gesellschaft letters: 171224 -> 1617)."""
    m = re.match(r'(\d{2})\d{4}', name)
    return century + int(m.group(1)) if m else None


# --------------------------------------------------------------------------------------------------------------
# small utilities for adjust() hooks
# --------------------------------------------------------------------------------------------------------------
def enclosing(el, name: str, **attrs):
    """Nearest ancestor-or-self with local name ``name`` (and the given attribute values), else None."""
    p = el
    while p is not None:
        if C.ln(p.tag) == name and all(p.get(k) == v for k, v in attrs.items()):
            return p
        p = p.getparent()
    return None


def set_years(m, y_from: Optional[int], y_to: Optional[int], source: Optional[str]) -> None:
    """Override the document dates of one mention (per-mention dating); in_period/doc_year follow the years."""
    from .model import best_year, period_overlap
    m.doc_year_from, m.doc_year_to, m.date_source = y_from, y_to, source
    m.doc_year = best_year(y_from, y_to)
    m.in_period = period_overlap(y_from, y_to)


def strip_prefix(token: str, prefixes: Iterable[str]) -> str:
    """'psn:philipp_hainhofer' -> 'philipp_hainhofer' for the listed prefixes."""
    for p in prefixes:
        if token.startswith(p + ':'):
            return token[len(p) + 1:]
    return token


def opaque_from_prefixed(m, slug: str, prefixes: Iterable[str], attr_order=('ref', 'key', 'corresp')) -> None:
    """Set local_key '<slug>:<id>' (in_register False) from the first ``prefix:id`` token of the raw refs."""
    for tok in (m.ref_raw or '').split():
        for p in prefixes:
            if tok.startswith(p + ':'):
                m.local_key, m.in_register = f'{slug}:{tok[len(p) + 1:]}', False
                return
