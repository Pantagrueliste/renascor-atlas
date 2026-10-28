"""Helpers shared by the batch-1 collection modules (hearth_tax, ssrq, germanc, kaiserin, ...).

Kept apart from common.py (not to be edited by module authors).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from . import common as C
from .model import Mention

_NUM = re.compile(r'^\d{5,}$')


# ---------------------------------------------------------------------------------------------------------------------
# hits built by hand (elements the generic scanner skips or does not know)
# ---------------------------------------------------------------------------------------------------------------------
def ancestors_of(el) -> tuple:
    out = []
    p = el.getparent()
    while p is not None:
        n = C.ln(p.tag)
        if n:
            out.append(n)
        p = p.getparent()
    return tuple(reversed(out))


def refs_of(el, extra_from: Optional[object] = None) -> dict:
    """The non-empty @ref/@key/@corresp/@sameAs values of an element (and, optionally, of another element)."""
    out = {}
    for src in (el, extra_from):
        if src is None:
            continue
        for a in C.REF_ATTRS:
            v = src.get(a)
            if v and v.strip() and v.strip() != '#' and a not in out:
                out[a] = v.strip()
    return out


def make_hit(el, kind: str, tag: str, role: str, zone: str = 'hdr', refs: Optional[dict] = None,
             lang: Optional[str] = None, surface: Optional[str] = None) -> C.Hit:
    return C.Hit(kind=kind, tag=tag, name=C.ln(el.tag), surface=surface if surface is not None else C.surface_text(el),
                 zone=zone, role=role, refs=refs if refs is not None else refs_of(el), lang=lang, context=None,
                 el=el, ancestors=ancestors_of(el))


# ---------------------------------------------------------------------------------------------------------------------
# per-mention fix-ups
# ---------------------------------------------------------------------------------------------------------------------
def repository_role(m: Mention, hit: C.Hit) -> Mention:
    """A place (or institution) inside msIdentifier is the archive that holds the manuscript, not a mention."""
    if 'msIdentifier' in hit.ancestors:
        m.role = 'repository'
    return m


def set_opaque(m: Mention, slug: str, ident: str, doc_id: Optional[str] = None) -> Mention:
    """Give the mention an edition-scoped key without a register ('<slug>:<id>' or '<slug>:<doc>#<id>')."""
    m.local_key = C.make_key(slug, ident, doc_id)
    m.in_register = False
    return m


def drop_url_other(m: Mention, prefix: str) -> None:
    """Remove 'url:<prefix>...' entries from auth.other (local ids written as URLs, not authority ids)."""
    m.auth['other'] = [x for x in m.auth.get('other', []) if not x.startswith('url:' + prefix)]


def bare_number_as_authority(m: Mention, tok: str, kind: str) -> bool:
    """CroALa-style bare numbers: GeoNames id for places (certain by project convention); for persons the
    number is *probably* a VIAF id but is not marked: kept as 'num:<id>' in auth.other, never in auth.viaf."""
    if not _NUM.match(tok):
        return False
    if kind == 'place':
        if not m.auth.get('geonames'):
            m.auth['geonames'] = tok
        return True
    x = f'num:{tok}'
    if x not in m.auth['other']:
        m.auth['other'].append(x)
    return True


# ---------------------------------------------------------------------------------------------------------------------
# anaphora (GerManC rs)
# ---------------------------------------------------------------------------------------------------------------------
_ANAPHORA = {
    # personal / possessive / demonstrative / relative pronouns and deictic adverbs (German, 17th-18th c. spellings)
    'er', 'sie', 'es', 'ihm', 'ihn', 'ihr', 'ihnen', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'sein', 'seine',
    'seinem', 'seinen', 'seiner', 'seines', 'deren', 'dessen', 'derselbe', 'dieselbe', 'dasselbe', 'denselben',
    'demselben', 'desselben', 'derselben', 'selbige', 'selbigen', 'selbiger', 'diese', 'dieser', 'diesem', 'diesen',
    'dieses', 'der', 'die', 'das', 'dem', 'den', 'des', 'welcher', 'welche', 'welches', 'welchem', 'welchen',
    'welchen', 'wir', 'uns', 'unser', 'unsere', 'unserm', 'unserer', 'ich', 'mich', 'mir', 'meiner', 'mein',
    'meine', 'meinen', 'meinem', 'du', 'dich', 'dir', 'dein', 'deine', 'euch', 'euer', 'eure', 'man', 'sich',
    'dahier', 'hier', 'dort', 'da', 'daselbst', 'allda', 'alda', 'dorten', 'dorthin', 'dahin', 'daher', 'wo',
    'woselbst', 'woher', 'wohin', 'hierher', 'hieher', 'hinein', 'herein', 'darin', 'darinnen', 'drinnen',
    'dorinn', 'dahin', 'jener', 'jene', 'jenem', 'jenen', 'jenes', 'beyde', 'beide', 'beyden', 'beiden', 'alle',
    'allen', 'aller', 'einer', 'eine', 'einem', 'einen', 'ein', 'sich', 'selbst', 'wer', 'wen', 'wem', 'wessen',
}


def is_anaphoric(surface: str) -> bool:
    """True when every word of the surface is a pronoun / deictic word (so the rs is a coreference link)."""
    words = [w for w in re.split(r'[\s.,;:!?()\[\]"\'=/-]+', surface.lower()) if w]
    return bool(words) and all(w in _ANAPHORA for w in words)


# ---------------------------------------------------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------------------------------------------------
_DMS = re.compile(r'''([NSEWOnsewo])\s*(\d{1,3})\s*[°º]\s*(\d{1,2})\s*['′’]\s*([\d.,]+)\s*(?:["″”]|'')?''')
# hemisphere written after the numbers: 41°53′35″N 12°28′58″E. 'O' is read as Spanish / Italian 'Oeste' = West
# ('O0°52′58.91″' = 0°52′58.91″ W; only the Spanish H15corpus calls this parser, a German 'O' = Ost would be wrong).
# Without these two cases the string fell through to the decimal parser: (41.0, 53.0).
_DMS_POST = re.compile(r'''(\d{1,3})\s*[°º]\s*(\d{1,2})\s*['′’]\s*([\d.,]+)\s*(?:["″”]|'')?\s*([NSEWOnsewo])''')


def parse_coord(txt: str) -> Optional[tuple[float, float]]:
    """'45.466944, 9.19' or 'N45°27'51.37" E9°11'22.24"' -> (lat, lon); None if absent or out of range."""
    if not txt or not txt.strip():
        return None
    parts = _DMS.findall(txt)
    if len(parts) < 2:
        parts = [(h, d, mi, sec) for d, mi, sec, h in _DMS_POST.findall(txt)]
    if len(parts) >= 2:
        vals = {}
        for hemi, d, mi, s in parts:
            v = float(d) + float(mi) / 60 + float(s.replace(',', '.')) / 3600
            h = hemi.upper()
            if h == 'O':
                h = 'W'
            vals[h] = -v if h in 'SW' else v
        lat = vals.get('N', vals.get('S'))
        lon = vals.get('E', vals.get('W'))
        if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
            return round(lat, 6), round(lon, 6)
        return None
    return C._parse_geo(txt)


def count_stats(mentions: list[Mention]) -> Counter:
    return Counter((m.kind, m.role) for m in mentions)


# ---------------------------------------------------------------------------------------------------------------------
# base extractor: generic TEI + the clean-ups every batch-1 edition needs
# ---------------------------------------------------------------------------------------------------------------------
from .collections.generic_tei import GenericTEI  # noqa: E402


class BaseTEI(GenericTEI):
    """GenericTEI that (1) drops empty tags without any reference, (2) marks places inside msIdentifier as
    'repository'. Subclasses call ``super().adjust(m, hit)`` first and may return None to drop a mention."""

    def adjust(self, m, hit):
        if not m.surface and not m.ref_raw:
            return None
        return repository_role(m, hit)


# ---------------------------------------------------------------------------------------------------------------------
# test helper
# ---------------------------------------------------------------------------------------------------------------------
def extract_paths(slug: str, *paths: str):
    """Run a collection module on the given scan paths (exact path or unique suffix). Returns (mentions, entities)."""
    from .collections import get
    col = get(slug)
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    sel = []
    for p in paths:
        hit = [r for r in files if r.path == p or r.path.endswith(p)]
        if not hit:
            raise KeyError(p)
        sel.append(hit[0])
    ents = list(col.prepare(ctx))
    ms = []
    for r in sel:
        m, e = col.extract(r, ctx)
        ms += m
        ents += e
    return ms, ents


def synthetic(slug: str, xml: str, path: str = 'x/doc.xml'):
    """Run a module's extractor on an XML string (no scan info: registers are looked for). Register files are not
    treated specially. Returns (mentions, entities)."""
    from .collections import get
    col = get(slug)
    ctx = col.make_context()
    rec = C.FileRecord(path=path, coll='x')
    rec.data = xml.encode('utf8')
    return col.extract(rec, ctx)
