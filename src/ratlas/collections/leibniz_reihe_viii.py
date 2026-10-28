"""Leibniz, Saemtliche Schriften und Briefe, Reihe VIII (Naturwissenschaftliche, medizinische und technische
Schriften), vols 1-3, 352 chapter / section files (telota/LeibnizVIII-LaTeX_TEI).

Licence: CC-BY-NC-4.0, tier T2 (LICENSE_TIERS: Renascor_leibniz-reihe-viii).

There is no persName / placeName markup. Persons and places are printed-index entries placed in the text:
<index indexName="Namensregister|Personen"><term>Wallis (Wallisius), John 1616-1703</term></index> (persons) and
indexName="Ortsregister|Orte" (places). The index element marks the point of the mention in the running text, so:
  * one mention per top-level person / place index element; tag 'index@Namensregister', 'index@Personen',
    'index@Ortsregister', 'index@Orte'; surface = the printed index term (name with Latin variant and life dates);
  * context = the text just before and after the marker, separated by ' | ' (the mention itself is not tagged);
  * local_key = '<slug>:<kind>:<key>' (opaque, in_register False, there is no register); v2: the key is the printed term
    without parenthetical variants, life dates and dash variants (``helpers_v2.term_key``), so that the 73 split groups
    (Huygens 4 keys, Galilei 4, Boyle 6...) are one entity; terms without a letter (',') are dropped;
  * indexName Sachen / Sachverzeichnis (subjects, 27k) are not persons or places and are skipped. v2: index elements inside
    the critical apparatus (app / lem / rdg, 819 of 2,774) repeat a mention of the main text: zone ``apparatus``, excluded
    from the default counts;
  * no authority ids, no coordinates, no entities table.
Dating: the bracketed date of the chapter head ('[Frühjahr - Dezember 1675]'): years 1660-1710 found in the heads that
follow the title (header_dates would find nothing: the teiHeader is XInclude references). 72 chapters have no date in
the head: they get the span of the dated chapters of their volume (band01 1668-1676, band02 1668-1678, band03
1670-1705), date_source 'collection'.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch5 as H
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'leibniz_reihe_viii'
COLLS = ['originals_2026-10-15/oct-binary/oct__leibniz-reihe-viii']
RENASCOR_IDS = ['leibniz-reihe-viii']
LICENCE = 'CC-BY-NC-4.0'
TIER = 'T2'
DESCRIPTION = 'Leibniz VIII: person and place index terms placed in the text'
OPTIONS = {'zone_fn': lambda h: 'apparatus' if set(h.ancestors) & {'app', 'lem', 'rdg'} else None}

_KIND = {'Namensregister': 'person', 'Personen': 'person', 'Ortsregister': 'place', 'Orte': 'place'}
_BAND = {'band01': (1668, 1676), 'band02': (1668, 1678), 'band03': (1670, 1705)}
_HEAD = re.compile(rb'<head[^>]*>(.*?)</head>', re.S)


def head_years(data: bytes) -> tuple[int, int] | None:
    ys = []
    for h in _HEAD.findall(data[:120000])[:6][1:]:
        t = re.sub(rb'<[^>]+>', b'', h).decode('utf8', 'replace')
        if t.strip().startswith('[Fig'):
            continue
        ys += [int(y) for y in re.findall(r'(?<!\d)(1[4-7]\d\d)(?!\d)', t) if 1660 <= int(y) <= 1710]
    return (min(ys), max(ys)) if ys else None


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def scan(self, root, rec, ctx, counters):
        dl = C.doc_language(root)
        hits = []
        for el in root.iter():
            if C.ln(el.tag) != 'index':
                continue
            par = el.getparent()
            if par is not None and C.ln(par.tag) == 'index':
                continue
            name = el.get('indexName') or ''
            kind = _KIND.get(name)
            if kind is None:
                continue
            term = next((c for c in el.iter() if C.ln(c.tag) == 'term'), None)
            surface = C.surface_text(term) if term is not None else ''
            if not surface:
                continue
            ctxt = H.point_context(el) if ctx.want_context else None
            hits.append(H.make_hit(el, kind=kind, tag=f'index@{name}', surface=surface, zone='txt', role='mention',
                                   refs={}, lang=dl, context=ctxt))
        return hits

    def adjust(self, m, hit):
        key = V.term_key(m.surface)
        if not key:
            return None
        m.local_key = f'{SLUG}:{m.kind}:{key}'
        m.in_register = False
        return m

    def extract(self, rec, ctx):
        data = rec.read()
        root = C.parse_xml(data)
        if root is None:
            raise ValueError('unparseable XML')
        hits = self.scan(root, rec, ctx, None)
        yrs = head_years(data)
        if yrs:
            date = C.DocDate(yrs[0], yrs[1], 'header')
        else:
            band = next((b for b in _BAND if f'/{b}/' in rec.path), None)
            date = C.DocDate(*_BAND[band], 'collection') if band else C.DocDate()
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []


EXTRACTOR = Extractor()
