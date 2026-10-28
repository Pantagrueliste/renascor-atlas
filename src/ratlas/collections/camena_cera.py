"""CAMENA CERA (Universitaet Mannheim; printed Neo-Latin letter collections: Acidalius, Ascham, Baier...),
100 TEI.2 files (ISO-8859-1), GitHub republication of nevenjovanovic.

Conventions of this edition:
  * <name type="person|place"> in running text; no @ref/@key, no register: surface strings only (local_key None);
  * the letters are not encoded as correspDesc or separate items and the header has no sender/addressee/place/date
    metadata: roles are all 'mention';
  * 45 of 100 files are editions of 1700-1765: they are extracted but come out with in_period False
    (the colls.json count of 432 in-period person tags is the in_period True rows);
  * dating: imprint year in sourceDesc/bibl (the edition year, not the composition year), source 'header'. v2: an imprint after 1700
    (Burman, Sylloge, Leiden 1727: letters of Lipsius and his correspondents, 1564-1606) says nothing about the letters, so such a
    volume is undated (in_period None) instead of out of period (it was 4,003 of the 4,435 mentions);
  * roles (v2): the tagged letters (Burman Sylloge, Goldast) have a head 'EPISTOLA n. SENDER ADDRESSEE S.D. Place': the first person
    of a head is the sender, the second the addressee, the place the destination (sent_to);
  * untyped <name> (201) is skipped.
Licence: CC-BY-SA-4.0 (LICENSE_TIERS CAMENA_CERA_Letters, T1).
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch4 as H
from .generic_tei import GenericTEI

SLUG = 'camena_cera'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/cera']
RENASCOR_IDS = ['camena']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Printed Neo-Latin epistolae: typed person and place names (surface forms only)'
OPTIONS = {}


class Extractor(GenericTEI):
    def doc_date(self, root, rec, ctx):
        d = H.sourcedesc_year_date(root)
        if d.year_from is not None and d.year_from > 1700:
            return C.DocDate()                       # the imprint of a later edition is not the date of the letters
        return d

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        seen: dict = {}
        for h in hits:
            p = h.el.getparent()
            if p is None or C.ln(p.tag) != 'head' or not C.surface_text(p).upper().startswith('EPISTOLA'):
                continue
            n = seen.setdefault(id(p), [0, p])      # keep the head alive: lxml proxies are short-lived
            if h.kind == 'person' and n[0] < 2:
                h.role = 'sender' if n[0] == 0 else 'addressee'
                n[0] += 1
            elif h.kind == 'place':
                h.role = 'sent_to'
        return hits


EXTRACTOR = Extractor()
