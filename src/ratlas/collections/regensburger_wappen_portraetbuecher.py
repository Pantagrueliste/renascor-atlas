"""Regensburger Wappen- und Portraetbuecher, Band 06 Vormundamt (Univ. Graz / Stadtarchiv Regensburg): one TEI file
(rpb-06-text.xml), lists of officials 1537-1700 (continued to 1787).

Licence: LicenseRef-AllRightsReserved, tier T3 (LICENSE_TIERS: Renascor_regensburger-wappen-portraetbuecher; project
imprint, copyright, no licence in the TEI). Internal work only; no context snippets.

Mentions: 1,674 <persName key="info:fedora/o:rpb-c-0010"> (the GAMS object id of the person); names are often split
into forename / surname / roleName children (the surface is the whole text). No place tags. local_key =
'regensburger_wappen_portraetbuecher:rpb-c-0010' (opaque, in_register False: the person objects are not in the harvest);
no authority ids (the header keys TGNid:7003097 Graz and TGNid:7013496 Regensburg belong to the digital edition and the
archive, not to mentions, and are skipped with creation/placeName and msIdentifier/settlement).
Dating per mention: the file spans 1537-1787 and the persons are listed under year headings <date when="YYYY">, so every
mention takes the year of the nearest preceding date in the text (date_source 'text'); persons before the first date
(preface 1649 excepted) stay undated. doc_year_from = doc_year_to = that year; in_period is therefore decided per mention
(1700 and earlier in period, the continuation to 1787 not).
"""
from __future__ import annotations

import re

from .. import common as C
from ..model import best_year, period_overlap
from .generic_tei import GenericTEI

SLUG = 'regensburger_wappen_portraetbuecher'
COLLS = ['renascor_2026-10/regensburger-wappen-portraetbuecher']
RENASCOR_IDS = ['regensburger-wappen-portraetbuecher']
LICENCE = 'LicenseRef-AllRightsReserved'
TIER = 'T3'
DESCRIPTION = 'Regensburg Vormundamt officials 1537-1787; GAMS person keys, per-mention year from year headings'
OPTIONS = {}

_PID = re.compile(r'(rpb-c-\d+)')


def person_years(root) -> dict:
    """persName element -> year of the nearest preceding <date when> in the text (document order)."""
    cur = None
    out = {}
    in_text = False
    for el in root.iter():
        n = C.ln(el.tag)
        if n == 'text':
            in_text = True
        if not in_text or n is None:
            continue
        if n == 'date':
            y = C._attr_year(el.get('when') or el.get('notBefore') or '')
            if y:
                cur = y
        elif n == 'persName':
            out[el] = cur
    return out


class Extractor(GenericTEI):
    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=False, doc_lang=C.doc_language(root),
                               counters=counters, skip_ancestors=('msIdentifier', 'creation'))

    def extract_root(self, root, rec, ctx):
        years = person_years(root)
        hits = self.scan(root, rec, ctx, None)

        def adj(m, hit):
            mm = _PID.search(m.ref_raw or '')
            if mm:
                m.local_key, m.in_register = f'{SLUG}:{mm.group(1)}', False
            y = years.get(hit.el)
            if y:
                m.doc_year = m.doc_year_from = m.doc_year_to = y
                m.date_source, m.in_period = 'text', period_overlap(y, y)
            return m

        date = C.DocDate()
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=adj), []


EXTRACTOR = Extractor()
