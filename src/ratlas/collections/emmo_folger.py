"""Folger EMMO (Early Modern Manuscripts Online): 100 diplomatic transcriptions (Bagot papers etc.) in eng_dh/emmo.zip.

The text marks <name xml:id="name-001"> (untyped, personal names according to the project's tagUsage) and
<placeName>; there is no register and no @ref/@key (the xml:id is file-local), so nothing resolves.
Special handling:
  * untyped <name> in the text is read as a person (the scanner skips untyped names, so type="person" is set on the
    in-memory tree first); <name> in the header (respStmt: encoders) is not touched;
  * surface = diplomatic reading (<orig> kept, <reg> dropped), e.g. 'mr Nicholas Allin';
  * dating: header msDesc/history/origin (@when or text), date_source 'header' (the publicationStmt date is the digital one);
  * the zip holds each file twice (dup=true on 100 records): skipped.
Licence: CC BY-SA 4.0, stated in the TEI headers and quoted by the Renascor catalogue; reclassified to T1 by the
owner decision of 2026-11-04 (data/licences.tsv decision column).
"""
from __future__ import annotations

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'emmo_folger'
COLLS = ['eng_dh/emmo.zip']
RENASCOR_IDS = ['emmo-folger']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Folger EMMO manuscripts 1552-1697: untyped name (persons) and placeName, no register'
OPTIONS = {}


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        for el in root.iter():
            if C.ln(el.tag) == 'origin':
                rg = C.date_range(el, 1000, 2100)
                if rg:
                    return C.DocDate(rg[0], rg[1], 'header')
        return super().doc_date(root, rec, ctx)

    def scan(self, root, rec, ctx, counters):
        for el in root.iter('{*}name'):
            if el.get('type') is None:
                p, in_hdr = el.getparent(), False
                while p is not None:
                    if C.ln(p.tag) == 'teiHeader':
                        in_hdr = True
                        break
                    p = p.getparent()
                if not in_hdr:
                    el.set('type', 'person')
        return super().scan(root, rec, ctx, counters)


EXTRACTOR = Extractor()
