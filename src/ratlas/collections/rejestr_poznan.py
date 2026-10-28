"""Rejestr kontrybucji diecezji poznanskiej z 1561 r. (LHDB / geo-ecclesiae, Krakow): one TEI file, two manuscripts.

Tagged: <rs type="person|place|institution" ref="Qnnn"> in the running text and the list rows: 811 persons, 112 places
(deaneries), 1,733 institutions (parishes, offices such as 'canonicus Posnaniensis'; kind org). One rs has no type
and is not read. The Qnnn are NOT Wikidata ids (Q207 is the project's id for Sigismund August): they are identifiers
of the LHDB wiki / geo-ecclesiae database. common.classify_token would read them as Wikidata, so adjust() clears
auth and gives the stable local key 'rejestr_poznan:Qnnn' (in_register False). The file has no register (standOff
holds only an apparatus), no coordinates: no entities.
v2: the single file transcribes the same 1561 register TWICE, as two manuscripts (div ASK91, 1,328 rs, and div ASK3, 1,329 rs, with
parallel seg ids ASK91.v251 / ASK3.v251): every person, parish and place was counted twice. Each witness now has its own doc_id;
ASK91 is the canonical one and the rows of ASK3 carry ``excluded_reason`` 'parallel_witness' (about 1,330 mentions are left).
Dating: the register is of 1561 (the title); the header has no date: default_years (1561, 1561), date_source 'collection'.
Licence: CC BY-NC-SA 4.0 (catalogue entry, inferred from the LHDB research-data page; the TEI itself carries no
licence); LICENSE_TIERS T2.
"""
from __future__ import annotations

from .. import common as C
from ..model import empty_auth
from .generic_tei import GenericTEI

SLUG = 'rejestr_poznan'
COLLS = ['renascor_2026-10/rejestr-kontrybucji-poznan-1561']
RENASCOR_IDS = ['rejestr-kontrybucji-poznan-1561']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Contribution register of the diocese of Poznan 1561: persons, parishes/offices, deaneries (LHDB ids)'
OPTIONS = {'default_years': (1561, 1561)}


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def adjust(self, m, hit):
        p = hit.el.getparent()
        while p is not None:
            wid = p.get(C.XML_ID)
            if C.ln(p.tag) == 'div' and wid in ('ASK91', 'ASK3'):
                m.doc_id = wid
                if wid == 'ASK3':
                    m.excluded_reason = 'parallel_witness'
                break
            p = p.getparent()
        ref = (hit.refs.get('ref') or '').strip()
        if ref:
            m.auth = empty_auth()                    # LHDB ids, not Wikidata
            m.local_key = C.make_key(SLUG, ref)
            m.in_register = False
        return m


EXTRACTOR = Extractor()
