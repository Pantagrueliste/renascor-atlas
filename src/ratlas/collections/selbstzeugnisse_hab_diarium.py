"""Duke August the Younger of Braunschweig-Wolfenbuettel, Ephemerides. Sive Diarium (1594-1635), HAB edition
(I. H. Ralle): the single file diarium.xml of the 'Selbstzeugnisse' collection. (The sz2 zip of the same collection
is a different edition: module selbstzeugnisse_hab_sz2.)

Conventions of this edition:
  * <rs type="person|place|org" ref="#Id"> (or ref="register_personen.xml#id") in the running text; 'libri' (books)
    is not an entity; typos ('lirbi', 'placec' -> place) are mapped;
  * the register sits in the same file (text/back: listPerson 303 persons with GND on persName, birth/death,
    listPlace 2,466 places with <geo>lat lon</geo> and a Getty TGN <idno type="URI">): ids are edition-wide, one file,
    so keys are shared ('<slug>:<id>'); coordinates come from the tei:geo of the register (real source data);
  * the front matter (introduction by the modern editor) also contains rs tags; they are read as mentions;
  * dating: dates in the running text (<date when>, 1594-1635), source 'text'. v2: each mention takes the date of the last
    <date> before it in reading order (1,697 dates for 1,649 entries), not the one midpoint 1614 of the whole diary; the
    document span stays in doc_year_from / to only for a mention that precedes every date.
Licence: CC-BY-SA-4.0 (TEI header: CC-BY-SA 4.0; LICENSE_TIERS Renascor_selbstzeugnisse-hab T1).
"""
from __future__ import annotations

from .. import helpers_batch4 as H
from .. import helpers_v2 as V

SLUG = 'selbstzeugnisse_hab_diarium'
COLLS = ['renascor_2026-10/selbstzeugnisse-hab/files/diarium.xml']
RENASCOR_IDS = ['selbstzeugnisse-hab']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Diary of Duke August the Younger 1594-1635 with a coordinate-bearing place register'
OPTIONS = {'opaque_keys': True, 'type_map': {'placec': 'place', 'lirbi': 'person'}}


class Extractor(H.SharedRegisterTEI):
    def doc_date(self, root, rec, ctx):
        return H.text_dates_doc_date(root)

    def extract_root(self, root, rec, ctx):
        self._rd = V.RunningDate(root, 1500, 1700)
        return super().extract_root(root, rec, ctx)

    def adjust(self, m, hit):
        rd = getattr(self, '_rd', None)
        r = rd.range_at(hit.el) if rd else None
        if r:
            V.set_range(m, r[0], r[1], 'text')
        return m


EXTRACTOR = Extractor()
