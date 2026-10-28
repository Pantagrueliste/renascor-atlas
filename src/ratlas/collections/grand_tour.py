"""Grand Tour digital: TEI dataset of early modern travel testimonies (HAB Wolfenbuettel, Zenodo 23055358).

Worked example of a collection module. What is special here:
  * mentions are <rs type="person|place|org" ref="#id"> in the journals (Reiseberichte/), the register is in
    three separate files (register/listPerson|listPlace|listOrg_grand_tour.xml), found by the generic
    is_register_file() and loaded once by prepare(); journals reference them with '#id';
  * the journals carry no header date, so the document years come from the <date when> elements of the text
    (date_source 'text'); the register has no coordinates, register/itinerar_wagener.json adds lat/lon to places
    through their GeoNames id (coord_source 'json:itinerar_wagener');
  * introductions, edition rules and the index are modern editorial text: not read.
v2: each mention takes the date of the last <date when> before it in reading order (483, 139 and 51 dated entries in the three
diaries), not the single midpoint of the journal (1667, 1655, 1689); the journal's span stays for a mention before every date.
Known gaps: Cod. Guelf. 221 has ref="#" on many tags (kept as unresolved, local_key None).
"""
from __future__ import annotations

import json

from .. import common as C
from .. import helpers_v2 as V
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'grand_tour'
COLLS = ['renascor_2026-10/grand-tour-digital-tei-dataset-of-early-modern-travel-testimonies-grand-tour-tei']
RENASCOR_IDS = ['grand-tour-digital-tei-dataset-of-early-modern-travel-testimonies-grand-tour-tei']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'German travel journals 1652-1689 with a manually corrected person/place/org register'
OPTIONS = {}


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        recs = super().iter_files(ctx, include_dups)
        return [r for r in recs if '/Reiseberichte/' in r.path and not r.name.startswith('index_')
                or '/register/' in r.path]

    def doc_date(self, root, rec, ctx):
        years = []
        for el in root.iter():
            if C.ln(el.tag) == 'date' and el.get('when'):
                y = C._attr_year(el.get('when'))
                if y:
                    years.append(y)
        if years:
            return C.DocDate(min(years), max(years), 'text')
        return super().doc_date(root, rec, ctx)

    def extract_root(self, root, rec, ctx):
        self._rd = V.RunningDate(root, 1500, 1750)
        return super().extract_root(root, rec, ctx)

    def adjust(self, m, hit):
        rd = getattr(self, '_rd', None)
        rg = rd.range_at(hit.el) if rd else None
        if rg:
            V.set_range(m, rg[0], rg[1], 'text')
        return m

    def prepare(self, ctx):
        ents = super().prepare(ctx)
        by_gn = {e.auth['geonames']: e for e in ents if e.kind == 'place' and e.auth.get('geonames')}
        # the JSON is not an XML record in the scan: read it from the same archive
        arch = next((r.archive for r in ctx.all_files if '/register/' in r.path), None)
        if arch:
            try:
                rows = json.loads(C.read_raw(arch + '!grand-tour-tei/register/itinerar_wagener.json'))['rows']
            except (KeyError, OSError, ValueError):
                rows = []
            for row in rows:
                e = by_gn.get(str(row.get('GeoNames ID') or '').strip())
                if e is not None and e.lat is None:
                    try:
                        e.lat, e.lon = float(row['latitude']), float(row['longitude'])
                        e.coord_source = 'json:itinerar_wagener'
                    except (KeyError, ValueError):
                        pass
        return ents


EXTRACTOR = Extractor()
