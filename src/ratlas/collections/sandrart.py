"""Joachim von Sandrart, Teutsche Academie (1675-1679) and Iconologia deorum (1680): DTA edition with the Sandrart.net
entity annotation (7 files, ~990k words). The richest editor-tagged corpus of the batch.

Licence: CC-BY-NC-3.0-DE (TEI header ``licence target=.../by-nc/3.0/de/``), tier T2. The DTA tarball is labelled
CC-BY-SA-4.0 / T1 but these files are CC BY-NC in their own headers; the stricter header licence is applied (the stand-alone
Sandrart catalogue entry is also T2). Raw: ``dtae/dta_1600-1699.zip!dta_komplett_1600-1699_2026-02-10/sandrart_*.TEI-P5.xml``
(the other DTA 1600-1699 files belong to the module ``dta_1600``).

What is special here
  * every running-text ``persName`` / ``placeName`` carries a ``@ref`` that is a space-separated list of URIs: the project
    id ``http://ta.sandrart.net/-person-4`` / ``-place-19`` first, then authorities: GND (d-nb.info), VIAF, Getty ULAN
    (old ``ULANFullDisplay?...subjectid=`` form) for persons; GeoNames and Getty TGN for places; Arachne /
    census.bbaw.de for antique places. There is no register file: the project id is the stable key,
    ``local_key = sandrart:person-4`` with in_register False (opaque), and also kept in auth.other as ``sandrart:person-4``;
  * ``<name type="artificialWork">`` (3,618 artworks with museum links) is neither person, place nor organisation: not read;
  * header: the author persName (Sandrart, GND) is role ``author``; DTA editors / respStmt are modern and skipped;
  * dates: the printing year of the volume in ``sourceDesc/biblFull/publicationStmt/date`` (1675, 1679, 1680; header);
    the DTA publication / import dates are never used.
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'sandrart'
COLLS = ['dtae/dta_1600-1699.zip!dta_komplett_1600-1699_2026-02-10/sandrart_']
RENASCOR_IDS = ['sandrart-teutsche-academie', 'dta']
LICENCE = 'CC-BY-NC-3.0-DE'
TIER = 'T2'
DESCRIPTION = 'Sandrart Teutsche Academie 1675-1680: persons and places with GND/VIAF/ULAN and GeoNames/TGN ids'
OPTIONS = {}

_SANDRART = re.compile(r'ta\.sandrart\.net/-?(person|place|artwork|org)-(\d+)', re.I)


class Extractor(GenericTEI):
    def adjust(self, m, hit):
        mm = _SANDRART.search(m.ref_raw or '')
        if mm:
            key = f'{mm.group(1).lower()}-{mm.group(2)}'
            m.local_key, m.in_register = f'sandrart:{key}', False
            m.auth['other'] = [x for x in m.auth['other'] if 'ta.sandrart.net' not in x]
            m.auth['other'].append(f'sandrart:{key}')
        if m.zone == 'hdr':
            m.context = None
        return m


EXTRACTOR = Extractor()
