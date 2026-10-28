"""CAMENA Neo-Latin historical and political prose (Univ. Mannheim, github camena-neolatinlit/historicapolitica),
316 TEI.2 / teixlite files, 13.5 M words.

Licence: CC-BY-SA-4.0, tier T1 (LICENSE_TIERS: CAMENA_NeoLatin_HistThes; repository LICENSE.md).

Entity markup is almost absent: 22 <name type=person> in a single file (Pantaleon_res_memorabiles_1.xml: the authors
cited in a list of historians). They are kept as person mentions (tag name@person); typed myth / nation / religion / org
names would be mapped like corpus_corporum (none occur). No ids, no register. The header respStmt names (Ruediger Niehl,
...) are encoders and never mentions.
Dating: the file has no date element; the imprint year of the source is read from sourceDesc/bibl ('Basel: [Guarin], 1581.',
first year 1000-1800), date_source 'header'. Reprints 1700-1760 are dated and out of period.
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'camena_historicapolitica'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/historicapolitica']
RENASCOR_IDS = ['camena']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'CAMENA historical-political Neo-Latin prose; 22 typed names in one file'
OPTIONS = {'type_map': {'myth': 'person', 'religion': 'person', 'nation': 'org', 'populus': 'org'}}

_YEAR = re.compile(r'(?<!\d)(1[0-7]\d\d)(?!\d)')


def _imprint_year(root, rec):
    for el in root.iter():
        if C.ln(el.tag) == 'sourceDesc':
            m = _YEAR.search(C.surface_text(el))
            if m:
                return C.DocDate(int(m.group(1)), int(m.group(1)), 'header')
    return None


class Extractor(GenericTEI):
    def doc_date(self, root, rec, ctx):
        d = C.header_dates(root)
        if d.known and d.source != 'letter':
            return d
        return _imprint_year(root, rec) or d


EXTRACTOR = Extractor()
