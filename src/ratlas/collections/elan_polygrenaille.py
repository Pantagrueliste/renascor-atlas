"""ELAN / POLY-GRENAILLE: Francois de Grenaille, La Mode (1642), one TEI file (75,765 words).

Practically no entity markup: one <persName corresp="#grenaille"> on the title page ("Par FRANCOIS DE GRENAILLE ...",
role 'author'; the id is an opaque local key, there is no register) and a placeholder <settlement>[A REMPLIR]</settlement>
in the empty msIdentifier of the header (dropped). Modern editors (principal, respStmt, sponsors) are skipped by the
scanner. Dating: the work is of 1642 (default_years, date_source 'collection'); the header has no date but the edition's
(2017).
Licence: the header states CC BY-NC-SA 4.0 (the entry says "Mixed"), stricter than LICENSE_TIERS
(LicenseRef-CC-BY-unversioned, T1): the stricter is applied, tier T2 (non-commercial).
"""
from __future__ import annotations

from .generic_tei import GenericTEI

SLUG = 'elan_polygrenaille'
COLLS = ['elan/elan__polygrenaille']
RENASCOR_IDS = ['elan-polygrenaille']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Grenaille, La Mode (1642): a single tagged author name'
OPTIONS = {'default_years': (1642, 1642), 'opaque_keys': True}


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        from .. import common as C
        return C.DocDate(1642, 1642, 'collection')

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        out = []
        for h in hits:
            if h.surface.startswith('[A REMPLIR'):
                continue
            if h.name == 'persName' and 'titlePart' in h.ancestors:
                h.role = 'author'
            out.append(h)
        return out


EXTRACTOR = Extractor()
