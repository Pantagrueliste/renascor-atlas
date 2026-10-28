"""Religionsfrieden-Quellenedition (IEG Mainz / ULB Darmstadt): peace treaties and religious-peace edicts 1485-1788.

Conventions of this edition:
  * <rs type="person|place" ref="#id"> in the running text (type="bibl" is a source citation: not an entity);
  * every file carries its own register in the header (profileDesc/particDesc/listPerson, settingDesc/listPlace)
    with GND / VIAF (persons) and GeoNames or GND urls (places, www.geonames.org/ID/slug.html: normalised by
    common.classify_token). Registers repeat per file with different ids, so keys are file-local
    ('slug:<doc_id>#<id>'); the merge across files is a later stage (GND / GeoNames);
  * no document date in the header: the year is taken from the title of the treaty ('... (19. Maerz 1563)').
    Four files are after 1700 (1712, 1781-83, 1787, 1788) and come out with in_period False;
  * index.html (non-TEI) is not read. No coordinates in the files (GeoNames ids give them in a later stage).
Licence: CC-BY-4.0 in every TEI header and in LICENSE_TIERS (T1).
"""
from __future__ import annotations

from .. import helpers_batch4 as H
from .generic_tei import GenericTEI

SLUG = 'religionsfrieden_quellenedition'
COLLS = ['renascor_2026-10/religionsfrieden-quellenedition']
RENASCOR_IDS = ['religionsfrieden-quellenedition']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Peace treaties and edicts 1485-1788 with per-file persons/places registers (GND, GeoNames)'
OPTIONS = {}


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.path.lower().endswith('.xml')]

    def doc_date(self, root, rec, ctx):
        return H.title_year_date(root)


EXTRACTOR = Extractor()
