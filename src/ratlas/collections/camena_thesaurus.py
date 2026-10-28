"""CAMENA 'thesaurus' sub-corpus (Universitaet Mannheim; Neo-Latin reference works, historical and biographical
dictionaries: Alegambe, Avity, Imperiali...), 218 TEI.2 files (ISO-8859-1), GitHub republication of nevenjovanovic.

Conventions of this edition:
  * proper names in running text are <name type="person|place|myth|religion|nation|org"> (plus a few <rs type=...>);
    NO @ref/@key and NO register: surface strings only (inflected Latin forms), local_key None, no authority;
  * mapping: person -> person, place -> place, org -> org; myth (417) and religion (406, mostly Christus/Deus)
    are mapped to kind 'person' and recognisable only by the tag label name@myth / name@religion (the register
    flag note_type does not exist without a register); 'nation' (peoples, 128) and untyped <name> (482) are skipped;
  * dating: imprint year in sourceDesc/bibl (the edition year, 1550-1699), source 'header';
  * only ~14 tags per 1,000 words: this is an inventory of surface forms, not an index.
Licence: CC-BY-SA-4.0 (LICENSE_TIERS CAMENA_NeoLatin_HistThes, T1).
"""
from __future__ import annotations

from .. import helpers_batch4 as H
from .generic_tei import GenericTEI

SLUG = 'camena_thesaurus'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/thesaurus']
RENASCOR_IDS = ['camena']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Neo-Latin reference works: typed proper names (surface forms only)'
OPTIONS = {'type_map': {'myth': 'person', 'religion': 'person'}}


class Extractor(GenericTEI):
    def doc_date(self, root, rec, ctx):
        return H.sourcedesc_year_date(root)


EXTRACTOR = Extractor()
