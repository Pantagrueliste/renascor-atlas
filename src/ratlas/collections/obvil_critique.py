"""OBVIL 'critique' corpus (Sorbonne Labex OBVIL, Teinte TEI): only the two files dated 1450-1700.

The folder holds 657 files of French literary criticism, 99.7% from the 18th to the 20th century; editor tagging there
is untyped <persName>/<name> with no ref, and the header keys are author keys. In scope for the atlas period are
la-bruyere/la-bruyere_ouvrages-esprit_1694.xml (La Bruyere, Des ouvrages de l'esprit, 1694; 12 bare persName, no ref)
and la-grange/la-grange_preface.xml (Preface to Moliere's works, 1682; no tagged name). These are selected by path
(COLLS); the 8 files of 1700-1750 and the rest of the corpus are not read. Header authors are <author key="...">
attributes, not persName tags, so they are not mentions. No register, no authority ids, no coordinates.
Dating: creation/date in the header (1694, 1682).
v2 (issue 274): the 12 bare persName of La Bruyère's *Des ouvrages de l'esprit* (Ariste, Zoïle, Zélotes, Hermodore, Arsène...) are the
fictional characters of *Les Caractères*, not historical persons: person_class 'character' (excluded from the default counts).
Faceties 1557 (obvil_faceties/xml_orig/xml_orig_Faceties_011.xml: 2 persName, 1 placeName) is also in the period, but belongs to another
OBVIL corpus folder than this module's selection: it is listed here for a later decision, not read.
Licence: CC BY-NC-ND 3.0 FR in each header (OBVIL, availability status restricted); LICENSE_TIERS OBVIL_French T2.
"""
from __future__ import annotations

SLUG = 'obvil_critique'
COLLS = ['obvil_critique/la-bruyere/la-bruyere_ouvrages-esprit_1694.xml',
         'obvil_critique/la-grange/la-grange_preface.xml']
RENASCOR_IDS = ['obvil-french']
LICENCE = 'CC-BY-NC-ND-3.0-FR'
TIER = 'T2'
DESCRIPTION = 'OBVIL critique, the two in-period files (La Bruyere 1694, La Grange 1682)'
OPTIONS = {}

from .. import classify  # noqa: E402,F401
from .generic_tei import GenericTEI  # noqa: E402


class Extractor(GenericTEI):
    def adjust(self, m, hit):
        if m.kind == 'person' and 'la-bruyere' in m.source_path:
            m.person_class = 'character'
        return m


EXTRACTOR = Extractor()
