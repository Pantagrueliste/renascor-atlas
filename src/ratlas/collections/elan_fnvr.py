"""ELAN / FNVR, Fictions narratives en vers de la Renaissance (Litt&Arts, Grenoble): 6 TEI verse texts.

Persons are tagged in two of the six texts only: palamon (Anne de Graville, Palamon et Arcita, 568 persName) and
eurialus (Eurialus et Lucretia, 862 persName). Each has its register in teiHeader/sourceDesc/listPerson (123 and 83
persons, no idno): <persName ref="#slug"> resolves to <person xml:id="slug"> of the same file (local_key
'elan_fnvr:<doc>#<slug>'). The characters are literary figures (Palamon, Arcita, Emilia, but also gods and heroes
of the classical subject matter): the register has no type, so note_type stays empty rather than guessed; the text
is a prose/verse adaptation of fiction, not a mention of historical persons. The four other texts carry 3-4 header
persName at most (principal / respStmt: modern project staff, never read).
Dating: bibl / source dates in the header (c. 1493 for Eurialus); Palamon et Arcita has no date in its header:
default_years (1493, 1537), the span of the FNVR source editions, date_source 'collection'.
v2 (issue 117): all the register persons are characters of two prose romances (Palamon, Arcita, Emilia, Eurialus, Lucretia...) or
figures of the classical subject matter (Thésée, Actéon, Ménélas, César). They are flagged: ``note_type`` 'fictional' (person_class
'character': excluded from the default counts); the gods and mythical heroes of the register (Actéon, Adonis, Vénus, Jupiter...)
are 'mythological' (listed in ``_MYTHICAL``); nothing here is a historical person. No year is invented for Palamon et Arcita: the
collection span 1493-1537 is a range (doc_year null).
Known wart: the context of the first mentions of a text can start with names of the header listPerson (common.py
builds contexts from all text in document order).
Licence: the TEI headers and the catalogue say CC BY-NC-SA 4.0; LICENSE_TIERS has LicenseRef-CC-BY-unversioned (T1):
conflict, the stricter header licence is used (T2).
"""
from __future__ import annotations

import unicodedata

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'elan_fnvr'
COLLS = ['elan/elan__fnvr']
RENASCOR_IDS = ['elan-fnvr']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Renaissance verse fictions (Palamon, Eurialus): characters with in-file registers'
OPTIONS = {'default_years': (1493, 1537)}

# gods and heroes of the classical subject matter, by folded name (the register has no type)
_MYTHICAL = {'acteon', 'adonis', 'venus', 'jupiter', 'junon', 'mars', 'diane', 'apollon', 'minerve', 'saturne', 'mercure', 'cupidon',
             'amour', 'neptune', 'pluton', 'proserpine', 'hercule', 'thesee', 'menelas', 'achille', 'ulysse', 'hector', 'priam',
             'helene', 'agamemnon', 'jason', 'perseus', 'persee', 'bacchus', 'cerberus', 'cerbere', 'hippolyte', 'cassandre',
             'andromede', 'pallas', 'pallas athena', 'hymenee', 'pan', 'circe', 'medee', 'ariane', 'phaeton', 'orphee'}


def _fold(s: str) -> str:
    s = unicodedata.normalize('NFKD', (s or '').casefold())
    return ''.join(ch for ch in s if not unicodedata.combining(ch)).strip()


class Extractor(GenericTEI):

    def extract_root(self, root, rec, ctx):
        ms, ents = super().extract_root(root, rec, ctx)
        for e in ents:
            e.note_type = 'mythological' if _fold(e.name).split(' ')[0] in _MYTHICAL or _fold(e.name) in _MYTHICAL else 'fictional'
        return ms, ents


EXTRACTOR = Extractor()
