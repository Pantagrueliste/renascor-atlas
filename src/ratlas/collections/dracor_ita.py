"""DraCor Italian drama corpus (78 plays, Biblioteca Italiana based, Wikidata-linked), 1451-1695.

Licence per file (from the TEI headers; LICENSE_TIERS DraCorItalian is T2 'CC0-1.0 AND CC-BY-NC-4.0 AND
LicenseRef-BibliotecaItaliana-NonCommercial'): 62 files carry the Biblioteca Italiana wording "Ogni uso commerciale e
vietato" (LicenseRef-BibliotecaItaliana-NonCommercial, T2, the module default) and 16 files a CC0 1.0 licence element
(set per file in extract_root: CC0-1.0, T1). Raw: ``originals_2026-10-15/dracor_api_tei/ita/``.

What the editors tagged: the cast (``particDesc/listPerson``, 959 characters, ~16% with an ``@ana`` Wikidata URL, e.g.
Orpheo Q174353) and the author (``titleStmt/author/persName`` + ``idno type=wikidata``, 48 authors). Not read: the
3,987 untyped, unreferenced ``<name>`` in the verse (no type, no id: not classifiable without NER) and ``sp/@who``.
Cast mentions are role ``cast`` (zone hdr); register entries are characters (``note_type`` 'character', mythological,
historical or fictional: the edition does not say). No places are tagged in this corpus.
62 files carry bibit ids that also exist in ``bibit_tei`` (same text, other encoding): renascor_id is
``dracor;biblioteca-italiana`` for these, ``dracor`` for the others; deduplicate against biblioteca-italiana before
counting texts.  Dates: standOff events written > premiere > print (header).
"""
from __future__ import annotations

import re

from .. import helpers_batch0 as H

SLUG = 'dracor_ita'
COLLS = ['originals_2026-10-15/dracor_api_tei/ita']
RENASCOR_IDS = ['dracor', 'biblioteca-italiana']
LICENCE = 'LicenseRef-BibliotecaItaliana-NonCommercial'   # strictest of the two; CC0 files are set per file
TIER = 'T2'
DESCRIPTION = 'Italian plays 1451-1695: cast lists with Wikidata ids, authors'
OPTIONS = {}


class Extractor(H.DracorTEI):
    def file_licence(self, root, rec):
        for el in H.iter_local(root, 'licence'):
            if 'publicdomain/zero' in (el.get('target') or ''):
                return ('CC0-1.0', 'T1')
        return None

    def file_renascor_id(self, root, rec, ctx):
        for el in H.iter_local(root, 'publisher'):
            if 'biblioteca italiana' in H.text_of(el).lower():
                return 'dracor;biblioteca-italiana'
        return 'dracor'


EXTRACTOR = Extractor()
