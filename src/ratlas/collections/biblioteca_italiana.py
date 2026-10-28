"""Biblioteca Italiana (BibIt, Sapienza), TEI P3 (root TEI.2), 989 files / 30M words of Italian and Latin literature.

There is almost no editor-tagged entity in this corpus:
  * <persName>/<placeName>/<orgName> do not exist; the only typed tags are <rs type="name"> (person) and
    <rs type="luogo"> (place) in two files (bibit000316 L'Ermenegildo, bibit000996 Mantegna, Rime), 82 tags;
  * the 8,563 untyped <name> elements of 62 files (chronicles, verse) carry neither type nor @ref and cannot be
    split into persons and places without NER: they are NOT extracted (v1 = editor-tagged entities only);
  * no @ref / register: local_key None, no authority. Header <name> is encoder staff (not read).
  * dating: the TEI header only gives a century code and the modern edition's year; the harvest assigns each text a
    year from rebuild/bibit_source_selection (both files: 1550) -> date_source 'collection'.
v2 (issue 141): the header century code is the date. ``creation/date`` 500 = Cinquecento (1500-1599), 600 = Seicento, 700 = Settecento:
L'Ermenegildo (bibit000316) carries 700, an 18th-century text, so its 81 mentions are out of period (they were placed in 1550);
Mantegna's Rime (bibit000996) is 500. The range is kept (doc_year null for a century); the SELECTION_YEAR placeholder of the harvest is
only the fallback when a header has no code.
Only files with typed rs tags are read (iter_files); the other 987 files would give zero rows.
Licence: LicenseRef-BibIt-AttributionNonCommercial (free for personal/scientific use, attribution, no commercial
use; LICENSE_TIERS BibIT, T2).
"""
from __future__ import annotations

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'biblioteca_italiana'
COLLS = ['bibit_tei']
RENASCOR_IDS = ['biblioteca-italiana']
LICENCE = 'LicenseRef-BibIt-AttributionNonCommercial'
TIER = 'T2'
DESCRIPTION = 'Only the typed rs name/luogo tags of two Italian texts; untyped <name> elements are not extracted'
OPTIONS = {'type_map': {'name': 'person', 'luogo': 'place'}}

# year assigned to the text by the harvest's source selection (RQ/rebuild/bibit_source_selection_2026-06-13.tsv)
SELECTION_YEAR = {'bibit000316': 1550, 'bibit000996': 1550}


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.count('txt|rs') > 0]

    def doc_date(self, root, rec, ctx):
        for el in root.iter():
            if C.ln(el.tag) == 'creation':
                for d in el.iter():
                    t = C.norm(''.join(d.itertext())) if C.ln(d.tag) == 'date' else ''
                    if t.isdigit() and len(t) == 3 and 100 <= int(t) <= 800:
                        lo = 1000 + int(t)
                        return C.DocDate(max(lo, 1701) if lo == 1700 else lo, lo + 99, 'header')    # Settecento starts after 1700
        y = SELECTION_YEAR.get(rec.stem)
        return C.DocDate(y, y, 'collection') if y else C.DocDate()


EXTRACTOR = Extractor()
