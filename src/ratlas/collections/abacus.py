"""ABaC:us Austrian Baroque Corpus (queue3/abacus.zip): 5 sermon editions by Abraham a Sancta Clara.

Mentions: <persName key="..." type="hist|bibl|myth"> and <placeName key="..." type="city|coun|stre|..."> in the
running text (the editions are word-tokenised, <w> inside the tags). <rs type="death|profession|..." ref="#term"> is a
thematic term index, not a name: never read. Registers: data/index/output/abacus-index_persnames.xml (998 persons)
and abacus-index_placenames.xml (338 places) with name variants and per-file mention counts. A mention's @key is the @key
of the main name of a register entry, not always its xml:id (the id is ascii-folded: xml:id 'matthaus1', key
'matthäus1'): both are indexed as aliases. No GND/VIAF/GeoNames, no coordinates in the harvest.
note_type of persons: hist -> person, bibl -> biblical, myth -> mythological; of places: the place type (city, coun,
stre, vill, rive, ...). Dating: imprint date in the header of each edition (1680, 1680, 1681, 1693, 1710; Todten_Capelle
1710 is out of the period and kept with in_period False). The unzipped duplicate copy (abc-data-main/) is dup=true.
msIdentifier/settlement (place of the holding library) is dropped.
Licence: the TEI headers say CC BY-NC-SA 4.0 but LICENSE_TIERS ABaCus_Baroque is LicenseRef-unclear, T3 (recipe-only,
not shipped); the stricter T3 is kept: used internally, no context stored.

v2: re-harvest of 2026-10-17 (abc-data main at 0edf72e, 14 Jul 2026): only the teiHeader changed (editions, abacus.xml and
the two registers have the same 998 persons / 338 places). The headers now say CC BY 4.0 (the repository LICENSE.md still
says CC BY-NC-SA 4.0): the tier is the owner's decision and stays T3. Registers: note_type biblical / mythological
give person_class biblical / mythological; ``type=moun|rive|coun`` of a place is kept in note_type (the geocoder must read it).
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'abacus'
COLLS = V.pick(['queue3/abacus.zip'], [V.NEW + '/abacus/abacus.zip'])
RENASCOR_IDS = ['abacus-baroque']
LICENCE = 'LicenseRef-unclear'
TIER = 'T3'
DESCRIPTION = 'Abraham a Sancta Clara sermons (ABaC:us) with persons/places registers'
OPTIONS = {}

_PERS_NOTE = {'hist': 'person', 'bibl': 'biblical', 'myth': 'mythological', 'undefined': 'undefined'}


class Extractor(GenericTEI):
    def adjust(self, m, hit):
        if 'msIdentifier' in hit.ancestors:      # <settlement> of the holding library (Wien, Graz, Melk...): not a mention
            return None
        return m

    def parse_register_file(self, rec, ctx, root):
        out = super().parse_register_file(rec, ctx, root)
        byid = {e.get(C.XML_ID): e for e in root.iter() if e.get(C.XML_ID)}
        res = []
        for ent, aliases in out:
            if ent.kind == 'person' and ent.note_type:
                ent.note_type = _PERS_NOTE.get(ent.note_type, ent.note_type)
            el = byid.get(aliases[0]) if aliases else None
            # mentions cite the @key of the main name ('matthäus1'), the xml:id is its ascii-folded form ('matthaus1')
            extra = [c.get('key') for c in el if c.get('key')] if el is not None else []
            res.append((ent, list(aliases) + extra))
        return res


EXTRACTOR = Extractor()
