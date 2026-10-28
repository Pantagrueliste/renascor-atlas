"""GerManC - Historical Corpus of German 1650-1800 (OTA 2544; de_dh/germanc.zip, folder TEI/ only).

336 texts in 8 genres x 3 periods; the file name carries genre_period_region_year_title (DRAM_P1_NoD_1673_Leonilda).
The zip also holds the same texts as LING-GATE/ (a second format of the same content): excluded, only TEI/ is read.

Entities are linguistic annotation in the running text: <rs type="person|place|org|god|ship|currency" key="KPa">
and <name type=...>. Handling:
  * kinds: person, place, org as usual; type 'god' is mapped to person (tag 'rs@god' keeps the type); ship and
    currency are not entities of the atlas and are skipped (counted by the scanner as untyped).
  * @key is a mnemonic that is only meaningful inside one text (KPa, Wa, RT): local_key '<slug>:<doc_id>#<key>',
    in_register False. No register, no authority ids, no coordinates. Cross-text identity needs reconciliation.
  * anaphora: 85% of the person tags (and some place/org tags) are pronouns or deictic adverbs ('er', 'ihm', 'dahier',
    'demselben') annotated for coreference. They are kept but get the non-standard role 'anaphor' (helpers_batch1.is_anaphoric: every word of
    the surface is a pronoun or deictic); role 'mention' is therefore the set of nominal mentions.
  * dating: the year in the file name (date_source 'filename'): the header date is the source's own publication
    date for newspapers but absent for the others. 1650-1699 texts are in period; P2 (1700-1749) and P3 are not.
Licence: CC-BY-NC-SA-3.0 (LICENSE_TIERS, 'contains NC material'), T2.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch1 as H

SLUG = 'germanc'
COLLS = ['de_dh/germanc.zip!TEI/']
RENASCOR_IDS = ['germanc']
LICENCE = 'CC-BY-NC-SA-3.0'
TIER = 'T2'
DESCRIPTION = 'GerManC 1650-1800: rs/name annotation (person, place, org, god) with text-local keys'
OPTIONS = {'type_map': {'god': 'person', 'gods': 'person'}}


class Extractor(H.BaseTEI):
    def doc_date(self, root, rec, ctx):
        y = C.year_from_filename(rec.name, 1600, 1810)
        if y:
            return C.DocDate(y, y, 'filename')
        return super().doc_date(root, rec, ctx)

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is None:
            return None
        key = hit.refs.get('key')
        if key and key.strip() and m.local_key is None:
            H.set_opaque(m, SLUG, key.strip().split()[0], m.doc_id)
        if m.role == 'mention' and H.is_anaphoric(m.surface):
            m.role = 'anaphor'
        return m


EXTRACTOR = Extractor()
