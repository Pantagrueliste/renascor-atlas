"""Familiensache? Korrespondenz der Kaiserin Eleonora Magdalena (1655-1720), EMT working data (ACDH-CH / IHB OeAW).

1,605 letters (Transkribus HTR, TEI) in emt-working-data.tar.gz. Running text has <rs type="person|place|org"
ref="#emt_person_id__N"> (also pronouns and titles such as 'Kaiser', 'sohn': the ref makes them mentions of a
person); the header correspDesc gives sender / addressee / place of sending as persName / placeName with the same
id space. The ids are stable inside the edition but their register (kaiserin-eleonora.oeaw.ac.at) is not in the
harvest: opaque keys 'kaiserin_eleonora_magdalena:emt_person_id__N' (in_register False); no authority ids, no coordinates.

One letter (kasten_blau_44_9_0269) carries a small register in <back> (Vienna with coordinates and GeoNames, Eleonora
Magdalena and Johann Wilhelm with GND). The ids are edition-wide, so prepare() loads those entries as a SHARED register
('kaiserin_eleonora_magdalena:emt_person_id__9'): every mention of these ids, in all 1,605 letters, then carries the
GND / GeoNames id (in_register True); the register is not duplicated as a document-local one.

Handling:
  * roles from correspAction (sender / addressee / sent_from / sent_to) via the generic scanner;
  * places inside msIdentifier (country / settlement of the archive, Munich...) get role 'repository';
  * dating: correspAction/date@when-iso; letters of 1701-1716 are kept with in_period False.
Licence: CC-BY-4.0 (imprint of the site; the TEI licence element only says 'copyright'), LICENSE_TIERS T1.

v2: re-harvest of 2026-10-17 (emt-working-data main at fc4898f, 15 Oct 2026; 8 of the 1,605 files corrected since the first
copy; the repository is a work in progress: re-fetch just before the release) and the project's registers
(``emt-entities`` at ab024be, harvested as ``kaiserin_eleonora_magdalena-registers/``: listperson.xml 1,391 persons with
GND / Wikidata / NDB, listplace.xml 471 places with GeoNames and coordinates, listorg.xml 72 institutions; same
``emt_person_id__N`` / ``emt_place_id__N`` ids as the letters). They are loaded into the shared register by ``prepare()``,
so the persons get real names (not 'Mayestät'), authority ids and the places coordinates. The archive's country and city
(msIdentifier) are zone 'repository' and excluded from the default counts.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch1 as H
from .. import helpers_v2 as V

SLUG = 'kaiserin_eleonora_magdalena'
COLLS = V.pick(['renascor_2026-10/kaiserin-eleonora-magdalena-korrespondenz'],
               [V.NEW + '/kaiserin_eleonora_magdalena/files/emt-working-data.tar.gz'])
_REG = V.NEW + '/kaiserin_eleonora_magdalena-registers/emt-entities/indices/'
RENASCOR_IDS = ['kaiserin-eleonora-magdalena-korrespondenz']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Letters of Empress Eleonora Magdalena 1677-1716: sender, addressee, places and tagged mentions'
OPTIONS = {'opaque_keys': True}


class Extractor(H.BaseTEI):
    def prepare(self, ctx):
        ents = V.load_registers(ctx, [_REG + 'listperson.xml', _REG + 'listplace.xml', _REG + 'listorg.xml'])
        for rec in ctx.all_files:
            if not rec.count('reg|person', 'reg|place', 'reg|org'):
                continue
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for ent, aliases in C.parse_register(root, ctx, rec.path, doc_id=None)[0]:
                ctx.shared.add(ent, aliases)
                ents.append(ent)
        return ents

    def extract_root(self, root, rec, ctx):
        # the in-letter register is shared (see prepare): no document-local entities
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []


EXTRACTOR = Extractor()
