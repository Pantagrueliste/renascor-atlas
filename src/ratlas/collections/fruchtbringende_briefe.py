"""Fruchtbringende Gesellschaft, Briefe 1617-1626 (vol. 1 of the critical edition, HAB Wolfenbuettel).

91 letter files (briefe/YYMMDD.xml) with <rs type="person|place" ref="#slug"> in headings, regests, transcriptions and
commentary. The slugs are defined in register-person/place/bibl.xml which are external DOCTYPE entities and NOT in the
harvest, so no entity can be built: the slugs are stable, readable ids of the edition and are kept as opaque keys
(local_key '<slug>:<id>', in_register False).
Special handling:
  * header <author> holds the modern editors (Conermann, Birchner), not the letter's sender: dropped;
  * sender / addressee: the letter heading is "<rs>Sender</rs> an <rs>Addressee</rs>" (no correspDesc), roles are set
    from the first such heading;
  * dating: the file name is YYMMDD (171224 -> 1617), date_source 'filename'; the headers carry no origin date.
v2: register-person.xml and register-place.xml (``ed000213/register/``, harvested in ``fruchtbringende_briefe-registers/``)
are loaded by ``prepare()``: 100% of the mentions resolve; persons carry 715 GND keys ('GND_135668344'), places 276 TGN
ids and 262 coordinates; entries noted as Greek mythology get note_type mythological.
Licence: LICENSE_TIERS CC-BY-SA-3.0-DE (T1, from the METS rights statement); the TEI headers only say
"availability restricted ... (c) HAB". Tier T1 kept, conflict reported.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_v2 as V
from ..helpers_batch3 import assign_heading_roles, yymmdd_year
from .generic_tei import GenericTEI

SLUG = 'fruchtbringende_briefe'
COLLS = ['originals_2026-10-15/oct-binary/oct__fruchtbringende-gesellschaft-briefe-1617-1626']
RENASCOR_IDS = ['fruchtbringende-gesellschaft-briefe-1617-1626']
LICENCE = 'CC-BY-SA-3.0-DE'
TIER = 'T1'
DESCRIPTION = 'Letters of the Fruchtbringende Gesellschaft 1617-1626, rs person/place with unresolved local slugs'
OPTIONS = {'opaque_keys': True, 'default_years': (1617, 1626)}


_REG = V.NEW + '/fruchtbringende_briefe-registers/ed000213/register/'


class Extractor(GenericTEI):
    def prepare(self, ctx):
        return V.load_registers(ctx, [_REG + 'register-person.xml', _REG + 'register-place.xml'], fix=V.hab_person_fix)

    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if '/briefe/' in r.path]

    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        y = yymmdd_year(rec.name)
        if y and 1617 <= y <= 1626:
            return C.DocDate(y, y, 'filename')
        return super().doc_date(root, rec, ctx)

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        hits = [h for h in hits if not (h.zone == 'hdr' and h.role == 'author')]   # modern editors
        assign_heading_roles(hits)
        return hits

    def adjust(self, m, hit):
        # the root xml:id is shared by many letters of a year: the file stem identifies the letter
        m.doc_id = m.source_path.rsplit('/', 1)[-1][:-4]
        return m


EXTRACTOR = Extractor()
