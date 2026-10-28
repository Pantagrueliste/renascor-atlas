"""Folger "Rare or Unique Poems from manuscripts, 1500-1660" (RUP), pilot sample of 20 poems (MS V.a.103.1).

Licence: CC-BY-SA-4.0 in the poem headers; LICENSE_TIERS lists RUP_Folger as T1 (LicenseRef-PublicDomain), the
header licence is applied (stricter): CC-BY-SA-4.0, T1.  Raw: ``eng_dh/rup.zip`` (the extracted copy
``eng_dh/Rare-or-Unique-...-main`` is a duplicate and is skipped).

Edition conventions
  * ``persName/@ref="../persons/persons.xml#P0003"`` (+ placeName / orgName without ref) in the poems; the register
    ``personography/persons.xml`` (22 persons: birth, death, ODNB / LOC / find-a-grave links) is loaded by prepare()
    and resolved by xml:id (the path prefix is stripped by common.classify_token);
  * the attributed author of a poem is given in ``titleStmt/respStmt[resp='attributed author']/name`` (5 poems):
    emitted as role ``author`` (the scanner would drop respStmt names as modern editors); the other respStmt
    entries (TEI encoding, AI summary) are modern and ignored;
  * persons in ``<back><div type="historical">`` (the editor's historical note) are kept: they are tagged by the
    editor and identify people the poem is about; zone ``txt``;
  * persons inside ``bibl/editor`` of the back-matter 'Works cited' are modern scholars (J. Barry, P. Elmer): dropped;
  * LOC links ``lccn.loc.gov/n...`` are stored as ``loc:n...`` in auth.other;
  * dates: ``profileDesc/creation/date`` (header); 4 poems have none: default_years 1599-1625 (the range of the
    dated poems of the same manuscript), date_source ``collection``.

v2: re-harvest of 2026-10-17 (``rup.zip`` = git archive of 4b4815c, 9 Jun 2026): one more poem
(``poems/rup_000021_add11811_3v.xml``); its ``persons.xml#P0023`` dangles upstream (persons.xml ends at P0022) and stays
unresolved.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch0 as H
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'rup'
COLLS = V.pick(['eng_dh/rup'], [V.NEW + '/rup/rup.zip'])
RENASCOR_IDS = ['rup-folger']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Folger Rare or Unique Poems pilot: 20 poems with a 22-person personography'
OPTIONS = {'default_years': (1599, 1625)}


class Extractor(GenericTEI):
    def parse_register_file(self, rec, ctx, root):
        ents = super().parse_register_file(rec, ctx, root)
        for ent, _ in ents:
            H.loc_from_url(ent.auth)
            ent.note_type = ent.note_type or ent.kind
        return ents

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        hits += H.author_respstmt_hits(root, ('author',))
        return hits

    def adjust(self, m, hit):
        if m.kind == 'person' and 'bibl' in hit.ancestors and 'editor' in hit.ancestors:
            return None          # editors of cited modern works in the 'Works cited' note (J. Barry, P. Elmer)
        H.loc_from_url(m.auth)
        if m.zone == 'hdr':
            m.context = None
        return m


EXTRACTOR = Extractor()
