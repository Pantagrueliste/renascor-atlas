"""Filippo Cavriana's correspondence (1566-1574), digital edition by C. Godbarge (Zenodo 10.5281/zenodo.14789457).

Own project of the catalogue owner, CC-BY-4.0 in every file header and in LICENSE_TIERS (T1).
Raw data: ``cavriana/`` (115 letters ``letters/YYYY-MM-DD.xml`` + two shared authority files).

What is special here
  * every ``persName`` / ``placeName`` carries ``ref="#pers-..."`` / ``ref="#place-..."`` pointing to the two
    authority files ``letters/persNames.xml`` (VIAF) and ``letters/placeNames.xml`` (TGN + ``<geo>``); the generic
    register detection loads them once in ``prepare()``;
  * place entries hold several names (historical Italian, modern, country, region): the entity name is the modern
    name, the historical spellings are variants (the surface forms found in the letters);
  * correspDesc gives sender / addressee (``sender``, ``addressee``) and the sending place (``sent_from``); the
    sending place of some letters is plain text without a ref (kept, unresolved);
  * 55 of the 115 letters are metadata-only placeholders: they only have the correspDesc mentions;
  * ``<msIdentifier><placeName>`` (the repository town, e.g. Florence) is the place of the archive, not a mention
    of the text: role ``repository`` (nonstandard role);
  * dates: ``correspAction[@type=sent]/date`` (date_source ``letter``).

v2 (re-harvest of 2026-10-17, CavrianaCorr main at 581aab1, TGN ids re-derived upstream; legacy path kept):
  * 118 files in ``letters/``: 113 letters + persNames/placeNames + the new groupNames/eventNames/medicalterms. Only the
    letters and the three person/place/group authority files are read; the event and medical-term lists are not
    entity registers;
  * ``groupNames.xml`` (personGrp: peoples, confessions, soldiery) is read as ORG entries (note_type 'group',
    person_class 'people'); ``<rs type="group|ethnic">`` mentions are org mentions of class 'people' (they are
    collectives, not persons); an ethnic rs that points to a place entry keeps its own kind and inherits no place ids;
  * the repository city of ``msIdentifier`` is role/zone ``repository`` (excluded from the default counts).
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch0 as H
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'cavriana'
COLLS = V.pick(['cavriana'], [V.NEW + '/cavriana/letters/'])
RENASCOR_IDS = ['cavriana-correspondence']
LICENCE = 'CC-BY-4.0'        # file headers + LICENSE_TIERS (T1)
TIER = 'T1'
DESCRIPTION = 'Filippo Cavriana letters 1566-1574 with VIAF person and TGN place authority files'
OPTIONS = {'type_map': {'ethnic': 'org', 'group': 'org'}}


# typos of the edition: ids used in letters that are not in the authority files
REF_FIXES = {'pers-concino-b': 'pers-concini-b', 'place-florence': 'place-firenze'}


_REGISTERS = ('persNames.xml', 'placeNames.xml', 'groupNames.xml')
_LETTER = re.compile(r'^\d{4}-\d{2}-\d{2}')


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        # letters (YYYY-MM-DD*.xml) and the authority files; eventNames/medicalterms are lists of vocabulary, not documents
        return [r for r in super().iter_files(ctx, include_dups)
                if r.name in _REGISTERS or _LETTER.match(r.name)]

    def is_register_file(self, rec, ctx):
        return rec.name in _REGISTERS

    def parse_register_file(self, rec, ctx, root):
        ents = super().parse_register_file(rec, ctx, root)
        if rec.name == 'groupNames.xml':
            for ent, _aliases in ents:           # personGrp: peoples, confessions, soldiery
                ent.kind, ent.note_type, ent.person_class = 'org', 'group', 'people'
            return ents
        els = {e.get(C.XML_ID): e for e in root.iter('{*}place') if e.get(C.XML_ID)}
        for ent, _aliases in ents:
            if ent.kind != 'place':
                continue
            el = els.get(ent.local_key.split(':', 1)[1])
            if el is None:
                continue
            names = [(p.get('type'), C.surface_text(p)) for p in el if C.ln(p.tag) == 'placeName']
            modern = [n for t, n in names if t == 'modern']
            hist = [n for t, n in names if t in ('historical', None)]
            if modern:
                ent.name = modern[0]
                ent.variants = list(dict.fromkeys(v for v in modern[1:] + hist if v and v != ent.name))
            ent.note_type = 'place'
        for ent, _aliases in ents:
            ent.note_type = ent.note_type or ent.kind
        return ents

    def extract(self, rec, ctx):
        self._ctx = ctx          # set before use in forked workers (adjust has no ctx argument)
        return super().extract(rec, ctx)

    def adjust(self, m, hit):
        if m.kind == 'org' and hit.name == 'rs':
            m.person_class = 'people'            # rs type=group|ethnic: a collective, not an institution
        if m.zone == 'hdr':
            m.context = None     # header fields are not running text: a snippet would show neighbouring fields
            if 'msIdentifier' in hit.ancestors:
                m.role = 'repository'
        if m.local_key is None:
            ctx = self._ctx
            ent = None
            for tok in (m.ref_raw or '').split():
                ent = ctx.shared.lookup(REF_FIXES.get(tok.lstrip('#'), ''))
                if ent:
                    break
            if ent is None and m.kind == 'place' and m.role in ('sent_from', 'sent_to', 'repository'):
                # sending place given as text only (Paris, Parigi, Firenze...): exact name of an authority entry
                ent = H.unique_by_name(ctx.shared, m.surface, 'place')
            if ent is not None:
                H.attach(m, ent)
        return m


EXTRACTOR = Extractor()
