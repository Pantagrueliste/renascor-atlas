"""Correspondence of Christofforo Suriano, Venetian envoy at The Hague, 1616-1623 (Huygens Institute / Nina Lamal).

Licence: CC-BY-4.0, T1 (LICENSE_TIERS). Raw: ``suriano/`` (publisher GitHub repository).

What the editors tagged (TEI): nothing in the running text. Only the letter header carries names:
``correspAction[@type=sent]/name[@ref="bio.xml#cs"]`` (the sender, always Suriano), ``correspAction[@type=received]/name``
("Doge and Senate of Venice", an institution: kind ``org``, role ``addressee``), ``settlement`` in each correspAction
(sending place, role ``sent_from``; receiving place ``sent_to``) and the repository town in ``msIdentifier``
(Venice, role ``repository``, nonstandard).  Untyped ``name`` elements are not caught by the generic scanner, so this
module reads the correspAction names itself.

Not read (decisions):
  * ``datasource/transcriptions/teiSimple/*.xml`` (11 pandoc volumes repeating the letter text, no correspDesc);
  * the Text-Fabric NER layer (``tf/1.0.3e``: 787 PER entities, ~11,960 occurrences, produced by a gazetteer match of
    the editors' trigger spreadsheet ``ner/specs/persons*.xlsx``) and ``datasource/entities/entitymeta.json`` (820 person
    biographies): they are not TEI tags and are rule-based annotation, outside the "editor-tagged TEI" scope of v1.
    They can be added as a separate source later (the eid slugs would be the keys);
  * ``bio.xml`` (target of the sender ref) is not in the harvest: ``bio.xml#cs`` is kept as an opaque key
    ``suriano:cs`` (in_register False).
Dating: ``correspAction[@type=sent]/date/@when`` (date_source ``letter``), 725 of 725 letters.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch0 as H
from .generic_tei import GenericTEI

SLUG = 'suriano'
COLLS = ['suriano']
RENASCOR_IDS = ['suriano-correspondence']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Suriano letters 1616-1623: sender, addressee and sending places from correspDesc'
OPTIONS = {'opaque_keys': True}

_ORG = re.compile(r'senate|doge|council|collegio|states|republic|signoria|senato', re.I)


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.path.startswith('suriano/datasource/tei/')]

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        for ca in H.iter_local(root, 'correspAction'):
            t = (ca.get('type') or '').lower()
            role = 'sender' if t.startswith('sen') else ('addressee' if t.startswith(('rec', 'del')) else 'mention')
            for el in ca:
                if C.ln(el.tag) == 'name':
                    s = C.surface_text(el)
                    kind = 'org' if _ORG.search(s) else 'person'
                    hits.append(H.make_hit(el, kind, f'name@{kind}', role, 'hdr',
                                           ancestors=('TEI', 'teiHeader', 'profileDesc', 'correspDesc', 'correspAction')))
        return hits

    def adjust(self, m, hit):
        m.context = None             # header fields: no running text
        if m.tag == 'settlement' and 'msIdentifier' in hit.ancestors:
            m.role = 'repository'
        return m


EXTRACTOR = Extractor()
