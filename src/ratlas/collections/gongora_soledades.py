"""Gongora, Soledades: digital annotated edition by Antonio Rojas Castro (GitHub arojascastro/soledades, 2017).

Three versions exist (annotated, simple, critical). Only the ANNOTATED one (edicion/anotada/input/soledades_anotada.xml)
is read: the simple version repeats the same tags (counting both would double every mention) and the critical edition
has none.

Licence: the annotated file is CC-BY-4.0; the critical edition is CC-BY-NC-3.0 and is not read. The module carries the
collection's LICENSE_TIERS value ('CC-BY-4.0 AND CC-BY-NC-3.0', T2) so that the tier is the same as the archive's.

Register: the file holds its own listPerson (84 entries, '@xml:id' act001, apo009..., each with a <note type="bio">
gloss) and listPlace (31, geogName / placeName). Mentions are <persName|placeName|geogName ref="#id"> in the verses
(308, 99.7 % resolved). Persons are overwhelmingly mythological; note_type is derived from the edition's own bio gloss:
'mythological' when the gloss speaks of mythology, gods, nymphs, heroes..., 'historical' otherwise (checked by hand in
the report); places get 'place'. No authority ids or coordinates exist in the file.
Keys are shared ('gongora_soledades:act001'). Dating: composition 1612-1613 (collection span, date_source 'collection';
the dates in the header are printed witnesses of 1627-1654).
"""
from __future__ import annotations

import re
from collections import Counter

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'gongora_soledades'
COLLS = ['originals_2026-10-15/github-lowresource-tei/GitHubTEI/arojascastro__soledades/edicion/anotada/']
RENASCOR_IDS = ['gongora-soledades']
LICENCE = 'CC-BY-4.0 AND CC-BY-NC-3.0'      # LICENSE_TIERS (GitHubTEI); the file read is the CC-BY-4.0 annotated one
TIER = 'T2'
DESCRIPTION = 'Gongora Soledades annotated edition: in-file onomastic and toponym registers'
OPTIONS = {'default_years': (1612, 1613)}

_MYTH = re.compile(r'mitolog|\bdios(a|es)?\b|ninfa|centauro|tit[aá]n|cíclope|ciclope|h[eé]roe|musa|oráculo|'
                   r'personific|divinidad|deidad|fabulos|\bEneida\b|Eneas|castigo|dios marino|\bZeus\b|\bJúpiter\b|semidi|náyade|'
                   r'nereida|sátiro|furia|parca|hija del anciano pescador', re.I)


class Extractor(GenericTEI):
    def scan(self, root, rec, ctx, counters):
        # the witnesses' repositories (country / settlement of msIdentifier) are archives, not mentions
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root), counters=counters,
                               skip_ancestors=('msIdentifier', 'listWit'))

    def doc_date(self, root, rec, ctx):
        y0, y1 = ctx.default_years
        return C.DocDate(y0, y1, 'collection')

    def extract_root(self, root, rec, ctx):
        local = C.Register()
        ents = []
        notes = {}
        for el in root.iter():
            if C.ln(el.tag) == 'person' and el.get(C.XML_ID):
                nt = next((c for c in el if C.ln(c.tag) == 'note'), None)
                notes[el.get(C.XML_ID)] = C.surface_text(nt) if nt is not None else ''
        for ent, aliases in C.parse_register(root, ctx, rec.path, doc_id=None)[0]:
            if ent.kind == 'person':
                bio = notes.get(ent.local_key.split(':', 1)[1], '')
                ent.note_type = 'mythological' if _MYTH.search(bio) else 'historical'
            else:
                ent.note_type = 'place'
            local.add(ent, aliases)
            ents.append(ent)
        hits = self.scan(root, rec, ctx, Counter())
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=local, date=date, hits=hits, adjust=self.adjust), ents


EXTRACTOR = Extractor()
