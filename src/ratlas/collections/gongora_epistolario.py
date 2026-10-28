"""Gongora, Epistolario (OBVIL, ed. Antonio Carreira): the only file of the Gongora project with entity markup.

<persName type="destinatary|polemist|authority|author" key="name string" [ref="viaf url"]> and <placeName key="Madrid">.
There is no register and the keys are name strings, not ids (variants of the same person exist), so nothing is
resolved to a local_key; the key is kept in ref_raw. A VIAF url on a tag (Gongora, 3 tags) goes to auth.viaf.
The other four files (Obra poetica: verse without entity tags; the three plays: cast lists with xml:id roles and
person/@corresp stage configurations, i.e. fictional characters, no persName) are not read.
Roles: type 'destinatary' (letter heading) -> addressee, type 'author' -> author; a placeName followed by a
<date> in the same paragraph (the closing "Madrid, 4 de julio de 1617") -> sent_from; rest 'mention'.
Dating: per letter (div): years of the <date> elements of the letter (date_source 'letter'); letters without a dated
closing fall back to the header creation range 1580-1625 ('header').
Licence: CC-BY-NC-ND-3.0-FR (OBVIL; header and LICENSE_TIERS), T2 (ND: extracted for internal work, flagged).
"""
from __future__ import annotations

import re

from .. import common as C
from ..helpers_batch3 import enclosing, set_years
from .generic_tei import GenericTEI

SLUG = 'gongora_epistolario'
COLLS = ['gongoraobra/gongora_epistolario.xml']
RENASCOR_IDS = ['gongoraobra-obvil']
LICENCE = 'CC-BY-NC-ND-3.0-FR'
TIER = 'T2'
DESCRIPTION = "Gongora's letters: persName/placeName with name-string keys, no register"
OPTIONS = {}

_SEP = re.compile(r'^[\s,.;y]{0,5}$')


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        for h in hits:
            el = h.el
            t = (el.get('type') or '').lower()
            if h.kind == 'person' and h.name == 'persName':
                if t == 'destinatary':
                    h.role = 'addressee'
                elif t == 'author':
                    h.role = 'author'
            elif h.kind == 'place' and h.name == 'placeName':
                nxt = el.getnext()
                if nxt is not None and C.ln(nxt.tag) == 'date' and _SEP.match(el.tail or ''):
                    h.role = 'sent_from'
        return hits

    def adjust(self, m, hit):
        div = enclosing(hit.el, 'div')
        if div is not None:
            rgs = [r for x in div.iter() if C.ln(x.tag) == 'date' and (r := C.date_range(x, 1500, 1700))]
            if rgs:
                set_years(m, min(r[0] for r in rgs), max(r[1] for r in rgs), 'letter')
        return m


EXTRACTOR = Extractor()
