"""Netzwerke der Nonnen: the letters of Kloster Luene (HAB Wolfenbuettel, ed000248), 413 letters, Low German and Latin.

Licence: CC-BY-SA-3.0 (Germany), tier T1 (LICENSE_TIERS: Renascor_luene-nonnenbriefe; stated in each header).

Mentions: <rs type=person|org|place ref="#slug"> in correspDesc (roles sender / addressee, sent_from / sent_to) and in
the text (original-spelling reading; <reg> is dropped). 470 person and 170 place tags have no ref and stay without key.
The register (names, GND, GeoNames) was not harvested (mets.xml only lists the letters), so '#slug' ids are opaque
keys; person, place and organisation ids can coincide (#lueneburg is the town council AND the town), so the key
carries the kind: 'luene_nonnenbriefe:person:buelow_busse', ':place:bleckede', ':org:luene_kloster' (in_register
False). No authority ids, no coordinates, no entities.
v2: the person register (``ed000248/texts/personenregister.xml``, 599 persons, harvested in ``luene_nonnenbriefe-registers/``) is
loaded by ``prepare()`` under the kind-carrying key ('luene_nonnenbriefe:person:<id>'): 49% of the mentions resolve (persons
only; the place and organisation lists return 404, so places and orgs stay opaque keys). Most register entries are known by a
role or an initial only ('Subpriorin', 'A'), no GND. The placeholder 'unbekanntes Kloster' is excluded as a placeholder.
Dating: correspAction/date (when-custom / text year); else a year 1400-1560 in the letter title; else the whole
collection span 1460-1555 set by the project (date_source 'collection', 212 letters, 'undatiert' in the header).
The mets.xml index is not a text and is skipped; the repository settlement of msIdentifier and empty placeholder
<rs/> are not mentions.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch5 as H
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'luene_nonnenbriefe'
COLLS = ['renascor_2026-10/luene-nonnenbriefe']
RENASCOR_IDS = ['luene-nonnenbriefe']
LICENCE = 'CC-BY-SA-3.0'
TIER = 'T1'
DESCRIPTION = 'Letters of the Luene convent c. 1460-1555; #slug ids without register'
OPTIONS = {'opaque_keys': True, 'default_years': (1460, 1555)}

_TITLE_YEAR = re.compile(r'(?<!\d)(1[34]\d\d|15[0-5]\d)(?!\d)')


def _title_year(root, rec):
    for el in root.iter():
        if C.ln(el.tag) == 'title':
            m = _TITLE_YEAR.search(C.surface_text(el))
            if m:
                return C.DocDate(int(m.group(1)), int(m.group(1)), 'header')
            break
    return None


_REG = V.NEW + '/luene_nonnenbriefe-registers/ed000248/texts/'


def _kind_key(ent, aliases, els):
    ident = ent.local_key.split(':', 1)[1]
    ent.local_key = f'{SLUG}:{ent.kind}:{ident}'
    ent.note_type = ent.note_type or ent.kind
    return ent, aliases


class Extractor(GenericTEI):
    def prepare(self, ctx):
        return V.load_registers(ctx, [_REG + 'personenregister.xml'], fix=_kind_key)

    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if not r.name.lower().startswith('mets')]

    def doc_date(self, root, rec, ctx):
        return C.resolve_doc_date(root, rec, default_years=ctx.default_years, fallback=_title_year)

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root), counters=counters, skip_ancestors=('msIdentifier',))

    def adjust(self, m, hit):
        if not m.surface and not m.ref_raw:
            return None          # empty <rs type="place"/> placeholder of an unknown sender place
        if m.local_key and not m.in_register:
            ident = m.local_key.split(':', 1)[1]
            m.local_key = f'{SLUG}:{m.kind}:{ident}'
        return H.drop_header_context(m)


EXTRACTOR = Extractor()
