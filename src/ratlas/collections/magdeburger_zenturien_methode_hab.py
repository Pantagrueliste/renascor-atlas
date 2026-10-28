"""Historische Methode und Arbeitstechnik der Magdeburger Zenturien (HAB Wolfenbuettel, ed000086): 153 letters and
documents of Flacius, Wigand, Judex, Nidbruck and others (files Ort/Archiv/YYMMDD_*.xml; mets.xml skipped).

Licence: CC-BY-SA-3.0-DE, tier T1 (LICENSE_TIERS: Renascor_magdeburger-zenturien-methode-hab; the headers say
'restricted, copyright HAB': the module keeps the catalogue licence, as the tier table does).

Mentions: <rs type=person ref="#slug"> in the text and in the header title of the manuscript description
(msDesc/head/title 'Matthias Flacius (Koeln) an Caspar von Nidbruck'), <rs type=author ref="#slug"> for cited authors
(tag 'rs@author': Ambrosius, Hinkmar...). No place markup in the files. The person register (register/register-person.xml)
is not in the harvest: '#slug' ids are opaque keys ('magdeburger_zenturien_methode_hab:caspar_nidbruck', in_register
False), no entities, no authority ids.
Roles: the files have no correspDesc; the sender and addressee are the rs persons of the msDesc title before / after
the word 'an' (role sender / addressee; other titles keep 'mention').
v2: register-person.xml (``ed000086/register/``, harvested in ``magdeburger_zenturien_methode_hab-registers/``, CC BY-SA 3.0 DE)
is loaded by ``prepare()``: 89.7% of the mentions resolve, 315 GND ids ('pnd_118521330' keys). The sender / addressee test
reads ' an ' with any whitespace (12 titles have a line break after 'an'). Three documents of the METS list return 404
(two are commented out in the METS).
Dating: msDesc/head/origDate (when), else the YYMMDD prefix of the file name read as 15YY, else the span 1552-1568.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch5 as H
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'magdeburger_zenturien_methode_hab'
COLLS = ['renascor_2026-10/magdeburger-zenturien-methode-hab']
RENASCOR_IDS = ['magdeburger-zenturien-methode-hab']
LICENCE = 'CC-BY-SA-3.0-DE'
TIER = 'T1'
DESCRIPTION = 'Magdeburg Centuriators letters 1552-1568; #slug person refs, no register'
OPTIONS = {'opaque_keys': True, 'default_years': (1552, 1568), 'type_map': {'author': 'person'}}

_PREFIX = re.compile(r'^(\d\d)(\d\d)(\d\d)')


def _name_date(root, rec):
    m = _PREFIX.match(rec.name)
    if m and 1 <= int(m.group(2)) <= 12:
        y = 1500 + int(m.group(1))
        return C.DocDate(y, y, 'filename')
    return None


def _title_text_offsets(title):
    """(full text, {id(el): start offset}) of an element's text; used to locate rs elements before / after ' an '."""
    parts, offs, pos = [], {}, 0

    def rec(e):
        nonlocal pos
        if e.text:
            parts.append(e.text); pos += len(e.text)
        for c in e:
            if isinstance(c.tag, str):
                offs[id(c)] = pos
                rec(c)
            if c.tail:
                parts.append(c.tail); pos += len(c.tail)

    rec(title)
    return ''.join(parts), offs


_REG = V.NEW + '/magdeburger_zenturien_methode_hab-registers/ed000086/register/'
_AN = re.compile(r'\san\s')


class Extractor(GenericTEI):
    def prepare(self, ctx):
        return V.load_registers(ctx, [_REG + 'register-person.xml'], fix=V.hab_person_fix)

    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if not r.name.lower().startswith('mets')]

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root), counters=counters, skip_ancestors=('msIdentifier',))

    def doc_date(self, root, rec, ctx):
        return C.resolve_doc_date(root, rec, default_years=ctx.default_years, fallback=_name_date)

    def adjust(self, m, hit):
        if m.local_key and "'" in m.local_key:       # typo in the source: ref="'johann_friedrich_sachsen_ii"
            m.local_key = m.local_key.replace("'", '')
        if m.zone == 'hdr' and m.kind == 'person' and 'head' in hit.ancestors and 'msDesc' in hit.ancestors:
            title = hit.el
            while title is not None and C.ln(title.tag) != 'title':
                title = title.getparent()
            if title is not None:
                text, offs = _title_text_offsets(title)
                mo = _AN.search(text)
                if mo and id(hit.el) in offs:
                    m.role = 'sender' if offs[id(hit.el)] < mo.start() else 'addressee'
        return H.drop_header_context(m)


EXTRACTOR = Extractor()
