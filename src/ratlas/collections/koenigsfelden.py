"""Koenigsfelden: charters and records of the convent / Oberamt (StAAG, Aargau), 1,557 documents 1250-1660.

Licence: CC-BY-4.0, T1 (entry + LICENSE_TIERS; the website is CC BY-NC-SA, images public domain).
Raw: ``renascor_2026-10/koenigsfelden-urkunden-akten/files/koenigsfelden-data-v2024-04-09.zip`` (eXist-db package).

What is special here
  * mentions are ``persName`` / ``placeName`` / ``orgName`` in the diplomatic transcription with ``@ref="per000089"``,
    ``"loc000008"``, ``"org000030"`` (bare ids); the three registers ``data/registers/{people,places,organizations}.xml``
    are stand-alone register files (generic detection) loaded once by prepare(); keys ``koenigsfelden:per000089``;
  * the registers hold dates (``birth`` / ``death`` as ``dd.mm.yyyy`` strings, ``event Ersterwähnung`` ignored),
    places hold ``geo`` + country / region + HLS / GND / ortsnamen.ch ids (ortsnamen.ch ids go to auth.other as
    ``ortsnamen.ch:<id>``); organisations are families, convents, dioceses: ``note_type`` is the German ``desc``
    (``familie``, ``zisterzienserkonvent``...), for places the German place type (``stadt``, ``gemeinde``, ``bistum``...);
  * a handful of refs end in a literal ``\\u0009`` (export artefact): stripped before resolution;
  * unresolved refs are ids that the register's own comments list as missing (API timeouts: per026256... loc00019...);
  * ``respStmt`` persons (transcribers, modern) are skipped by the scanner; ``origPlace`` is empty in 1,480 of 1,481
    documents and is not read;
  * dates: ``msDesc/history/origin/origDate`` (header) for every document; 458 of 1,557 documents fall in 1450-1700
    (the in_period flag is kept per mention: the atlas keeps the earlier charters, flagged False).
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'koenigsfelden'
COLLS = ['renascor_2026-10/koenigsfelden-urkunden-akten']
RENASCOR_IDS = ['koenigsfelden-urkunden-akten']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Koenigsfelden charters 1250-1660 with person/place/org registers (HLS, GND, coordinates)'
OPTIONS = {}

_ESC = re.compile(r'\\u0009|\t')


class Extractor(GenericTEI):
    def parse_register_file(self, rec, ctx, root):
        ents = super().parse_register_file(rec, ctx, root)
        els = {e.get(C.XML_ID): e for e in root.iter() if C.ln(e.tag) in ('person', 'place', 'org') and e.get(C.XML_ID)}
        for ent, _ in ents:
            el = els.get(ent.local_key.split(':', 1)[1])
            if el is None:
                continue
            if ent.kind == 'place':
                notes = [C.norm(''.join(n.itertext())).lower() for n in el.iter() if C.ln(n.tag) == 'note' and n.get('type') == 'type']
                ent.note_type = notes[0] if notes else 'place'
            elif ent.kind == 'org':
                descs = [C.norm(''.join(d.itertext())).lower() for d in el if C.ln(d.tag) == 'desc' and d.get('{http://www.w3.org/XML/1998/namespace}lang') == 'de']
                ent.note_type = descs[0] if descs else 'org'
            else:
                ent.note_type = 'person'
            for idn in el.iter():
                if C.ln(idn.tag) == 'idno' and (idn.get('type') or '').lower() == 'ortsnamen.ch':
                    x = 'ortsnamen.ch:' + C.norm(''.join(idn.itertext()))
                    if x not in ent.auth['other']:
                        ent.auth['other'].append(x)
        return ents

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        for h in hits:
            for a, v in list(h.refs.items()):
                v2 = _ESC.sub('', v).strip()
                if v2 != v:
                    h.refs[a] = v2
        return hits


EXTRACTOR = Extractor()
