"""Deutsches Textarchiv, 1600-1699 publisher zip (dtae/dta_1600-1699.zip), without the 7 Sandrart files.

The folder name 'dtae' is misleading: this is DTA 'komplett' 1600-1699 (1,107 distinct TEI-P5 files after removing
the extracted copy, flagged dup, and the Sandrart Teutsche Academie / Iconologia files, which are out of this module).
Editor tagging in running text exists in 17 files only (Dresden Hofdiarium 1657/1665/1674 with Wikidata refs, a few
sermons/treatises with untagged persName/placeName); everything else gets only the header: the author of the printed
source, <titleStmt/author/persName ref=GND> (role 'author'). The same author in sourceDesc/biblFull is a copy and is
dropped. Modern editors, DTA project and publisher are never read (respStmt/editor/publicationStmt).
Refs: GND and Wikidata URLs go to auth (some Wikidata ids denote offices, e.g. Q824760, kept as given); 'dresden 1657'
has a few local labels (kept in ref_raw, unresolved). No register, no coordinates.
Dating: imprint date in sourceDesc (header), i.e. the original print year; the Dresden Hofdiarium files take the year of
their file name (the header gives the year the manuscript copy was made, one year later).
Licence: per file from teiHeader availability/licence (CC BY-SA 4.0, BY-SA 3.0 DE, BY-SA 2.0 DE, CC0, BY-NC 3.0 DE);
LICENSE_TIERS DTA_komplett is T1 (CC-BY-SA-4.0) but the BY-NC files are T2: m.licence / m.tier are set per file and
the module TIER is the strictest (T2).
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'dta_1600'
COLLS = ['dtae/dta_1600-1699.zip']
RENASCOR_IDS = ['dta']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'DTA 1600-1699 (header authors + the few tagged texts, e.g. Dresden Hofdiarium)'
OPTIONS = {}

_LIC = [(re.compile(r'zero|cc0|publicdomain', re.I), 'CC0-1.0', 'T1'),
        (re.compile(r'by-nc-sa/(\d\.\d)(?:/(\w+))?', re.I), None, 'T2'),
        (re.compile(r'by-nc/(\d\.\d)(?:/(\w+))?', re.I), None, 'T2'),
        (re.compile(r'by-sa/(\d\.\d)(?:/(\w+))?', re.I), None, 'T1')]


def licence_of(root) -> tuple[str, str]:
    tgt = txt = ''
    for el in root.iter():
        if C.ln(el.tag) == 'licence':
            tgt = el.get('target') or ''
            txt = C.norm(''.join(el.itertext()))
            break
    s = tgt or txt
    for rx, fixed, tier in _LIC:
        m = rx.search(s)
        if m:
            if fixed:
                return fixed, tier
            fam = 'CC-BY-NC-SA' if 'by-nc-sa' in s.lower() else 'CC-BY-NC' if 'by-nc' in s.lower() else 'CC-BY-SA'
            ver = m.group(1)
            ctry = (m.group(2) or '').upper()
            ctry = ctry if len(ctry) == 2 else ''
            return f'{fam}-{ver}' + (f'-{ctry}' if ctry else ''), tier
    return LICENCE, TIER


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups)
                if 'sandrart' not in r.name.lower() and r.name.lower().endswith('.xml')]

    def is_register_file(self, rec, ctx):
        return False

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        main = {(h.refs.get('ref') or h.surface) for h in hits if 'sourceDesc' not in h.ancestors}
        return [h for h in hits if 'sourceDesc' not in h.ancestors or (h.refs.get('ref') or h.surface) not in main]

    def doc_date(self, root, rec, ctx):
        m = re.search(r'hofdiarium_(\d{4})', rec.name)       # the diary year, not the date of the manuscript copy
        if m:
            y = int(m.group(1))
            return C.DocDate(y, y, 'filename')
        return super().doc_date(root, rec, ctx)

    def adjust(self, m, hit):
        if m.zone == 'hdr':
            m.context = None
        return m

    def extract_root(self, root, rec, ctx):
        ms, es = super().extract_root(root, rec, ctx)
        lic, tier = licence_of(root)
        for m in ms:
            m.licence, m.tier = lic, tier
        return ms, es


EXTRACTOR = Extractor()
