"""Ethica Complementoria digital edition (17th-c. German etiquette prints; Rockenberger / HAB; queue5/ethica.zip).

Only the transcriptions (folder transcriptions/: 13 Ethica files 1643-1665, Cochleatio Novissima, occasional poetry)
are read; the modern scholarly article and bibliography (Aufsatz_*, 2016-17) are not sources and are excluded, as are
duplicates. Tagging is almost absent: 23 name tags in the text (typed person / place, some with gnd urls or
'pnd121716198' keys), persName in the DTA-format headers (author Greflinger, with placeholder gnd urls) and a tiny
listPerson / listPlace in 'Klag und Trost' (file-local register, ids #SusannaEngels, #O0002).
Handling: file-local registers are resolved ('ethica_complementoria:<doc>#<id>'); 'pnd<digits>' keys (old GND name)
become auth.gnd; placeholder ids (gnd/XXX, pnd:0000) are removed; untyped <name> is skipped (kind unknown).
Dating: the print year in the file or folder name (1643_A1/, ..._1648_...; the header dates of the register events
would give e.g. 1622 for a 1652 print), else the header, else the catalogue period 1643-1665
(date_source 'collection').
v2 (issue 210): the folders hold several versions of the same transcription (Klag und Trost_1652 and its '+mxDesc' copy; the two Drebber
epicedia; the Bamberg DTA-format file and its 'ß zu sz' variant): their mentions were counted twice. ``prepare()`` reads the tagged
mentions (kind, tag, surface) of every file; a file whose set is contained in the set of another file of the same folder (the larger,
or the first in path order when equal, is kept) is a variant and returns no rows. 'N. N.' and empty names are placeholders (excluded
from the default counts) and take no authority id.
Licence: LICENSE_TIERS gives T1 'LicenseRef-CC-BY-unversioned' (repo LICENSE.txt: CC BY). The DTA-format headers state
CC BY-NC 3.0 DE; files with that licence element are tagged per row 'CC-BY-NC-3.0-DE' / T2, the others T1. Module
constants carry the strictest (CC-BY-NC-3.0-DE, T2).
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch1 as H

SLUG = 'ethica_complementoria'
COLLS = ['queue5/ethica.zip!EthicaComplementoria-master/transcriptions/']
RENASCOR_IDS = ['ethicacomplementoria']
LICENCE = 'CC-BY-NC-3.0-DE'
TIER = 'T2'
DESCRIPTION = 'Ethica Complementoria transcriptions: very sparse editor tagging of persons and places'
OPTIONS = {'default_years': (1643, 1665)}

_PND = re.compile(r'^pnd[:_ ]?(\d{7,10}[\dXx])$', re.I)


class Extractor(H.BaseTEI):
    def prepare(self, ctx):
        sigs = {}
        for rec in sorted(ctx.all_files, key=lambda r: r.path):
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            hits = C.scan_mentions(root, type_map=ctx.type_map)
            sigs[rec.path] = (rec.path.rsplit('/', 1)[0], frozenset((h.kind, h.tag, h.surface) for h in hits if h.surface))
        variants = set()
        for p, (folder, s) in sigs.items():
            if not s:
                continue
            for q, (f2, s2) in sigs.items():
                if q != p and f2 == folder and (s < s2 or (s == s2 and q < p)):
                    variants.add(p)
                    break
        ctx.options['_variant_files'] = variants
        return super().prepare(ctx)

    def extract(self, rec, ctx):
        if rec.path in ctx.options.get('_variant_files', ()):
            return [], []
        return super().extract(rec, ctx)

    def doc_date(self, root, rec, ctx):
        y = C.year_from_filename(rec.member or rec.path, 1600, 1700)    # print year in the file / folder name
        if y:
            return C.DocDate(y, y, 'filename')
        d = C.resolve_doc_date(root, rec)
        if d.known:
            return d
        return C.DocDate(ctx.default_years[0], ctx.default_years[1], 'collection')

    def extract_root(self, root, rec, ctx):
        ms, es = super().extract_root(root, rec, ctx)
        nc = any(C.ln(e.tag) == 'licence' and 'by-nc' in (e.get('target') or '').lower() for e in root.iter())
        lic, tier = ('CC-BY-NC-3.0-DE', 'T2') if nc else ('LicenseRef-CC-BY-unversioned', 'T1')
        for m in ms:
            m.licence, m.tier = lic, tier
        for e in es:
            e.licence, e.tier = lic, tier
        return ms, es

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is None:
            return None
        for tok in (m.ref_raw or '').split():
            mt = _PND.match(tok)
            if mt and not re.fullmatch(r'0+', mt.group(1)):
                m.auth['gnd'] = m.auth.get('gnd') or mt.group(1).upper()
        m.auth['other'] = [x for x in m.auth['other'] if 'XXX' not in x and not re.search(r'[:/]0{4,}$', x)]
        return m


EXTRACTOR = Extractor()
