"""FreEM max OA, sub-corpora other than BVH (Descartes, IG, Conde, Sermo, Mercure galant, dramacode, ...).

BVH has its own module (``freem_bvh``); Theatre classique, Mercure galant, Antonomaz, Fabula, FRANTEXT, Wikisource,
gallicaEpub, Scrapping carry no entity markup at all. Licence: LICENSE_TIERS FreEM ``LicenseRef-Mixed``, T2 (per-file
CC licences; 44 files 'All rights reserved' are listed in ``FreEM_all_rights_reserved_excluded.tsv`` and are excluded
here); the per-file licence is read from the harmonised ``1_header`` copy of each file
(``creativecommons.org/licenses/by-nc-nd/3.0/fr`` -> CC-BY-NC-ND-3.0-FR ...) and set on every row; ND files keep no
context snippet. Raw: ``originals_2026-10-15/github-neolatin-french-tei/FreEMmax_OA/0_source/<corpus>/``.

Files read: every non-BVH ``0_source`` file that has at least one name tag in the scan (about 90 of 1,061: Descartes
(clerselier 1-3), IG, Conde, Sermo, Mercure, dramacode, APW, CEPM, Fabula). What is actually usable:
  * Descartes ``clerselier_1/2/3`` (Lettres de Mr Descartes, 1666/1667 edition): 364 untyped ``<name>`` inside the
    letter ``<salute>`` headings ("AU R. P. MERSENNE.", "A MONSIEUR DE BEAUNE."): the addressees. Emitted as person,
    role ``addressee``, tag ``name@addressee`` for headings that start with "A / AU / AUX" and name someone (headings
    "REPONSE DE MR DESCARTES" and bare asterisks are not addressees and are skipped). No refs;
  * IG ``francois_a_londres`` (Boissy, 1758): persName with local ``#Eliante`` refs to nothing in the file (characters),
    placeName/``rs type=city`` (Londres, Paris), ``rs type=institution|character``; out of period (1758);
  * Conde: author and printer persNames in the header (``Basnage, Henri (1615-1695)``): role author; 5 in the text;
  * Sermo / Mercure / dramacode / APW / CEPM / Fabula: only the modern editors in respStmt -> no mentions.
Dating: the harmonised ``1_header`` file gives a ``date/@when`` (Descartes 1666, IG 1758, Conde 1678...); else the own
header; date_source ``header``.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from .. import common as C
from .. import helpers_batch0 as H
from .generic_tei import GenericTEI

SLUG = 'freem_other'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/FreEMmax_OA/0_source']
RENASCOR_IDS = ['freem']
LICENCE = 'LicenseRef-Mixed'
TIER = 'T2'
DESCRIPTION = 'FreEM max OA non-BVH sub-corpora: letter addressees (Descartes), header authors, one 18th-c. play'
OPTIONS = {'type_map': {'institution': 'org', 'character': 'person', 'city': 'place'}}

_BASE = 'originals_2026-10-15/github-neolatin-french-tei/'
_ARR_TSV = _BASE + 'FreEM_all_rights_reserved_excluded.tsv'
_TAGKEYS = ('persName', 'placeName', 'orgName', 'geogName', 'settlement', 'rs', 'name')
_LIC = re.compile(r'creativecommons\.org/licenses/(by(?:-[a-z]+)*)/(\d\.\d)(?:/([a-z]{2}))?', re.I)
_HEADING = re.compile(r'^\s*(A|AU|AUX|À)\s+\S', re.I)


def _spdx(target: str):
    m = _LIC.search(target or '')
    if not m:
        return None
    return 'CC-' + m.group(1).upper() + '-' + m.group(2) + (('-' + m.group(3).upper()) if m.group(3) else '')


def _header_path(path: str) -> str:
    p = PurePosixPath(path)
    parts = list(p.parts)
    i = parts.index('0_source')
    parts[i] = '1_header'
    return str(PurePosixPath(*parts).with_name(p.stem + '_dAlemBERT.xml'))


_SALUT_PREFIX = re.compile(r'^(?:A|AU|AUX|À)\s+', re.I)
_HONORIFIC = re.compile(r'^(?:MONSIEUR|MONSEIGNEUR|MADAME|MADEMOISELLE|MESSIEURS|MR|M\.|REVEREND|R\.|PERE|P\.|MADAM)\b[\s.,]*', re.I)
_CAPRUN = re.compile(r"[A-ZÀ-ÝŒ*.'’\-\s,]+")


def clean_salutation(s: str) -> str:
    """'A MONSIEUR CHANUTBrasset destinataire selon AT V 349...' -> 'Chanut': the leading 'A / AU / AUX', the honorific and the editorial
    note glued to the name (it starts with a capital + lower-case letter, or at ', destinataire') are cut (audit issue 71)."""
    s = _SALUT_PREFIX.sub('', s.strip())
    s = re.split(r',?\s+destinataire\b', s, maxsplit=1, flags=re.I)[0]
    m = _CAPRUN.match(s)
    if m:
        run = m.group(0)
        if m.end() < len(s) and s[m.end()].islower() and len(run) > 1:
            run = run[:-1]                       # the capital of the editor's note glued to the name
        s = run
    for _ in range(3):
        s2 = _HONORIFIC.sub('', s.strip())
        if s2 == s:
            break
        s = s2
    s = re.sub(r'[*\s]+$', '', s.strip(' ,.;'))
    return ' '.join(w[:1].upper() + w[1:].lower() if w.isupper() else w for w in s.split())


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        try:
            arr = {ln.split('\t')[1] for ln in C.read_raw(_ARR_TSV).decode('utf8').splitlines()[1:] if '\t' in ln}
        except OSError:
            arr = set()
        out = []
        for r in super().iter_files(ctx, include_dups):
            if '/0_source/BVH/' in r.path:
                continue
            rel = r.path.split('FreEMmax_OA/', 1)[-1]
            if rel in arr:
                continue
            if any(k.split('|', 1)[1] in _TAGKEYS for k in r.c if '|' in k):
                out.append(r)
        return out

    def _hdr_root(self, rec):
        try:
            return C.parse_xml(C.read_raw(_header_path(rec.path)))
        except OSError:
            return None

    def doc_date(self, root, rec, ctx):
        h = self._hdr_root(rec)
        if h is not None:
            d = C.header_dates(h)
            if d.known:
                return d
        d = C.header_dates(root)
        return d if d.known else H.sourcedesc_year(root)

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        if '/Descartes/' in rec.path:
            for el in H.iter_local(root, 'name'):
                if any(C.ln(a.tag) == 'salute' for a in el.iterancestors()):
                    s = C.surface_text(el)
                    rest = re.sub(r'^(A|AU|AUX|À)\s+', '', s, flags=re.I)
                    rest = re.sub(r'(?i)\b(monsieur|mr|messieurs|reverend|pere|r\.|p\.)\b|[\s*.,;]', '', rest)
                    if _HEADING.match(s) and rest:
                        h = H.make_hit(el, 'person', 'name@addressee', 'addressee', 'txt',
                                       ancestors=('TEI', 'text', 'body', 'div', 'opener', 'salute'))
                        h.surface = clean_salutation(s) or h.surface
                        hits.append(h)
        return hits

    def extract_root(self, root, rec, ctx):
        ms, ents = super().extract_root(root, rec, ctx)
        h = self._hdr_root(rec)
        lic = None
        if h is not None:
            for el in H.iter_local(h, 'licence'):
                lic = _spdx(el.get('target'))
                if lic:
                    break
        for m in ms:
            if lic:
                m.licence = lic
                if '-ND' in lic:
                    m.context = None
        return ms, ents


EXTRACTOR = Extractor()
