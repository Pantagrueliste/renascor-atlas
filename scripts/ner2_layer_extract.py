"""NER layer v1, stage 1: inventory of the text archive + extraction of the texts to run the NER model on (runs on Hypatia, CPU).

Reads the tarballs of ``RQ/release/text_archive_v1.0`` (copied to Hypatia, never modified) and the frozen database ``corpus_master_FROZEN_2026-06-26.sqlite``
(catalogue: collection, language, year, date confidence, word count, duplicate flag). For every document of a handled tarball it decides

    ok            read, no editor place markup, dated 1450-1700, not excluded -> text written for the model
    place_markup  the TEI body has placeName / settlement / geogName / country / region / district / bloc (or rs/name typed as a place)
    v1_doc        the document is in data_v1 (train / dev / test passages of the first NER round: ``doc_meta.tsv.gz``)
    v1_edition    the whole collection is, or overlaps with, a train / dev / test edition of data_v1 (collection-level rule, listed in EDITION_COLLECTIONS)
    gold          the document is one of the 30 passages of gold_seed_v1
    dup           flagged duplicate in the catalogue
    year_out      the catalogue year is outside 1450-1700
    lang_other    language outside the languages the models were trained on (en la it es fr de)
    short         fewer than 50 words after cleaning
    parse_error / no_body / unmatched  (member that cannot be read / has no body / is not in the catalogue)

and, for catalogue documents of the archive that no handler read, ``unhandled`` (format not read here) or ``not_found``.

    python scripts/ner2_layer_extract.py run --archive A --db DB --meta M --out O --procs 22
    python scripts/ner2_layer_extract.py merge --out O

Outputs (under --out): ``texts/<lang>/<task>.jsonl.gz`` (one line per document: ``{"id", "t"}``), ``docs_parts/<task>.jsonl`` (one row per document) and, after
``merge``, ``docs.parquet`` and ``shards.tsv``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import io
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import tarfile
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

YEAR_LO, YEAR_HI = 1450, 1700
RUN_LANGS = {'eng': 'en', 'lat': 'la', 'ita': 'it', 'spa': 'es', 'fra': 'fr', 'deu': 'de'}
MIN_WORDS = 50
WORD = re.compile(r'\w+|[^\w\s]', re.U)
PLACE_LOCAL = frozenset(('placeName', 'settlement', 'geogName', 'country', 'region', 'district', 'bloc'))
RS_PLACE = re.compile(r'^(place|location|loc|settlement|region|country|geog\w*|toponym)$', re.I)
SKIP_TAGS = frozenset(('note', 'figure', 'pb', 'fw', 'gap', 'teiHeader', 'del', 'corr', 'reg', 'expan', 'supplied', 'rdg', 'imprimatur', 'surplus', 'ex'))

# collections that are, or overlap with, a v1 train / dev / test edition (data_v1/tables/editions.tsv): excluded as a whole.
EDITION_COLLECTIONS = {
    'BullingerLetters': 'bullinger', 'CKCC_Epistolarium': 'ckcc_epistolarium', 'CAMENA': 'camena_*', 'CERA': 'camena_cera', 'CAMENA_CERA_Letters': 'camena_cera',
    'CroALa': 'croala', 'FreEM': 'freem_bvh / freem_other', 'DTA': 'dta_1600', 'DTA_komplett': 'dta_1600', 'PostScriptum': 'postscriptum',
    'GerManC': 'germanc', 'Casebooks': 'casebooks', 'REED_English': 'reed (test)', 'BVH_Epistemon_Print': 'epistemon_bvh', 'Controversia_Confessio': 'controversia_confessio',
    'Cavriana_Correspondence': 'cavriana', 'BDPN_Novohispano': 'bdpn', 'QHOD': 'qhod_sema', 'Epistemon': 'epistemon_bvh', 'CIDTC': 'cidtc', 'CELT': 'celt',
    'Suriano_Correspondence': 'suriano', 'OldBaileyOnline': 'old_bailey', 'ELAN_crenum': 'elan_crenum (test)', 'ELAN_fnvr': 'elan_fnvr (test)',
}
PRIVATE = {'CELT', 'CIDTC', 'London_Lives', 'MAP'}
# machine-read sources (OCR / handwriting recognition): single letters there are recognition errors, not detached drop capitals; no repair
NO_DROPCAP_REPAIR = {'Mazarinades', 'Gallicorpora_French_HTR', 'NOSCEMUS',
                     # keyed or edited texts where the pattern fires on tables, gaps, numerals and variables rather than on drop capitals (> 10 per 10 000 words in a first pass)
                     'Salamanca', 'EvansTCP_American', 'EBBA_Ballads', 'EMROC_Recipes', 'IMPACTes_OriginalImprint', 'WikisourceDE_Renaissance', 'FrenchEEBO'}


# ---------------------------------------------------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------------------------------------------------
def alnum(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', s.lower())


def stem_of(name: str) -> str:
    b = name.replace('\\', '/').split('/')[-1]
    b = re.sub(r'\.(xml|txt|json|jsonl|odt|epub|zip|html)$', '', b, flags=re.I)
    b = re.sub(r'\.(xml|txt)$', '', b, flags=re.I)
    b = re.sub(r'\.(P5|TEI-P5)$', '', b)
    return b


# family = tarball -> (handler, DB collections, member filter, member key, DB key)
def fam(handler, colls, mkey, dbkey, accept=lambda n: True):
    return {'handler': handler, 'colls': colls, 'mkey': mkey, 'dbkey': dbkey, 'accept': accept}


def _noscemus_id(n):
    m = re.match(r'.*/(\d+)/[^/]+\.txt$', n)
    return m.group(1) if m else None


FAMILIES = {
    'raw__eebotcp_fill': fam('tei', ['EEBO_TCP_fill'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml') and '/xml/' in n),
    'raw__eebo_phase2': fam('tei', ['EEBOphase2'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml') and '/xml/' in n),
    'raw__latineebo': fam('tei', ['LatinEEBO'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml') and '/xml/' in n),
    'raw__corpuscorp': fam('tei', ['CorpusCorp'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml')),
    'raw__bibit_tei': fam('tei', ['BibIT'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml')),
    'nas__mazarinades': fam('tei', ['Mazarinades'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml') and '/extracted/' in n),
    'raw__dracor_fre': fam('tei', ['DraCorFrench'], lambda n: stem_of(n), lambda r: stem_of(r['path'] or r['filename']), lambda n: n.endswith('.xml')),
    'raw__dracor_eng': fam('tei', ['DraCorEnglish'], lambda n: stem_of(n), lambda r: stem_of(r['path']) if r['path'] else re.sub(r'^dracor_eng_', '', stem_of(r['filename'])), lambda n: n.endswith('.xml')),
    'nas__nakala_renaissance': fam('tei', ['Nakala'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: n.endswith('.xml')),
    'raw__salamanca': fam('tei', ['Salamanca'], lambda n: stem_of(n), lambda r: stem_of(r['filename']), lambda n: re.search(r'/W\d+\.xml$', n) is not None),
    'nas__Noscemus_Digital_Sourcebook': fam('txt_noscemus', ['NOSCEMUS'], _noscemus_id, lambda r: r['filename'].split('_')[1] if '_' in r['filename'] else None,
                                            lambda n: n.endswith('.txt')),
    'raw__osta': fam('txt_osta', ['OSTA'], lambda n: re.sub(r'^TEXT\.', '', stem_of(n)), lambda r: stem_of(r['filename']),
                     lambda n: '/transcriptions/TEXT.' in n and n.endswith('.txt')),
    'raw__estilometria': fam('txt', ['Estilometria_SigloOro'], lambda n: 'estilom_' + stem_of(n), lambda r: stem_of(r['filename']),
                             lambda n: '/Siglo_de_Oro/' in n and n.endswith('.txt')),
    'nas__corpus_v4_filtered': fam('txt', ['RoyalSocPre1750', 'WelshEEBO', 'ScottishEEBO', 'FrenchEEBO'], lambda n: stem_of(n), lambda r: stem_of(r['filename']),
                                   lambda n: n.split('/')[0] in ('RoyalSocPre1750', 'WelshEEBO', 'ScottishEEBO', 'FrenchEEBO') and n.endswith('.txt')),
    'nas__ws_it_renaissance': fam('txt', ['WikisourceIT_Renaissance'], lambda n: stem_of(n), lambda r: re.sub(r'^WS_it_', '', stem_of(r['filename'])),
                                  lambda n: n.endswith('.txt')),
    'nas__ws_es_renaissance': fam('txt', ['WikisourceES_Renaissance'], lambda n: stem_of(n), lambda r: re.sub(r'^WS_es_', '', stem_of(r['filename'])),
                                  lambda n: n.endswith('.txt')),
    'raw__impactes': fam('impactes', ['IMPACTes_OriginalImprint'], lambda n: n.split('/')[1] if re.match(r'^impactes/[^/]+/[^/]+\.xml$', n) else None,
                         lambda r: re.sub(r'^impactes_', '', stem_of(r['filename'])), lambda n: n.endswith('.xml')),
    'raw__liberliber': fam('liberliber', ['LiberLiber'], lambda n: alnum(re.sub(r'^LiberLiber_', '', stem_of(n))),
                           lambda r: alnum((r['path'] or r['filename']).replace('LiberLiber/', '').replace('/', '').replace('.txt', '')),
                           lambda n: n.lower().endswith(('.odt', '.epub', '.zip')) and '/files/' in n),
}
# plain jsonl tarballs: one record per document {collection, filename, raw, language, year}
JSONL_TARS = '''raw__EvansTCP_American raw__LEME_English raw__TextGrid_German raw__Galileo_Opere_Favaro raw__WikisourceIT_Renaissance raw__EBBA_Ballads raw__Lampeter_English
raw__Hakluyt_English raw__Newton_Project raw__InterregnumActs raw__StatutesOfTheRealm raw__WikisourceDE_Renaissance raw__LatinWikisource raw__WikisourceFR_Renaissance
raw__WikisourceES_Renaissance raw__ABU_French raw__Gallicorpora_French_HTR raw__SpanishGoldenAgeProse_WS raw__WikisourceES raw__WikisourceIT raw__WikisourceBibles_Renaissance
raw__WikisourceBibles_Italian raw__WikisourceCA raw__WikisourcePT_Renaissance raw__WikisourceNL_Renaissance raw__WikisourcePL_Renaissance raw__BVH_Epistemon_Print
raw__EMROC_Recipes raw__Controversia_Confessio raw__CAMENA_CERA_Letters raw__ckcc raw__Wikisource'''.split()


def load_catalogue(db_path: str):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        'select collection, filename, path, language, year, date_confidence, wordcount, is_dup, copyright_status, access_regime from texts')]
    return rows


def load_attributions(path: str):
    return {r['collection']: r for r in csv.DictReader(open(path, encoding='utf-8'), delimiter='\t')}


def doc_id_of(row) -> str:
    return f"{row['collection']}/{stem_of(row['filename'])}"


# ---------------------------------------------------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------------------------------------------------
def local(el):
    t = el.tag
    if not isinstance(t, str):
        return None
    return t.rsplit('}', 1)[-1]


def tei_body_text(root):
    """Diplomatic reading of the primary text (body only): notes, figures, page / line furniture and editorial additions are left out; the printed form
    (orig / abbr / sic) is kept; end-of-line hyphen glyphs and TCP line-break marks join the two halves of a word. Returns (text, has_place_markup, has_body)."""
    body = None
    for el in root.iter():
        if local(el) == 'body':
            body = el
            break
    if body is None:
        return '', False, False
    place = False
    buf: list[str] = []

    def walk(el):
        nonlocal place
        tag = local(el)
        if tag is None:
            return
        if tag in PLACE_LOCAL:
            place = True
        elif tag in ('rs', 'name'):
            ty = el.get('type') or el.get('role') or ''
            if RS_PLACE.match(ty):
                place = True
        if tag in SKIP_TAGS:
            return
        if tag == 'lb':
            if el.get('break') == 'no' or el.get('rend') == '¬':
                buf.append('\u0001')
            else:
                buf.append('\n')
            return
        if tag == 'g' and (el.get('ref') or '').endswith('EOLhyphen'):
            return
        if tag in ('p', 'l', 'lg', 'head', 'item', 'salute', 'dateline', 'closer', 'signed', 'trailer', 'label', 'speaker', 'stage', 'div', 'sp') and buf and not buf[-1].endswith('\n'):
            buf.append('\n')
        if el.text:
            buf.append(el.text)
        for c in el:
            walk(c)
            if c.tail:
                buf.append(c.tail)
        if tag in ('p', 'lg', 'head', 'item', 'sp'):
            buf.append('\n')

    walk(body)
    txt = ''.join(buf).replace('∣', '')           # TCP line-break mark inside words
    txt = re.sub(r'\u0001\s*', '', txt)
    return txt, place, True


def clean_text(t: str) -> str:
    t = html.unescape(t) if '&' in t and re.search(r'&(#\d+|\w+);', t) else t
    t = t.replace('\r', '')
    t = t.replace('­', '').replace('​', '').replace('﻿', '')
    t = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', ' ', t)
    t = re.sub(r'(?<=\w)[¬⸗-]\n\s*(?=[a-zà-ÿſ])', '', t)      # end-of-line hyphen before a lower-case letter: joined
    t = re.sub(r'(?<=\w)¬\n\s*(?=\w)', '', t)
    t = re.sub(r'[ \t\f\v]+', ' ', t)
    t = re.sub(r'\n\s*', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def osta_text(raw: str) -> str:
    t = re.sub(r'\{[^{}]*\}', ' ', raw)                 # {RMK: ...}, {CB1.}, {MIN.} control lines
    t = re.sub(r'\[(?:fol|f|p|pag)[^\]]{0,20}\]', ' ', t)
    t = t.replace('<', '').replace('>', '')            # editorial expansions <n> ... : kept as letters
    return t


def noscemus_text(raw: str) -> str:
    i = raw.find('End of metadata section')
    if i >= 0:
        raw = raw[i + len('End of metadata section'):]
        raw = raw.lstrip('-\n ')
    return raw


def odt_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        x = z.read('content.xml').decode('utf-8', 'replace')
    x = re.sub(r'</text:(p|h)>', '\n', x)
    x = re.sub(r'<text:line-break/>', '\n', x)
    x = re.sub(r'<text:s[^>]*/>', ' ', x)
    x = re.sub(r'<[^>]+>', '', x)
    return html.unescape(x)


def epub_text(data: bytes) -> str:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in sorted(z.namelist()):
            if n.lower().endswith(('.xhtml', '.html', '.htm')):
                x = z.read(n).decode('utf-8', 'replace')
                x = re.sub(r'</(p|h\d|div|li)>', '\n', x)
                x = re.sub(r'<br\s*/?>', '\n', x)
                x = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', x, flags=re.S)
                out.append(html.unescape(re.sub(r'<[^>]+>', '', x)))
    return '\n'.join(out)


def zip_txt(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        n = [x for x in z.namelist() if x.lower().endswith('.txt')]
        raw = z.read(n[0])
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1', 'replace')


def decode_bytes(b: bytes) -> str:
    for enc in ('utf-8', 'cp1252'):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode('latin-1', 'replace')



# ---------------------------------------------------------------------------------------------------------------------
# detached drop capitals ("L ondon", "f rance"): vendored from ratlas.ner.ner2_passages (NER v2 chain, 2026-10-21 version) so that this layer does not depend
# on a module that is still being edited
# ---------------------------------------------------------------------------------------------------------------------
import unicodedata

ALPHA = 'A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u017f\u00c6\u0152\u00e6\u0153'
LEGIT_SINGLE = {'en': set('aAIOoy'), 'la': set('aAeEoOiI&'), 'it': set('aAeEoOiI\u00e9\u00c9\u00e8\u00c8\u00e0\u00c0'), 'es': set('aAeEoOyYuU'),
                'fr': set('aAyY\u00e0\u00c0\u00f4\u00d4'), 'de': set('aAiIoO')}
DROPCAP_PAT = re.compile(r"(?<![\w'\u2019\-\u0300-\u036f])([^\W\d_]) ([a-z\u00e0-\u00ff\u017f]{2,})(?![\w\u0300-\u036f]|\w*['\u2019])")      # unicode-aware boundaries (combining tildes: "appetũt vulneratæ" is not a drop capital)


def fold(s: str) -> str:
    s = s.replace('\u017f', 's')
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').lower()


def doc_vocab(text: str):
    t = DROPCAP_PAT.sub(' ', text)
    return Counter(fold(w) for w in re.findall(rf'[{ALPHA}]+', t))


def repair_dropcaps(text: str, lang: str, vocab=None, sample=None):
    """Glue detached drop capitals to their word ("L ondon" -> "London"). Pattern of ner2_passages.repair_dropcaps: a lone letter that is not a one-letter word of the
    language, followed by one space and a lower-case word of >= 2 letters, at a word boundary. Two changes against that version, found on EEBO samples:
      * a lower-case lone letter ("h ad", "e foced", "m ght": keying gaps and broken words as often as drop capitals) is glued only when the joined word occurs
        elsewhere in the document; a capital is glued when the joined word occurs elsewhere or when the second half is not a word of the document on its own;
      * J / j (printed for I / i in 17th-century English, Latin and French: "J pray you") is a one-letter word."""
    legit = LEGIT_SINGLE.get(lang, set('aAeEoOiI')) | set('Jj')
    out, n, pos = [], 0, 0
    for m in DROPCAP_PAT.finditer(text):
        letter, rest = m.group(1), m.group(2)
        if letter in legit or (ord(letter) > 0x7f and letter not in '\u017f'):
            continue
        if vocab is not None:
            joined = fold(letter + rest)
            known = vocab.get(joined, 0) > 0
            if letter.islower() and letter != '\u017f':
                if not known:
                    continue
            elif not (known or vocab.get(fold(rest), 0) == 0):
                continue
        out.append(text[pos:m.start()])
        out.append(letter + rest)
        n += 1
        pos = m.end()
        if sample is not None and len(sample) < 12:
            sample.append(text[max(0, m.start() - 25):m.end() + 20].replace('\n', ' ') + '   => ' + letter + rest)
    out.append(text[pos:])
    return ''.join(out), n


# ---------------------------------------------------------------------------------------------------------------------
# one task = (tarball, shard k of K)
# ---------------------------------------------------------------------------------------------------------------------
G: dict = {}          # globals shared with the forked workers


def row_status_base(row, atts) -> str | None:
    """Status decided from the catalogue alone (None = still ok)."""
    coll = row['collection']
    if row['is_dup']:
        return 'dup'
    y = row['year']
    if y is None or not (YEAR_LO <= y <= YEAR_HI):
        return 'year_out'
    if coll in PRIVATE:
        return 'private'
    if coll in EDITION_COLLECTIONS:
        return 'v1_edition'
    return None


def make_doc_row(row, task, member, status, n_words=0, n_tok=0, n_chars=0, note='', n_rep=0):
    att = G['atts'].get(row['collection']) or {}
    return {'doc_id': doc_id_of(row), 'collection': row['collection'], 'member': member, 'task': task, 'lang_db': row['language'], 'lang': RUN_LANGS.get(row['language'], ''),
            'year': row['year'], 'date_confidence': row['date_confidence'], 'tier': att.get('tier', ''), 'licence_tag': att.get('license_tag', ''),
            'db_words': row['wordcount'], 'n_words': n_words, 'n_tok': n_tok, 'n_chars': n_chars, 'n_dropcap': n_rep, 'status': status, 'note': note}


def process_task(task):
    tar_base, k, K = task
    from lxml import etree
    out = Path(G['out'])
    name = f'{tar_base}.{k:02d}'
    fam_ = FAMILIES.get(tar_base)
    is_jsonl = tar_base in JSONL_TARS
    path = Path(G['archive']) / f'{tar_base}.tar.gz'
    rows_out = []
    texts = {}                   # lang -> list of (id, text)
    stats = Counter()
    seen_db = set()
    unmatched = Counter()
    lib_best: dict = {}          # liberliber: key -> (prio, member name, bytes)
    page_groups: dict = {}       # impactes: directory -> (row, [(member, bytes)])
    repair_samples: list = []    # (collection, doc, context) of the first drop-capital repairs, for the quality check
    if not path.exists():
        return {'task': name, 'rows': [], 'stats': {'missing_tar': 1}}
    t0 = time.time()
    if fam_:
        dbidx = G['dbidx'][tar_base]
    else:
        dbidx = G['fn_idx']
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True, recover=False)

    def finish(row, member, txt, place=False, has_body=True, kind='tei'):
        st = row_status_base(row, G['atts'])
        did = doc_id_of(row)
        seen_db.add(id(row))
        if st is None and member in G['v1_paths']:
            st = 'v1_doc'
        if st is None and stem_of(row.get('filename') or '') in G['gold_docs']:
            st = 'gold'
        if st is None and not has_body:
            st = 'no_body'
        if st is None and place:
            st = 'place_markup'
        if st is None and RUN_LANGS.get(row['language']) is None:
            st = 'lang_other'
        n_words = n_tok = n_chars = n_rep = 0
        if st is None:
            txt = clean_text(txt)
            lg = RUN_LANGS[row['language']]
            samp = []
            if row['collection'] in NO_DROPCAP_REPAIR:
                n_rep = 0
            else:
                txt, n_rep = repair_dropcaps(txt, lg, doc_vocab(txt), samp)
                if samp and len(repair_samples) < 400:
                    repair_samples.extend((row['collection'], did, x) for x in samp[:3])
            n_words = len(txt.split())
            if n_words < MIN_WORDS:
                st = 'short'
            else:
                n_tok = len(WORD.findall(txt))
                n_chars = len(txt)
                st = 'ok'
                texts.setdefault(RUN_LANGS[row['language']], []).append((did, txt))
        rows_out.append(make_doc_row(row, name, member, st, n_words, n_tok, n_chars, n_rep=n_rep))

    def handle(member, data):
        """data: bytes of one archive member (non-jsonl)."""
        key = fam_['mkey'](member)
        if key is None:
            return
        row = dbidx.get(key)
        if row is None:
            unmatched[os.path.splitext(member)[1].lower()] += 1
            return
        if id(row) in seen_db:
            return
        h = fam_['handler']
        # decide from the catalogue first: saves parsing
        st0 = row_status_base(row, G['atts'])
        if st0 is not None and st0 != 'v1_edition':
            finish_nodata(row, member, st0)
            return
        if st0 == 'v1_edition' or (member in G['v1_paths']) or RUN_LANGS.get(row['language']) is None:
            finish_nodata(row, member, st0 or ('v1_doc' if member in G['v1_paths'] else 'lang_other'))
            return
        try:
            if h == 'tei':
                root = etree.fromstring(data, parser)
                txt, place, has_body = tei_body_text(root)
                finish(row, member, txt, place, has_body)
            elif h == 'txt':
                finish(row, member, decode_bytes(data))
            elif h == 'txt_osta':
                finish(row, member, osta_text(decode_bytes(data)))
            elif h == 'txt_noscemus':
                finish(row, member, noscemus_text(decode_bytes(data)))
            elif h == 'impactes':
                page_groups.setdefault(key, (row, []))[1].append((member, data))
            elif h == 'liberliber':
                ext = os.path.splitext(member)[1].lower()
                prio = {'.odt': 0, '.epub': 1, '.zip': 2}.get(ext, 9)
                cur = lib_best.get(key)
                if cur is None or prio < cur[0]:
                    lib_best[key] = (prio, member, data, row)
        except Exception as e:                        # unreadable member: recorded, never fatal
            rows_out.append(make_doc_row(row, name, member, 'parse_error', note=repr(e)[:80]))
            seen_db.add(id(row))

    def finish_nodata(row, member, st):
        seen_db.add(id(row))
        rows_out.append(make_doc_row(row, name, member, st))

    with open(path, 'rb') as fh:
        tf = tarfile.open(fileobj=fh, mode='r|gz')
        for m in tf:
            if not m.isfile():
                continue
            nm = m.name
            if is_jsonl:
                if not nm.endswith('.jsonl') or zlib.crc32(nm.encode()) % K != k:
                    continue
                for line in tf.extractfile(m).read().decode('utf-8').split('\n'):
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    fn = d.get('filename') or ''
                    cands = G['fn_idx'].get(fn) or []
                    row = None
                    for r_ in cands:
                        if r_['collection'] == tar_base.split('__', 1)[-1] or len(cands) == 1:
                            row = r_
                            break
                    if row is None and cands:
                        row = cands[0]
                    if row is None:
                        y = None
                        try:
                            y = int(str(d.get('year') or '')[:4])
                        except ValueError:
                            pass
                        row = {'collection': (d.get('collection') or tar_base).replace(' ', '_')[:40], 'filename': fn, 'path': '', 'language': d.get('language') or '',
                               'year': y, 'date_confidence': 'jsonl', 'wordcount': 0, 'is_dup': 0}
                        stats['jsonl_no_catalogue_row'] += 1
                    if id(row) in seen_db:
                        continue
                    finish(row, f'{nm}#{fn}', d.get('raw') or '')
                continue
            if not fam_ or not fam_['accept'](nm):
                continue
            if zlib.crc32(nm.encode()) % K != k:
                continue
            handle(nm, tf.extractfile(m).read())
    for key, (row, pages) in page_groups.items():
        try:
            lines = []
            for member, data in sorted(pages):
                root = etree.fromstring(data, parser)
                got = []
                for tl in root.iter():
                    if local(tl) == 'TextLine':
                        for u in tl.iter():
                            if local(u) == 'Unicode' and u.text:
                                got.append(u.text)
                                break
                if not got:
                    for u in root.iter():
                        if local(u) == 'Unicode' and u.text:
                            got.append(u.text)
                lines.append('\n'.join(got))
            finish(row, f'impactes/{key}', '\n\n'.join(lines))
        except Exception as e:
            rows_out.append(make_doc_row(row, name, f'impactes/{key}', 'parse_error', note=repr(e)[:80]))
    for key, (prio, member, data, row) in lib_best.items():
        try:
            ext = os.path.splitext(member)[1].lower()
            t = odt_text(data) if ext == '.odt' else epub_text(data) if ext == '.epub' else zip_txt(data)
            n = len(t)
            t = t[int(n * 0.03): int(n * 0.97)]               # e-book front and back matter
            t = re.sub(r'\[\d+\]|\*+', ' ', t)
            finish(row, member, t)
        except Exception as e:
            rows_out.append(make_doc_row(row, name, member, 'parse_error', note=repr(e)[:80]))
    # write
    for lang, items in texts.items():
        d = out / 'texts' / lang
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / f'{name}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as f:
            for did, t in items:
                f.write(json.dumps({'id': did, 't': t}, ensure_ascii=False) + '\n')
    d = out / 'repair_samples'
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f'{name}.tsv', 'w', encoding='utf-8') as f:
        for c_, did_, x_ in repair_samples:
            f.write(f'{c_}\t{did_}\t{x_}\n')
    d = out / 'docs_parts'
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f'{name}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    st = Counter(r['status'] for r in rows_out)
    print(f'[{time.strftime("%H:%M:%S")}] {name}: {len(rows_out)} docs {dict(st)} unmatched {dict(unmatched)} {time.time() - t0:.0f}s', flush=True)
    return {'task': name, 'n': len(rows_out), 'status': dict(st), 'unmatched': dict(unmatched)}


# ---------------------------------------------------------------------------------------------------------------------
def build_globals(a):
    G['archive'] = a.archive
    G['out'] = a.out
    rows = load_catalogue(a.db)
    G['atts'] = load_attributions(str(Path(a.archive) / 'ATTRIBUTIONS.tsv'))
    G['rows'] = rows
    # only the documents of the archive (status 'in archive') are in the inventory
    inarch = {c for c, r in G['atts'].items() if r['status'].startswith('in archive')}
    G['inarch'] = inarch
    dbidx = {}
    for tb, fm in FAMILIES.items():
        idx = {}
        for r in rows:
            if r['collection'] in fm['colls'] and not r['is_dup']:
                k_ = fm['dbkey'](r)
                if k_:
                    idx.setdefault(k_, r)
        dbidx[tb] = idx
    G['dbidx'] = dbidx
    fn_idx = defaultdict(list)
    for r in rows:
        if r['collection'] in inarch and not r['is_dup']:
            fn_idx[r['filename']].append(r)
    G['fn_idx'] = dict(fn_idx)
    v1 = set()
    import pandas as pd
    dm = pd.read_csv(str(Path(a.meta) / 'doc_meta.tsv.gz'), sep='\t', usecols=['source_path'])
    v1.update(dm['source_path'].astype(str))
    G['v1_paths'] = v1
    gold = set()
    for line in open(Path(a.meta) / 'gold_seed_v1.jsonl', encoding='utf-8'):
        gold.add(json.loads(line)['doc'])
    G['gold_docs'] = gold


def cmd_run(a):
    build_globals(a)
    tasks = []
    arch = Path(a.archive)
    for tb in list(FAMILIES) + JSONL_TARS:
        if not (arch / f'{tb}.tar.gz').exists():
            print('missing tar', tb)
            continue
        K = a.shards.get(tb, 1)
        for k in range(K):
            tasks.append((tb, k, K))
    if a.only:
        tasks = [t for t in tasks if t[0] in a.only]
    print(len(tasks), 'tasks', flush=True)
    tasks.sort(key=lambda t: -(arch / f'{t[0]}.tar.gz').stat().st_size / t[2])
    with mp.get_context('fork').Pool(a.procs) as pool:
        for res in pool.imap_unordered(process_task, tasks, chunksize=1):
            pass
    print('done', flush=True)


def cmd_merge(a):
    import pandas as pd
    out = Path(a.out)
    rows = []
    for p in sorted((out / 'docs_parts').glob('*.jsonl')):
        for line in open(p, encoding='utf-8'):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates('doc_id', keep='first')
    df.to_parquet(out / 'docs_read.parquet', index=False)
    print(len(df), df['status'].value_counts().to_dict())
    # the catalogue documents of the archive that no reader produced a row for
    G['atts'] = load_attributions(str(Path(a.archive) / 'ATTRIBUTIONS.tsv'))
    cat = load_catalogue(a.db)
    inarch = {c for c, r in G['atts'].items() if r['status'].startswith('in archive')}
    handled = set(df['collection'])
    have = set(df['doc_id'])
    extra = []
    for r in cat:
        if r['collection'] not in inarch:
            continue
        did = doc_id_of(r)
        if did in have:
            continue
        st = row_status_base(r, G['atts']) or ('not_found' if r['collection'] in handled else 'unhandled')
        extra.append(make_doc_row(r, '', '', st))
    allf = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    allf.to_parquet(out / 'docs_all.parquet', index=False)
    print('inventory rows', len(allf), allf['status'].value_counts().to_dict())
    shards = []
    for p in sorted((out / 'texts').glob('*/*.jsonl.gz')):
        lang = p.parent.name
        sub = df[(df['task'] == p.name[:-len('.jsonl.gz')]) & (df['status'] == 'ok') & (df['lang'] == lang)]
        shards.append((str(p.relative_to(out)), lang, len(sub), int(sub['n_words'].sum()), int(sub['n_tok'].sum())))
    with open(out / 'shards.tsv', 'w') as f:
        f.write('file\tlang\tn_docs\tn_words\tn_tok\n')
        for s_ in shards:
            f.write('\t'.join(map(str, s_)) + '\n')
    print(len(shards), 'shards')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--archive', required=True)
    r.add_argument('--db', required=True)
    r.add_argument('--meta', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--procs', type=int, default=8)
    r.add_argument('--only', nargs='*')
    r.add_argument('--shards', type=json.loads, default={'raw__eebotcp_fill': 24, 'raw__eebo_phase2': 24, 'nas__Noscemus_Digital_Sourcebook': 4})
    m = sub.add_parser('merge')
    m.add_argument('--out', required=True)
    m.add_argument('--archive', required=True)
    m.add_argument('--db', required=True)
    a = ap.parse_args()
    {'run': cmd_run, 'merge': cmd_merge}[a.cmd](a)


if __name__ == '__main__':
    main()
