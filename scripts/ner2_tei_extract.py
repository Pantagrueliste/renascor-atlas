"""NER layer, TEI editions, step 1 (CPU, Hypatia): running text and editor name tags of the documents the first pass left out (status v1_edition / place_markup).

    python scripts/ner2_tei_extract.py cands   --docs-all layer_v1/extract/docs_all.parquet --db FROZEN.sqlite --out layer_tei/cands.parquet
    python scripts/ner2_tei_extract.py run     --archive A --cands C --meta layer_v1/meta_in --out layer_tei/work --procs 22 [--only bullinger ...]
    python scripts/ner2_tei_extract.py merge   --cands C --out layer_tei/work

Each document is found in the archive tarballs from its catalogue row (SOURCES below). Documents that are TEI are read by ``ner2_tei_text.extract_tei`` (body only,
diplomatic reading, editorial notes / apparatus / editorial divisions left out, tag ranges kept through the clean-up). Documents the archive holds as plain text
go through the same clean-up without tags (``source_format`` = plain). Documents in a language the models do not cover are not read (lang_other). Documents whose pieces are in the v1 train / dev / test lists
(``doc_meta.tsv.gz``, share >= V1_DROP_SHARE) or in the gold seed are read too and flagged (v1_doc, gold): the assembly step keeps them out of the main output
(the first pass excluded them) and writes them to a side set.

Outputs (under --out): ``texts/<src>.<k>.jsonl.gz`` ({id, t, lang}), ``tags_parts/*.parquet``, ``docs_parts/*.jsonl``, ``survey/*.json``; after ``merge``:
``docs_read.parquet``, ``tags_read.parquet``, ``survey.tsv``, ``inventory.tsv``.
"""
from __future__ import annotations

import argparse
import gzip
import html as htmllib
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import tarfile
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner2_layer_extract as L  # noqa: E402
import ner2_tei_text as T  # noqa: E402

V1_DROP_SHARE = 0.5
STATUS_EDITION = ('v1_edition', 'place_markup')
NAMEISH = frozenset(('persName', 'placeName', 'orgName', 'name', 'rs', 'settlement', 'country', 'region', 'geogName', 'geogFeat', 'bloc', 'district', 'term', 'title',
                     'roleName', 'forename', 'surname', 'addName', 'nameLink', 'genName', 'author', 'date', 'ref', 'ptr', 'seg', 'span', 'w', 'index'))
CKCC_LANG = {'Latin': 'lat', 'Dutch': 'nld', 'French': 'fra', 'English': 'eng', 'German': 'deu', 'Italian': 'ita', 'Spanish': 'spa'}


def alnum_fold(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^0-9a-z]', '', s.lower())


def base(n: str) -> str:
    return n.replace('\\', '/').split('/')[-1]


def stem_nodots(name: str) -> str:
    return re.sub(r'\.(xml|txt)$', '', base(name))


def doc_id_of(coll: str, filename: str) -> str:
    return f'{coll}/{L.stem_of(filename)}'


# source = tarball + how a member / a catalogue row give the same key. agg: several members make one document (key = document key)
SOURCES = {
    'bullinger': dict(tar='raw__bullinger', fmt='tei', colls=['BullingerLetters'], shards=8,
                      mkey=lambda n: n if re.match(r'^bullinger/\d+\.xml$', n) else None, rkey=lambda r: r['path']),
    'dtae': dict(tar='raw__dtae', fmt='tei', colls=['DTA', 'DTA_komplett'], shards=6,
                 mkey=lambda n: base(n) if n.endswith('.xml') and '/dta_komplett_' in n else None, rkey=lambda r: re.sub(r'\.txt$', '.xml', r['filename'])),
    'postscriptum': dict(tar='raw__postscriptum', fmt='tei', colls=['PostScriptum'], shards=2,
                         mkey=lambda n: base(n) if n.endswith('.xml') and 'XML-TEI_P5' in n else None, rkey=lambda r: base(r['path'])),
    'qhod': dict(tar='raw__qhod', fmt='tei', colls=['QHOD'], shards=1,
                 mkey=lambda n: re.sub(r'^(qhod/)+', '', n) if n.startswith('qhod/') and n.endswith('.xml') else None, rkey=lambda r: re.sub(r'^qhod/', '', r['path'])),
    'germanc': dict(tar='raw__de_dh', fmt='tei', colls=['GerManC'], shards=1,
                    mkey=lambda n: stem_nodots(n) if '/germanc/TEI/' in n and n.endswith('.xml') else None, rkey=lambda r: re.sub(r'RAW$', '', stem_nodots(r['filename']))),
    'nakala': dict(tar='nas__nakala_renaissance', fmt='tei', colls=['Nakala'], shards=1,
                   mkey=lambda n: L.stem_of(n) if n.endswith('.xml') else None, rkey=lambda r: L.stem_of(r['filename'])),
    'salamanca': dict(tar='raw__salamanca', fmt='tei', colls=['Salamanca'], shards=1,
                      mkey=lambda n: L.stem_of(n) if re.search(r'/W\d+\.xml$', n) else None, rkey=lambda r: L.stem_of(r['filename'])),
    'croala': dict(tar='raw__croala', fmt='tei', colls=['CroALa'], shards=1,
                   mkey=lambda n: stem_nodots(n) if '/txts/' in n and n.endswith('.xml') else None, rkey=lambda r: stem_nodots(r['filename'])),
    'bdpn': dict(tar='raw__bdpn', fmt='tei', colls=['BDPN_Novohispano'], shards=1, fuzzy=True,
                 mkey=lambda n: re.sub(r'\d+$', '', alnum_fold(stem_nodots(n))) if '/archivosXMLDocumentos/' in n and n.endswith('.xml') else None,
                 rkey=lambda r: re.sub(r'\d+$', '', alnum_fold(re.sub(r'^bdpn_', '', stem_nodots(r['filename']))))),
    'cavriana': dict(tar='raw__cavriana', fmt='tei', colls=['Cavriana_Correspondence'], shards=1, agg=True,
                     mkey=lambda n: 'all' if re.match(r'^cavriana/letters/[^/]+\.xml$', n) else None, rkey=lambda r: 'all'),
    'suriano': dict(tar='raw__suriano', fmt='tei', colls=['Suriano_Correspondence'], shards=1, agg=True,
                    mkey=lambda n: (re.match(r'^suriano/datasource/tei/(\d+)/[^/]+\.xml$', n) or [None, None])[1] if n.endswith('.xml') else None,
                    rkey=lambda r: (re.search(r'vol(\d+)', r['filename']) or [None, None])[1]),
    'ckcc': dict(tar='raw__ckcc', fmt='ckcc', colls=['CKCC_Epistolarium'], shards=1, agg=True, mkey=None, rkey=lambda r: stem_nodots(r['filename'])),
    'reed': dict(tar='raw__REED_English', fmt='jsonl', colls=['REED_English'], shards=1, rkey=lambda r: r['filename']),
    'bvh': dict(tar='raw__BVH_Epistemon_Print', fmt='jsonl', colls=['BVH_Epistemon_Print'], shards=1, rkey=lambda r: r['filename']),
    'controversia': dict(tar='raw__Controversia_Confessio', fmt='jsonl', colls=['Controversia_Confessio'], shards=1, rkey=lambda r: r['filename']),
    'camena_cera_letters': dict(tar='raw__CAMENA_CERA_Letters', fmt='jsonl', colls=['CAMENA_CERA_Letters'], shards=1, rkey=lambda r: r['filename']),
    'casebooks': dict(dir='', fmt='tei', colls=['Casebooks'], shards=1, agg=True, mkey=None,
                      dkey=lambda data: (re.search(rb'<altIdentifier type="fn_number"><idno n="([A-Za-z0-9]+)"', data) or [None, b''])[1].decode() or None,
                      rkey=lambda r: re.sub(r'^casebooks_', '', stem_nodots(r['filename']))),
    'dtae_dumps': dict(tar='raw__dtae_dumps', fmt='txt', colls=['DTA'], shards=1, fallback=True,
                       mkey=lambda n: re.sub(r'\.txt$', '', base(n)) if re.match(r'^dtae_dumps/\d+-\d+/[^/]+\.txt$', n) else None,
                       rkey=lambda r: re.sub(r'(\.TEI-P5)?\.txt$', '', r['filename'])),
    'corpus_v4': dict(tar='nas__corpus_v4_filtered', fmt='txt', colls=['FreEM', 'CAMENA', 'CERA'], shards=6,
                      mkey=lambda n: n if n.endswith('.txt') and n.split('/')[0] in ('FreEM', 'CAMENA', 'CERA') else None, rkey=lambda r: r['path']),
}
G: dict = {}


# ---------------------------------------------------------------------------------------------------------------------
def cmd_cands(a):
    import pandas as pd
    d = pd.read_parquet(a.docs_all)
    x = d[d['status'].isin(STATUS_EDITION)].copy()
    db = sqlite3.connect(f'file:{a.db}?mode=ro', uri=True)
    rows = pd.read_sql('select collection, filename, path, language, year, wordcount, is_dup from texts where is_dup = 0', db)
    rows['doc_id'] = rows['collection'] + '/' + rows['filename'].map(L.stem_of)
    rows = rows.drop_duplicates('doc_id', keep='first')
    m = x.merge(rows[['doc_id', 'filename', 'path']], on='doc_id', how='left')
    n0 = len(m)
    m = m.drop_duplicates('doc_id', keep='first').copy()      # the inventory lists one document twice (FreEM/MG-1693-03_dAlembert)
    m['path'] = m['path'].fillna('')
    m['filename'] = m['filename'].fillna('')
    m.to_parquet(a.out, index=False)
    print(len(m), 'candidates (', n0 - len(m), 'duplicate inventory rows removed )', m['status'].value_counts().to_dict(), 'without catalogue row:', int((m['filename'] == '').sum()))


def load_globals(a):
    import pandas as pd
    c = pd.read_parquet(a.cands)
    G['cands'] = c.to_dict('records')
    G['archive'] = a.archive
    G['out'] = a.out
    dm = pd.read_csv(str(Path(a.meta) / 'doc_meta.tsv.gz'), sep='\t', usecols=['source_path', 'slug'])
    stems = defaultdict(set)
    for sp, sl in zip(dm['source_path'].astype(str), dm['slug'].astype(str)):
        if sl == 'ckcc_epistolarium':
            m = re.search(r'/data/([a-z]+\d+)/', sp)
            stems[sl].add(alnum_fold(m.group(1)) if m else stem_key(sp))
        else:
            stems[sl].add(stem_key(sp))
    G['v1_stems'] = stems
    gold = set()
    for line in open(Path(a.meta) / 'gold_seed_v1.jsonl', encoding='utf-8'):
        gold.add(json.loads(line)['doc'])
    G['gold'] = gold
    by_coll = defaultdict(list)
    for r in G['cands']:
        by_coll[r['collection']].append(r)
    G['by_coll'] = by_coll


def rows_for(src):
    out = []
    for c in SOURCES[src]['colls']:
        out += G['by_coll'].get(c, [])
    if SOURCES[src].get('fallback'):                  # only the documents no other source produced
        out = [r for r in out if r['doc_id'] not in G.get('done', set())]
    return out


# ---------------------------------------------------------------------------------------------------------------------
def nonspace(s):
    return len(''.join(s.split()))


def file_excl(root):
    """(header, body_all, front_back) non-space character counts of a TEI file."""
    hdr = body = whole = 0
    hdr_el = body_el = None
    for el in root.iter():
        t = T.local(el)
        if t == 'teiHeader' and hdr_el is None:
            hdr_el = el
        elif t == 'body' and body_el is None:
            body_el = el
        if hdr_el is not None and body_el is not None:
            break
    whole = nonspace(''.join(root.itertext()))
    if hdr_el is not None:
        hdr = nonspace(''.join(hdr_el.itertext()))
    if body_el is not None:
        body = nonspace(''.join(body_el.itertext()))
    return hdr, body, max(0, whole - hdr - body)


def survey_of(root, coll, surv):
    body = None
    for el in root.iter():
        if T.local(el) == 'body':
            body = el
            break
    if body is None:
        return
    seen = set()
    for el in body.iter():
        t = T.local(el)
        if t is None:
            continue
        surv['n'][f'{coll}\t{t}'] += 1
        seen.add(t)
        if t in NAMEISH or t in ('note', 'div', 'head'):
            ty = el.get('type') or el.get('role') or el.get('subtype') or el.get('place') or ''
            if ty:
                surv['t'][f'{coll}\t{t}\t{ty}'] += 1
    for t in seen:
        surv['d'][f'{coll}\t{t}'] += 1


def tag_stats(tags):
    c = Counter()
    for t in tags:
        tg = t['tag']
        if tg in T.PLACE_TAGS or (tg in ('rs', 'name') and T.PLACE_TYPED.match(t['type'] or '')):
            c['place'] += 1
        elif tg == 'persName' or (tg in ('rs', 'name') and (t['type'] or '').lower() in ('person', 'perosn', 'peron', 'pers', 'persons')):
            c['pers'] += 1
        elif tg == 'orgName':
            c['org'] += 1
        else:
            c['other'] += 1
    return c


def base_row(r, src, fmt, member, status, note=''):
    return {'doc_id': r['doc_id'], 'collection': r['collection'], 'src': src, 'member': member, 'lang_db': r['lang_db'], 'lang': L.RUN_LANGS.get(r['lang_db'], ''),
            'year': r['year'], 'date_confidence': r['date_confidence'], 'tier': r['tier'], 'licence_tag': r['licence_tag'], 'db_words': r['db_words'],
            'inv_status': r['status'], 'source_format': fmt, 'status': status, 'note': note, 'n_words': 0, 'n_tok': 0, 'n_chars': 0, 'n_dropcap': 0, 'n_tags': 0,
            'n_tags_place': 0, 'n_tags_pers': 0, 'n_tags_org': 0, 'n_tags_other': 0, 'place_markup': False, 'kept_nows': 0, 'ex_header': 0, 'ex_front_back': 0,
            'ex_note_editorial': 0, 'ex_note_other': 0, 'ex_apparatus': 0, 'ex_div_editorial': 0, 'ex_head_editorial': 0, 'ex_standoff': 0, 'ex_diplomatic': 0,
            'dropped_tags': 0, 'n_members': 0, 'v1_hits': 0, 'v1_share': 0.0, 'v1_doc': False, 'gold': False}


# data_v1 (doc_meta.tsv.gz) lists the source files of its passages; their paths differ from the archive members, so the match is on the file stem, within the
# slugs that belong to the collection
COLL_SLUGS = {
    'BullingerLetters': ('bullinger',), 'DTA': ('dta_1600', 'sandrart'), 'DTA_komplett': ('dta_1600', 'sandrart'), 'PostScriptum': ('postscriptum',), 'GerManC': ('germanc',),
    'CroALa': ('croala',), 'BDPN_Novohispano': ('bdpn',), 'Cavriana_Correspondence': ('cavriana',), 'CAMENA': ('camena_cera', 'camena_poemata', 'camena_thesaurus', 'camena_historicapolitica'),
    'CERA': ('camena_cera',), 'CAMENA_CERA_Letters': ('camena_cera',), 'BVH_Epistemon_Print': ('epistemon_bvh', 'freem_bvh'), 'FreEM': ('freem_bvh', 'freem_other'),
    'REED_English': ('reed',), 'Casebooks': ('casebooks',), 'CKCC_Epistolarium': ('ckcc_epistolarium',), 'Controversia_Confessio': ('controversia_confessio',), 'QHOD': ('qhod_sema',),
}


def stem_key(m: str) -> str:
    b = base(m.split('!')[-1])
    b = re.sub(r'\.(xml|txt)$', '', b)
    b = re.sub(r'\.TEI-P5$', '', b)
    b = re.sub(r'(_tei|RAW)$', '', b)
    return alnum_fold(b)


def v1_keys_of(coll: str):
    out = set()
    for sl in COLL_SLUGS.get(coll, ()):
        out |= G['v1_stems'].get(sl, set())
    return out


def v1_check(r, keys):
    ks = v1_keys_of(r['collection'])
    hits = sum(1 for k in keys if k in ks)
    return hits, (hits / len(keys) if keys else 0.0)


def doc_keys(r, members):
    """Keys of the pieces of a document for the v1 match: member stems; REED: the record collection; CKCC: the correspondents."""
    if r['collection'] == 'REED_English':
        return [alnum_fold(r['filename'].split('-ridm')[0])]
    return [stem_key(m) for m in members]


def finish_doc(r, src, fmt, members, parts, out, keys=None):
    """parts: list of extract results (text, tags, ...) for the members of the document."""
    row = base_row(r, src, fmt, members[0] if members else '', 'ok')
    row['n_members'] = len(members)
    hits, share = v1_check(r, keys if keys is not None else doc_keys(r, members))
    row['v1_hits'], row['v1_share'] = hits, share
    row['v1_doc'] = bool(keys if keys is not None else members) and share >= V1_DROP_SHARE
    row['gold'] = L.stem_of(r['filename']) in G['gold']
    texts, tags, off = [], [], 0
    ex = Counter()
    dropped = 0
    for p in parts:
        texts.append(p['text'])
        for t in p['tags']:
            tags.append({**t, 'char_start': t['char_start'] + off, 'char_end': t['char_end'] + off})
        off += len(p['text']) + 2
        ex.update(p.get('ex', {}))
        dropped += sum(p.get('dropped_tags', {}).values())
        row['place_markup'] = row['place_markup'] or bool(p.get('place'))
        row['n_dropcap'] += p.get('n_dropcap', 0)
        for k in ('header', 'front_back'):
            row['ex_' + k] += p.get('file_' + k, 0)
    txt = '\n\n'.join(texts)
    row['n_words'] = len(txt.split())
    row['n_chars'] = len(txt)
    row['kept_nows'] = nonspace(txt)
    for k in ('note_editorial', 'note_other', 'apparatus', 'div_editorial', 'head_editorial', 'standoff', 'diplomatic'):
        row['ex_' + k] = ex.get(k, 0)
    row['dropped_tags'] = dropped
    st = tag_stats(tags)
    row['n_tags'] = len(tags)
    row['n_tags_place'], row['n_tags_pers'], row['n_tags_org'], row['n_tags_other'] = st['place'], st['pers'], st['org'], st['other']
    if row['n_words'] < L.MIN_WORDS:
        row['status'] = 'short'
        return row, None, []
    row['n_tok'] = len(L.WORD.findall(txt))
    for t in tags:
        t['doc_id'] = r['doc_id']
    return row, txt, tags


def pre_status(r, src, fmt, members, keys=None):
    """Decided without reading the text: a language the models do not cover. Returns a row or None."""
    first = members[0] if members else ''
    if not r['lang']:
        return base_row(r, src, fmt, first, 'lang_other')
    return None


def read_tei_member(data, parser, r, surv):
    from lxml import etree
    root = etree.fromstring(data, parser)
    surv_coll = r['collection']
    survey_of(root, surv_coll, surv)
    res = T.extract_tei(root, r['lang'], repair=r['collection'] not in L.NO_DROPCAP_REPAIR)
    if res is None:
        return None
    h, b, fb = file_excl(root)
    res['file_header'], res['file_front_back'] = h, fb
    return res


def html_to_marked(text: str):
    """CKCC letter (html fragment) -> (raw text, spans [(start, end, tag, type, ref)])."""
    import lxml.html
    root = lxml.html.fromstring('<div>' + text + '</div>')
    buf, spans = [], []
    n = [0]

    def put(s):
        if s:
            buf.append(s)
            n[0] += len(s)

    def walk(el):
        tag = el.tag if isinstance(el.tag, str) else None
        if tag is None:
            return
        if tag in ('img', 'figure'):
            return
        if tag in ('div', 'p', 'br') and n[0] and not ''.join(buf[-1:]).endswith('\n'):
            put('\n')
        cls = (el.get('class') or '') if tag == 'span' else ''
        start = n[0]
        put(el.text)
        for c in el:
            walk(c)
            if c.tail:
                put(c.tail)
        if tag in ('p', 'div'):
            put('\n')
        if cls in ('rpername', 'upername'):
            spans.append((start, n[0], 'persName', cls, el.get('title') or ''))
        elif 'place' in cls.lower():
            spans.append((start, n[0], 'placeName', cls, el.get('title') or ''))

    walk(root)
    return ''.join(buf), spans


def process_task(task):
    src, k, K = task
    from lxml import etree
    spec = SOURCES[src]
    out = Path(G['out'])
    name = f'{src}.{k:02d}'
    t0 = time.time()
    rows_all = rows_for(src)
    fmt = 'tei' if spec['fmt'] == 'tei' else 'plain'
    if spec['fmt'] == 'ckcc':
        fmt = 'html'
    docs, texts, alltags = [], [], []
    surv = {'n': Counter(), 'd': Counter(), 't': Counter()}
    by_key = defaultdict(list)
    for r in rows_all:
        r['lang'] = L.RUN_LANGS.get(r['lang_db'], '')
        kk = spec['rkey'](r) if r['filename'] or r['path'] else None
        if kk is not None:
            by_key[kk].append(r)
    found = set()
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True, recover=True)
    path = Path(G['archive']) / f"{spec['tar']}.tar.gz" if spec.get('tar') else None
    if path is not None and not path.exists():
        print('missing tar', path, flush=True)
        return
    groups = defaultdict(list)       # key -> [(member, bytes)]

    def resolve(key):
        """Row key of a member key: equal, or (fuzzy sources: BDPN names are cut in the catalogue) a unique prefix match."""
        if key in by_key:
            return key
        if spec.get('fuzzy'):
            c = {rk for rk in by_key if rk and key and (rk.startswith(key) or key.startswith(rk))}
            if len(c) == 1:
                return c.pop()
        return None

    def deliver(key, members_data):
        for r in by_key.get(key, []):
            if r['doc_id'] in found:
                continue
            found.add(r['doc_id'])
            members = [m for m, _ in members_data]
            pre = pre_status(r, src, fmt, members)
            if pre is not None:
                docs.append(pre)
                continue
            try:
                parts = []
                for m, data in members_data:
                    if spec['fmt'] == 'tei':
                        res = read_tei_member(data, parser, r, surv)
                        if res is None:
                            continue
                        parts.append(res)
                    else:
                        raise ValueError('fmt')
                if not parts:
                    row = base_row(r, src, fmt, members[0], 'no_body')
                    docs.append(row)
                    continue
                row, txt, tags = finish_doc(r, src, fmt, members, parts, out)
            except Exception as e:
                docs.append(base_row(r, src, fmt, members[0] if members else '', 'parse_error', repr(e)[:100]))
                continue
            docs.append(row)
            if txt is not None:
                texts.append((r['doc_id'], r['lang'], txt))
                alltags.extend(tags)

    def deliver_plain(r, member, raw, spans=None, keys=None):
        if r['doc_id'] in found:
            return
        found.add(r['doc_id'])
        if keys is None:
            keys = doc_keys(r, [r['filename'] if '#' in member else member])
        pre = pre_status(r, src, fmt, [member], keys)
        if pre is not None:
            docs.append(pre)
            return
        try:
            res = T.extract_plain(raw, r['lang'], repair=r['collection'] not in L.NO_DROPCAP_REPAIR, spans=spans)
            res['ex'] = {}
            row, txt, tags = finish_doc(r, src, fmt, [member], [res], out, keys)
        except Exception as e:
            docs.append(base_row(r, src, fmt, member, 'parse_error', repr(e)[:100]))
            return
        docs.append(row)
        if txt is not None:
            texts.append((r['doc_id'], r['lang'], txt))
            alltags.extend(tags)

    if spec.get('dir'):                              # a directory of files, grouped by a key read from the content (Casebooks volumes)
        for p_ in sorted(Path(spec['dir']).glob('*.xml'), key=lambda q: int(re.sub(r'\D', '', q.stem) or 0)):
            data = p_.read_bytes()
            key = spec['dkey'](data)
            if key is not None and key in by_key:
                groups[key].append((p_.name, data))
    with open(path, 'rb') if path is not None else open(os.devnull, 'rb') as fh:
        tf = tarfile.open(fileobj=fh, mode='r|gz') if path is not None else iter(())
        for m in tf:
            if not m.isfile():
                continue
            nm = m.name
            if spec['fmt'] == 'jsonl':
                if not nm.endswith('.jsonl'):
                    continue
                for line in tf.extractfile(m).read().decode('utf-8').split('\n'):
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    for r in by_key.get(d.get('filename') or '', []):
                        deliver_plain(r, f'{nm}#{d.get("filename")}', d.get('raw') or '')
                continue
            if spec['fmt'] == 'ckcc':
                if not nm.endswith('letters.jsonl'):
                    continue
                grp = defaultdict(list)
                for line in tf.extractfile(m).read().decode('utf-8').split('\n'):
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    pre = re.match(r'[a-z]+', d.get('pid') or '')
                    lg = CKCC_LANG.get(d.get('language') or '')
                    if pre and lg:
                        grp[f'ckcc_{pre.group(0)}_{lg}'].append(d)
                for key, letters in grp.items():
                    for r in by_key.get(key, []):
                        raws, spans, off = [], [], 0
                        for d in sorted(letters, key=lambda x: (x.get('date') or '', x.get('pid') or '')):
                            t, sp = html_to_marked(d.get('text') or '')
                            spans += [(s + off, e + off, tg, ty, ref) for s, e, tg, ty, ref in sp]
                            raws.append(t)
                            off += len(t) + 2
                        deliver_plain(r, f'ckcc/letters.jsonl#{key}', '\n\n'.join(raws), spans,
                                      keys=sorted({alnum_fold((d.get('pid') or '').split('/')[0]) for d in letters}))
                continue
            key = spec['mkey'](nm)
            if key is None or zlib.crc32(nm.encode()) % K != k:
                continue
            rk = resolve(key)
            if rk is None:
                continue
            key = rk
            data = tf.extractfile(m).read()
            if spec['fmt'] == 'txt':
                raw = L.decode_bytes(data)
                for r in by_key.get(key, []):
                    deliver_plain(r, nm, raw)
                continue
            if spec.get('agg'):
                groups[key].append((nm, data))
            else:
                deliver(key, [(nm, data)])
    for key, md in groups.items():
        deliver(key, sorted(md))
    nf = 0
    for kk, rs in by_key.items():
        for r in rs:
            if r['doc_id'] not in found and K == 1:
                docs.append(base_row(r, src, fmt, '', 'not_found'))
                nf += 1
    # write
    (out / 'docs_parts').mkdir(parents=True, exist_ok=True)
    (out / 'texts').mkdir(parents=True, exist_ok=True)
    (out / 'tags_parts').mkdir(parents=True, exist_ok=True)
    (out / 'survey').mkdir(parents=True, exist_ok=True)
    with open(out / 'docs_parts' / f'{name}.jsonl', 'w', encoding='utf-8') as f:
        for r in docs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with gzip.open(out / 'texts' / f'{name}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as f:
        for did, lg, t in texts:
            f.write(json.dumps({'id': did, 't': t, 'lang': lg}, ensure_ascii=False) + '\n')
    import pandas as pd
    cols = ['doc_id', 'char_start', 'char_end', 'tag', 'type', 'ref', 'cert', 'nested']
    pd.DataFrame(alltags, columns=cols).to_parquet(out / 'tags_parts' / f'{name}.parquet', index=False)
    with open(out / 'survey' / f'{name}.json', 'w') as f:
        json.dump({k_: dict(v) for k_, v in surv.items()}, f)
    st = Counter(r['status'] for r in docs)
    print(f'[{time.strftime("%H:%M:%S")}] {name}: {len(docs)} docs {dict(st)} tags {len(alltags)} {time.time() - t0:.0f}s', flush=True)


def run_pool(tasks, a):
    def size(t):
        sp = SOURCES[t[0]]
        pth = Path(a.archive) / f"{sp['tar']}.tar.gz" if sp.get('tar') else None
        return (pth.stat().st_size / t[2]) if pth is not None and pth.exists() else 0
    tasks = sorted(tasks, key=lambda t: -size(t))
    with mp.get_context('fork').Pool(min(a.procs, len(tasks))) as pool:
        for _ in pool.imap_unordered(process_task, tasks, chunksize=1):
            pass


def cmd_run(a):
    load_globals(a)
    if a.casebooks:
        SOURCES['casebooks']['dir'] = a.casebooks
    first, second = [], []
    for src, spec in SOURCES.items():
        if a.only and src not in a.only:
            continue
        if src == 'casebooks' and not a.casebooks:
            continue
        K = spec['shards'] if not spec.get('agg') else 1
        (second if spec.get('fallback') else first).extend((src, k, K) for k in range(K))
    print(len(first), 'tasks', flush=True)
    G['done'] = set()
    if first:
        run_pool(first, a)
    for p in (Path(a.out) / 'docs_parts').glob('*.jsonl'):
        for line in open(p, encoding='utf-8'):
            d = json.loads(line)
            if d['status'] != 'not_found':
                G['done'].add(d['doc_id'])
    if second:
        print(len(second), 'fallback tasks', flush=True)
        run_pool(second, a)
    print('done', flush=True)


def cmd_merge(a):
    import pandas as pd
    out = Path(a.out)
    rows = []
    for p in sorted((out / 'docs_parts').glob('*.jsonl')):
        for line in open(p, encoding='utf-8'):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # a document found by several tasks (should not happen) keeps its first row; candidates nobody produced a row for are not_found
    df = df.drop_duplicates('doc_id', keep='first')
    c = pd.read_parquet(a.cands)
    miss = c[~c['doc_id'].isin(set(df['doc_id']))]
    extra = []
    for r in miss.to_dict('records'):
        extra.append(base_row({**r, 'lang': L.RUN_LANGS.get(r['lang_db'], '')}, '', '', '', 'not_read', 'no reader for this collection or member not found'))
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df.to_parquet(out / 'docs_read.parquet', index=False)
    tg = [pd.read_parquet(p) for p in sorted((out / 'tags_parts').glob('*.parquet'))]
    tags = pd.concat(tg, ignore_index=True) if tg else pd.DataFrame()
    tags.to_parquet(out / 'tags_read.parquet', index=False)
    n, d, t = Counter(), Counter(), Counter()
    for p in (out / 'survey').glob('*.json'):
        j = json.load(open(p))
        n.update(j['n'])
        d.update(j['d'])
        t.update(j['t'])
    with open(out / 'survey.tsv', 'w') as f:
        f.write('collection\ttag\ttype\tn\tdocs\n')
        for key, v in sorted(n.items()):
            f.write(f'{key}\t\t{v}\t{d[key]}\n')
        for key, v in sorted(t.items()):
            f.write(f'{key}\t{v}\t\n')
    inv = df.groupby(['collection', 'lang_db', 'status']).agg(docs=('doc_id', 'size'), db_words=('db_words', 'sum')).reset_index()
    inv.to_csv(out / 'inventory.tsv', sep='\t', index=False)
    print(len(df), df['status'].value_counts().to_dict(), len(tags), 'tags')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cands')
    c.add_argument('--docs-all', required=True)
    c.add_argument('--db', required=True)
    c.add_argument('--out', required=True)
    r = sub.add_parser('run')
    r.add_argument('--archive', required=True)
    r.add_argument('--cands', required=True)
    r.add_argument('--meta', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--procs', type=int, default=8)
    r.add_argument('--only', nargs='*')
    r.add_argument('--casebooks', default='', help='directory cases/ of casebooks-data-master (read on the Mac, not in the archive)')
    m = sub.add_parser('merge')
    m.add_argument('--cands', required=True)
    m.add_argument('--out', required=True)
    a = ap.parse_args()
    {'cands': cmd_cands, 'run': cmd_run, 'merge': cmd_merge}[a.cmd](a)


if __name__ == '__main__':
    main()
