"""NER layer, extra documents: read the catalogue documents that the first extraction could not read (status unhandled / not_found / parse_error) in the six
languages of the model (en la it es fr de), same cleaning, same output layout as ner2_layer_extract.py.

    python scripts/ner2_layer_extra_extract.py run --archive A --v4 V4 --db DB --inventory docs_all.parquet --out O [--only EMLAP ...] [--probe]

Sources, in this order: the text archive (tarballs, never modified), then the files of the 2023 corpus (``Corpus_v4_2023-04-22``, read only) for the documents that
the archive does not hold (LiberLiber, three BibIT files, three EEBO-TCP files whose archive copy is empty). Every document gets a row in
``docs_parts/extra.jsonl`` with its final status; the texts of the ``ok`` documents go to ``texts/<lang>/extra.<collection>.jsonl.gz``. Duplicate control is a
separate step (ner2_layer_extra_dedup.py); this step only decides what can be read and what is not an early modern text of the place-untagged kind.

Statuses (in addition to those of ner2_layer_extract.py): ``translation`` (modern translation: not an early modern text), ``site_page`` (web page of a site, not a
text), ``place_markup`` (the source marks places: the editors' tagging holds), ``unreadable`` (no readable source; the note says why).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner2_layer_extract as X  # noqa: E402
import ner2_layer_extra_readers as R  # noqa: E402

SIX = {'eng', 'lat', 'ita', 'spa', 'fra', 'deu'}
TARGET_STATUS = {'unhandled', 'not_found', 'parse_error'}
# collections where the drop-capital repair would hurt (single letters are tokens, numerals, abbreviations)
NO_REPAIR = X.NO_DROPCAP_REPAIR | {'ReF_ENHG', 'Bartoli_Correspondence', 'GAMS_BaselRecords', 'PepysDiary', 'GeMi_Midwifery'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------------------------------------------------
def stem(fn: str) -> str:
    return X.stem_of(fn)


def load_targets(db_path: str, inv_path: str, only: set | None, liberliber_map: str = ''):
    """Catalogue rows to read: DB rows of the inventory's unread documents (1450-1700, six languages). LiberLiber is read by path (several folders hold the same
    file name) and only for the files that the catalogue maps to no archive file (the mapped ones were read in the first extraction)."""
    import csv
    import pandas as pd
    inv = pd.read_parquet(inv_path)
    inv = inv[inv['status'].isin(TARGET_STATUS) & inv['lang_db'].isin(SIX)]
    if only:
        inv = inv[inv['collection'].isin(only)]
    want = defaultdict(list)
    for r in inv.itertuples():
        want[r.doc_id].append(r)
    lmap = {}
    if liberliber_map and Path(liberliber_map).exists():
        lmap = {x['filename']: x['status'] for x in csv.DictReader(open(liberliber_map, encoding='utf-8'), delimiter='\t')}
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = []
    for r in db.execute('select collection, filename, path, language, year, date_confidence, wordcount, is_dup from texts where is_dup = 0'):
        did = f"{r['collection']}/{stem(r['filename'])}"
        if did not in want or r['year'] is None or not (X.YEAR_LO <= r['year'] <= X.YEAR_HI) or r['language'] not in SIX:
            continue
        d = dict(r) | {'doc_id': did, 'inv': want[did][0]._asdict()}
        if r['collection'] == 'LiberLiber':
            parent = (r['path'] or '').split('/')
            if lmap and lmap.get(r['filename']) == 'mapped' and want[did][0].status != 'parse_error':
                continue                                               # mapped to an archive file: read (or not) in the first extraction
            if len(parent) >= 3:
                d['doc_id'] = f"LiberLiber/{parent[1]}/{stem(r['filename'])}"
        rows.append(d)
    # inventory ids that no DB row matched (names cut by the catalogue): keep them from the inventory
    got = Counter(r['inv']['doc_id'] for r in rows)
    for did, lst in want.items():
        if lst[0].collection == 'LiberLiber':
            continue
        if got[did] < len(lst):
            for r in lst[got[did]:]:
                d = r._asdict()
                rows.append({'collection': d['collection'], 'filename': did.split('/', 1)[1], 'path': '', 'language': d['lang_db'], 'year': d['year'],
                             'date_confidence': d['date_confidence'], 'wordcount': d['db_words'], 'is_dup': 0, 'doc_id': did, 'inv': d})
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# archive access
# ---------------------------------------------------------------------------------------------------------------------
def scan_tar(path: Path, process, collect=None):
    """Stream a tarball once; ``process(name, read)`` returns None or (key, payload). Payloads are kept in a dict (key -> payload)."""
    out = {}
    with open(path, 'rb') as fh:
        tf = tarfile.open(fileobj=fh, mode='r|gz')
        for m in tf:
            if not m.isfile():
                continue
            ex = tf.extractfile(m)
            res = process(m.name, lambda ex=ex: ex.read())
            if res is not None:
                k, v = res
                if collect is not None:
                    collect(k, v)
                else:
                    out[k] = v
    return out


def match_key(key: str, keys, sizes=None, wc=0, taken=()):
    """Exact key; else, for names cut by the catalogue, the keys that start with it (the closest in length to the catalogue's word count when several)."""
    if key in keys and key not in taken:
        return key
    if len(key) < 40:
        return key if key in keys else None
    c = [k for k in keys if k.startswith(key) and k not in taken]
    if not c:
        return None
    if len(c) == 1 or not sizes or not wc:
        return c[0] if len(c) == 1 else None
    return min(c, key=lambda k: abs(sizes.get(k, 0) - wc))


def tei_doc(data: bytes):
    """-> (text, place_markup, has_body): the diplomatic reader of the first extraction (text of the first body); place markup looked for in every top-level body
    (the first extraction looked in the first body only: a second text of the same file could hold the editors' place tags and be missed)."""
    txt, place, has_body, _ = tei_root_text(R.parse_xml_loose(data))
    return txt, place, has_body


def tei_root_text(root):
    """Text of the first body (the catalogue's reading); place markup looked for in every top-level body. A second body is, in the files seen (CORPUS17), the same
    text again in another form (normalised), so it is not added; its size is returned in the fourth value."""
    bodies = R.top_level_bodies(root)
    if not bodies:
        return '', False, False, 0
    txt, place, _ = X.tei_body_text(root)
    place = place or R.has_place_markup(root)
    extra = 0
    for b in bodies[1:]:
        t, pl, _ = X.tei_body_text(_wrap(b))
        place = place or pl
        extra += len(t.split())
    return txt, place, True, extra


def _wrap(body):
    from lxml import etree
    import copy
    w = etree.Element('wrapper')
    w.append(copy.deepcopy(body))
    return w


def tei_payload(data: bytes):
    txt, place, has_body = tei_doc(data)
    if not has_body:
        return R.Skip('no_body', 'no TEI body')
    if place:
        return R.Skip('place_markup', 'place elements in the TEI body')
    return txt


def safe(fn, *a):
    """Reader result: text, or a Skip, or a Skip('parse_error')."""
    try:
        return fn(*a)
    except R.Skip as e:
        return e
    except Exception as e:                      # a document that cannot be read is recorded, never fatal
        return R.Skip('parse_error', repr(e)[:90])


# ---------------------------------------------------------------------------------------------------------------------
# readers by collection: each returns {key: text | Skip} for the keys the targets ask for
# ---------------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, a):
        self.arch = Path(a.archive)
        self.v4 = Path(a.v4) if a.v4 else None
        self.probe = a.probe


def src_emlap(ctx, rows):
    def proc(name, read):
        m = re.search(r'/emlap_txts/(\d+)\.txt$', name)
        if not m:
            return None
        return f'emlap_{m.group(1)}', R.decode_any(read())
    return scan_tar(ctx.arch / 'raw__emlap.tar.gz', proc)


def src_ref(ctx, rows):
    def proc(name, read):
        if not re.search(r'/ref-[^/]+/[^/]+\.xml$', name):
            return None
        data = read().decode('utf-8', 'replace')
        sig = R.ref_sigle(data[:2500], os.path.basename(name)[:-4])
        return f'ref_{sig}', R.ref_text(data)
    return scan_tar(ctx.arch / 'raw__ref_dl.tar.gz', proc)


def src_tml(ctx, rows):
    def proc(name, read):
        m = re.search(r'/TMLr-main/(\d+(?:th|rd))/([^/]+)\.html$', name)
        if not m or m.group(1) not in ('15th', '16th', '17th'):
            return None
        if R.TML_SITE_PAGES.match(m.group(2)):
            return f'{m.group(1)}_{m.group(2)}', R.Skip('site_page', 'page of the TML site')
        return f'{m.group(1)}_{m.group(2)}', R.tml_text(read().decode('utf-8', 'replace'))
    return scan_tar(ctx.arch / 'raw__tml.tar.gz', proc)


def src_pepys(ctx, rows):
    def proc(name, read):
        if not name.endswith('pepys_4200.txt'):
            return None
        return 'pepys_diary_1660-1669', safe(R.pepys_text, R.decode_any(read()))
    return scan_tar(ctx.arch / 'raw__PepysDiary.tar.gz', proc)


def src_bartoli(ctx, rows):
    by_year = defaultdict(list)

    def proc(name, read):
        if '/md_files_digital_edition/' not in name or not name.endswith('.md'):
            return None
        md = R.decode_any(read())
        y = R.bartoli_docdate_year(md) or 1567                  # same default as the catalogue's harvest
        return name, (y, R.bartoli_md_text(md))
    got = scan_tar(ctx.arch / 'raw__bartoli.tar.gz', proc)
    for name in sorted(got):
        y, t = got[name]
        if 1450 <= y <= 1700:
            by_year[y].append(t)
    return {f'bartoli_{y}': ' '.join(v) for y, v in by_year.items()}


def src_bibles(ctx, rows):
    out = {}

    def proc(name, read):
        m = re.search(r'/en_v2/pid_(\d+)\.txt$', name)
        if m:
            return f'en:{m.group(1)}', R.bible_page_text(R.decode_any(read()), 'kjv')
        m = re.search(r'/luther_v4/([^/]+)\.txt$', name)
        if m:
            return f'luther:{R.fold_name(m.group(1))}', R.bible_page_text(R.decode_any(read()), 'luther')
        return None
    out.update(scan_tar(ctx.arch / 'nas__bibles_ws.tar.gz', proc))

    def proc2(name, read):
        m = re.search(r'/pid_(\d+)\.txt$', name)
        return (f'it:{m.group(1)}', safe(lambda: R.diodati_check(R.bible_page_text(R.decode_any(read()), 'diodati')))) if m else None
    out.update(scan_tar(ctx.arch / 'nas__it_bibles_v2.tar.gz', proc2))
    return out


def ws_key(stem_: str, lang: str):
    """Wikisource file names of the catalogue: 'WS_it_' prefix, cut at 80 characters (the cut can leave '.tx')."""
    s = re.sub(rf'^WS_{lang}_', '', stem_)
    return re.sub(r'\.t(x(t)?)?$', '', s)


def bible_key(stem_: str):
    m = re.search(r'_pid(\d+)_', stem_)
    if m:
        return ('it:' if '_it_' in stem_[:8] else 'en:') + m.group(1)
    m = re.match(r'WS_de_LutherNT1522_(.+)$', stem_)
    if m:
        return 'luther:' + R.fold_name(m.group(1))
    return None


def src_tei_tar(tarname, pattern, keyfn):
    def f(ctx, rows):
        def proc(name, read):
            m = re.search(pattern, name)
            if not m:
                return None
            k = keyfn(m)
            return k, safe(tei_payload, read())
        return scan_tar(ctx.arch / tarname, proc)
    return f


def src_emothe(ctx, rows):
    def proc(name, read):
        m = re.search(r'/(EMOTHE\d{4})_[a-z]{2}\.xml$', name)
        if not m:
            return None
        data = read()
        try:
            root = R.parse_xml_loose(data)
            info = R.emothe_info(root)
            keep, why = R.emothe_decision(info)
            if not keep:
                return m.group(1), R.Skip('translation', why)
            txt, place, has_body, _ = tei_root_text(root)
            if not has_body:
                return m.group(1), R.Skip('no_body', 'no TEI body')
            if place:
                return m.group(1), R.Skip('place_markup', 'place elements in the TEI body')
            return m.group(1), (txt, why)
        except R.Skip as e:
            return m.group(1), e
        except Exception as e:
            return m.group(1), R.Skip('parse_error', repr(e)[:90])
    out = scan_tar(ctx.arch / 'nas__emothe.tar.gz', proc)
    return {k: (v[0] if isinstance(v, tuple) else v) for k, v in out.items()} | {k + '#note': v[1] for k, v in out.items() if isinstance(v, tuple)}


def src_plain_tar(tarname, pattern, keyfn, decode=R.decode_any):
    def f(ctx, rows):
        def proc(name, read):
            m = re.search(pattern, name)
            if not m:
                return None
            return keyfn(m), decode(read())
        return scan_tar(ctx.arch / tarname, proc)
    return f


def src_zip_free(ctx, rows):
    return {}


# Pollux (2023 corpus) readers: key = path of the catalogue
def v4_text(ctx, rel: str):
    p = ctx.v4 / rel                                   # file names of the 2023 corpus can hold a literal backslash (CorpusCorp/latin_xmls\\12135.xml.txt)
    if not p.exists():
        p = ctx.v4 / rel.replace('\\', '/')
    if not p.exists():
        return R.Skip('unreadable', f'not in the 2023 corpus: {rel}')
    return R.decode_any(p.read_bytes())


SOURCES = {
    # collection -> (function giving {key: text}, function giving the key from the row)
    'EMLAP': (src_emlap, lambda r: stem(r['filename'])),
    'ReF_ENHG': (src_ref, lambda r: stem(r['filename'])),
    'TML': (src_tml, lambda r: stem(r['filename'])),
    'PepysDiary': (src_pepys, lambda r: stem(r['filename'])),
    'Bartoli_Correspondence': (src_bartoli, lambda r: stem(r['filename'])),
    'WikisourceBibles_Renaissance': (src_bibles, lambda r: bible_key(stem(r['filename']))),
    'WikisourceBibles_Italian': (src_bibles, lambda r: bible_key(stem(r['filename']))),
    'GAMS_BaselRecords': (src_tei_tar('raw__gams.tar.gz', r'/(o_[a-z]+\.\d+)\.xml$', lambda m: 'gams_' + m.group(1)), lambda r: stem(r['filename'])),
    'EMOTHE': (src_emothe, lambda r: (re.search(r'EMOTHE\d{4}', r['filename']) or [None])[0]),
    'Nakala': (src_tei_tar('nas__nakala_renaissance.tar.gz', r'/(?:nakala_)?(10\.34847_nkl\.[a-z0-9]+)\.xml$', lambda m: 'nakala_' + m.group(1)),
               lambda r: stem(r['filename'])),
    'BibIT': (src_tei_tar('raw__bibit_tei.tar.gz', r'/(bibit\d+)\.xml$', lambda m: m.group(1)), lambda r: stem(r['filename'])),
    'CorpusCorp': (src_tei_tar('raw__corpuscorp.tar.gz', r'/xml/(\d+)\.xml$', lambda m: m.group(1)), lambda r: stem(r['filename'])),
    'CORPUS17_French': (src_tei_tar('raw__corpus17.tar.gz', r'/data/(EDI_\d+)\.xml$', lambda m: 'corpus17_' + m.group(1)), lambda r: stem(r['filename'])),
    'DiBiLit_German': (src_tei_tar('raw__dibilit.tar.gz', r'(?:^|/)dibilit/data/[^/]+/([^/]+)\.txt\.xml$', lambda m: 'dibilit_' + m.group(1)), lambda r: stem(r['filename'])),
    'DiBiPhil_Renaissance': (src_tei_tar('raw__dibiphil.tar.gz', r'(?:^|/)dibiphil/data/([^/]+)\.xml$', lambda m: 'dibiphil_' + m.group(1)), lambda r: stem(r['filename'])),
    'Translatoscope_NeoLat': (src_tei_tar('raw__translatoscope.tar.gz', r'/XML/([^/]+)/\1__([^/]+)\.xml$', lambda m: f'translatoscope_{m.group(1)}_{m.group(2)}'),
                              lambda r: stem(r['filename'])),
    'TBSO_GH': (src_plain_tar('raw__tbso.tar.gz', r'/txt/([^/]+)\.txt$', lambda m: 'tbso_' + m.group(1)), lambda r: stem(r['filename'])),
    'GeMi_Midwifery': (src_plain_tar('raw__gemi.tar.gz', r'/RAW Files/Main Text/([^/]+)\.txt$', lambda m: m.group(1)), lambda r: stem(r['filename'])),
    'Estilometria_SigloOro': (src_plain_tar('raw__estilometria.tar.gz', r'/Siglo_de_Oro/(?:[^/]+/)*([^/]+?)(?:\.txt)?$', lambda m: 'estilom_' + m.group(1)),
                              lambda r: stem(r['filename'])),
    'WikisourceIT_Renaissance': (src_plain_tar('nas__ws_it_renaissance.tar.gz', r'/([^/]+)\.txt$', lambda m: m.group(1), decode=lambda b: R.wikisource_plain(R.decode_any(b))), lambda r: ws_key(stem(r['filename']), 'it')),
    'WikisourceES_Renaissance': (src_plain_tar('nas__ws_es_renaissance.tar.gz', r'/([^/]+)\.txt$', lambda m: m.group(1), decode=lambda b: R.wikisource_plain(R.decode_any(b))), lambda r: ws_key(stem(r['filename']), 'es')),
}
# collections read from the 2023 corpus by catalogue path
V4_COLLECTIONS = {'LiberLiber', 'EEBOphase2'}
# documents of the above that are not in the 2023 corpus under the catalogue path, or whose source is elsewhere:
VESPASIANO = 'VespasianoLetters'
NOT_READABLE = {
    'GitHubTEI': 'the XML files were fetched from GitHub by the scan and not archived (the archive holds only the lists gh_repos / gh_trees / gh_candidates); the '
                 'catalogue years of this scan are often birth years of the authors',
}


# ---------------------------------------------------------------------------------------------------------------------
def make_row(r, status, task='extra', n_words=0, n_tok=0, n_chars=0, n_rep=0, note='', member='', atts=None):
    inv = r['inv']
    att = (atts or {}).get(r['collection']) or {}
    return {'doc_id': r['doc_id'], 'collection': r['collection'], 'member': member, 'task': task, 'lang_db': r['language'],
            'lang': X.RUN_LANGS.get(r['language'], ''), 'year': r['year'], 'date_confidence': r['date_confidence'], 'tier': att.get('tier', ''),
            'licence_tag': att.get('license_tag', ''), 'db_words': r['wordcount'], 'n_words': n_words, 'n_tok': n_tok, 'n_chars': n_chars, 'n_dropcap': n_rep,
            'status': status, 'note': note, 'prev_status': inv.get('status', '')}


def finish_text(r, txt, member, note, atts, texts, rows_out, samples):
    txt = X.clean_text(txt)
    lg = X.RUN_LANGS[r['language']]
    n_rep = 0
    if r['collection'] not in NO_REPAIR:
        samp = []
        txt, n_rep = X.repair_dropcaps(txt, lg, X.doc_vocab(txt), samp)
        samples.extend((r['collection'], r['doc_id'], x) for x in samp[:3])
    n_words = len(txt.split())
    if n_words < X.MIN_WORDS:
        rows_out.append(make_row(r, 'short', n_words=n_words, note=note, member=member, atts=atts))
        return
    rows_out.append(make_row(r, 'ok', n_words=n_words, n_tok=len(X.WORD.findall(txt)), n_chars=len(txt), n_rep=n_rep, note=note, member=member, atts=atts))
    texts.setdefault(lg, []).append((rows_out[-1]['doc_id'], txt))


def cmd_run(a):
    ctx = Ctx(a)
    atts = X.load_attributions(str(ctx.arch / 'ATTRIBUTIONS.tsv'))
    rows = load_targets(a.db, a.inventory, set(a.only) if a.only else None, a.liberliber_map)
    log(len(rows), 'documents to read')
    out = Path(a.out)
    texts, rows_out, samples = {}, [], []
    by_coll = defaultdict(list)
    for r in rows:
        by_coll[r['collection']].append(r)
    # LiberLiber: only the documents that the catalogue maps to no archive file (the others were read in the first extraction)
    for coll, rs in sorted(by_coll.items()):
        t0 = time.time()
        if coll in NOT_READABLE:
            for r in rs:
                rows_out.append(make_row(r, 'unreadable', note=NOT_READABLE[coll], atts=atts))
            log(coll, len(rs), 'not readable:', NOT_READABLE[coll][:60])
            continue
        if coll == VESPASIANO:
            for r in rs:
                rows_out.append(make_row(r, 'place_markup', note='HTML letters with class="Place" spans in the text and the metadata (editors\' tagging, HTML not TEI)', atts=atts))
            continue
        if coll in V4_COLLECTIONS:
            for r in rs:
                if coll == 'LiberLiber':
                    rel = r['path'] or ''
                    if not rel:
                        rows_out.append(make_row(r, 'unreadable', note='no catalogue path', atts=atts))
                        continue
                else:
                    rel = f"EEBOphase2/{stem(r['filename'])}.P5.txt"
                res = v4_text(ctx, rel)
                if isinstance(res, R.Skip):
                    rows_out.append(make_row(r, res.status, note=res.why, atts=atts))
                else:
                    finish_text(r, res, 'v4:' + rel, '2023 corpus', atts, texts, rows_out, samples)
            log(coll, len(rs), 'from the 2023 corpus', f'{time.time() - t0:.0f}s')
            continue
        if coll not in SOURCES:
            for r in rs:
                rows_out.append(make_row(r, 'unreadable', note='no reader for this collection', atts=atts))
            log(coll, len(rs), 'no reader')
            continue
        fn, keyfn = SOURCES[coll]
        got = fn(ctx, rs)
        keys = {k for k in got if not k.endswith('#note')}
        sizes = {k: (len(v.split()) if isinstance(v, str) else 0) for k, v in got.items() if not k.endswith('#note')}
        used = Counter()
        for r in rs:
            k = keyfn(r)
            kk = match_key(k, keys, sizes, r['wordcount'] or 0, taken={x for x, n in used.items() if n} if len(k or '') >= 40 else ()) if k else None
            if kk is None:
                # not in the archive: documents of these collections may be in the 2023 corpus under the catalogue path
                alt = None
                if ctx.v4 and r['path']:
                    alt = v4_text(ctx, r['path'])
                if alt is not None and not isinstance(alt, R.Skip):
                    finish_text(r, alt, 'v4:' + r['path'], '2023 corpus (not in the archive)', atts, texts, rows_out, samples)
                else:
                    rows_out.append(make_row(r, 'unreadable', note=f'no source for key {k!r} in the archive' + ('' if not (ctx.v4 and r['path']) else ' nor in the 2023 corpus'),
                                             atts=atts))
                continue
            used[kk] += 1
            v = got[kk]
            note = got.get(kk + '#note', '')
            if isinstance(v, R.Skip):
                rows_out.append(make_row(r, v.status, note=v.why, member=kk, atts=atts))
            else:
                finish_text(r, v, kk, note, atts, texts, rows_out, samples)
        multi = [k for k, n in used.items() if n > 1]
        log(coll, len(rs), 'docs;', Counter(x['status'] for x in rows_out if x['collection'] == coll), 'keys used twice:', multi[:3], f'{time.time() - t0:.0f}s')
    write_outputs(out, texts, rows_out, samples)


def write_outputs(out: Path, texts, rows_out, samples):
    for lang, items in texts.items():
        d = out / 'texts' / lang
        d.mkdir(parents=True, exist_ok=True)
        by = defaultdict(list)
        for did, t in items:
            by[did.split('/', 1)[0]].append((did, t))
        for coll, its in by.items():
            with gzip.open(d / f'extra.{coll}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as f:
                for did, t in its:
                    f.write(json.dumps({'id': did, 't': t}, ensure_ascii=False) + '\n')
    (out / 'docs_parts').mkdir(parents=True, exist_ok=True)
    with open(out / 'docs_parts' / 'extra.jsonl', 'w', encoding='utf-8') as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    (out / 'repair_samples').mkdir(parents=True, exist_ok=True)
    with open(out / 'repair_samples' / 'extra.tsv', 'w', encoding='utf-8') as f:
        for c_, d_, x_ in samples:
            f.write(f'{c_}\t{d_}\t{x_}\n')
    log('written', len(rows_out), 'rows;', dict(Counter(r['status'] for r in rows_out)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--archive', required=True)
    r.add_argument('--v4', default='')
    r.add_argument('--db', required=True)
    r.add_argument('--inventory', required=True, help='docs_all.parquet of the first extraction')
    r.add_argument('--out', required=True)
    r.add_argument('--only', nargs='*')
    r.add_argument('--liberliber-map', default='', help='rebuild/liberliber_source_mapping_2026-06-13.tsv')
    r.add_argument('--probe', action='store_true')
    a = ap.parse_args()
    {'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    main()
