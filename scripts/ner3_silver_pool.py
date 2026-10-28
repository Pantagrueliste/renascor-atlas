"""Silver passage pool for the LLM teacher (Spanish, Italian, Latin), built from the local text archive (read-only).

Stages (all deterministic, seed 20261024). Only the stdlib, numpy, pandas and the repo modules are needed.

    excl        8-gram hashes (64 bit) and document names of every text that must stay out: gold v2 (all files), gold seed, dev3, test_v2.
    salamanca   Salamanca TEI re-read with the dev3 reader (expansions kept, not the abbreviations).
    inv         document table: extraction rows (ner2_layer_extract.py / ner2_layer_extra_extract.py) + catalogue columns, exclusions by name.
    lex         Latin lexicon of clean texts (OOV filter for OCR collections, as in dev3).
    cut         per text shard: document 8-gram check against the excluded texts, then up to min(30, words/260) passages per document, one per chunk
                of the document (spread over the whole text), 120-200 words, sentence start to sentence end (ner2_passages.draw_passage, verse refused).
    select      passage dedup, caps (document 30, collection 25 %, 25-year bin, genre), tokenisation as in gold v2, output pool_{es,it,la}.jsonl.

Nothing is written to the archive. Temporary files live in --work.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from ratlas.ner.ner2_passages import ALPHA, SENT, fold, passage_ok, tokenize  # noqa: E402

SEED = 20261024
RQ = Path('/Users/clem/Projects/Papers/RQ')
NER = RQ / 'derived/atlas/ner'
LANGS = ('es', 'it', 'la')
DB_LANG = {'spa': 'es', 'ita': 'it', 'lat': 'la'}
DOC_CAP = 30
COLL_SHARE = 0.25
BIN_SHARE = 0.35                      # share cap of a 50-year bin (1450-99, 1500-49, 1550-99, 1600-49, 1650-1700)
GENRE_SHARE = 0.45
VERSE_SHARE = 0.20                    # verse and drama (speaker labels) together
TARGET = 60000
MIN_WORDS_PER_PASSAGE_SLOT = 260

# collections left out, with the reason (written to the README)
# editions of the first NER round (data_v1 train / dev): kept in the pool, flagged; the v1 TEST editions are not in the pool
V1_TRAIN_DEV_EDITIONS = {'BullingerLetters', 'CAMENA', 'CAMENA_CERA_Letters', 'CERA', 'CKCC_Epistolarium', 'CroALa', 'Cavriana_Correspondence', 'Suriano_Correspondence',
                         'BDPN_Novohispano', 'PostScriptum'}
EXCLUDED_COLLECTIONS = {
    'IMPACTes_OriginalImprint': 'HTR ground truth that loses the "st" and "ss" ligatures (broken words); dev3 left it out for the same reason',
    'Translatoscope_NeoLat': 'modern translations (flag "translation" in the catalogue)',
    'WikisourceBibles_Italian': 'Bible text with verse numbers; the same text recurs in several editions',
    'WikisourceBibles_Renaissance': 'Bible text with verse numbers; the same text recurs in several editions',
    'PostScriptum': 'the TEI reading doubles words where the edition gives both the original and the regularised form (agoraagora, esteaestea) and splits others (ver nos); '
                    'not repaired here, so the collection is left out',
}

# default genre per collection (coarse; refined by title words when the file name has any)
GENRE_DEFAULT = {
    'NOSCEMUS': 'science_medicine', 'EMLAP': 'science_medicine', 'Newton_Project': 'science_medicine', 'TML': 'science_medicine',
    'CorpusCorp': 'treatise_essay', 'LatinEEBO': 'sermon_theology', 'EEBOphase2': 'treatise_essay', 'LatinWikisource': 'treatise_essay',
    'Salamanca': 'law_politics', 'Nakala': 'treatise_essay',
    'Estilometria_SigloOro': 'fiction_satire', 'OSTA': 'documentary_mixed', 'SpanishGoldenAgeProse_WS': 'fiction_satire',
    'WikisourceES_Renaissance': 'fiction_satire', 'WikisourceES': 'other_prose', 'EMOTHE': 'fiction_satire', 'TBSO_GH': 'fiction_satire',
    'BibIT': 'treatise_essay', 'LiberLiber': 'history_chronicle', 'WikisourceIT_Renaissance': 'treatise_essay', 'WikisourceIT': 'other_prose',
    'Galileo_Opere_Favaro': 'science_medicine', 'Bartoli_Correspondence': 'letters',
    'CAMENA': 'verse_poetry', 'CERA': 'letters', 'CAMENA_CERA_Letters': 'letters', 'BullingerLetters': 'letters', 'CKCC_Epistolarium': 'letters', 'CroALa': 'verse_poetry',
    'PostScriptum': 'letters', 'Suriano_Correspondence': 'diplomatic_report', 'Cavriana_Correspondence': 'letters', 'BDPN_Novohispano': 'documentary_mixed',
}
GENRE_RULES = [
    ('letters', r'\b(lettere?|lettera|carta|cartas|epistol\w*|epistula\w*|briefe?|letters?|correspond\w*|carteggio)\b'),
    ('sermon_theology', r'\b(serm\w+|concio\w*|homil\w+|predic\w+|theolog\w+|teolog\w+|sacrament\w+|confess\w+|penitent\w+|biblia|psalm\w+|catechis\w+|ecclesi\w+|dogma\w*|doctrina|sancti|santa|santo)\b'),
    ('law_politics', r'\b(iur\w+|iust\w+|lex|leges|legum|ius|derecho|contrat\w+|cambios|usur\w+|politic\w+|principe|principi|discorsi|republic\w+|imperio|gobiern\w+|decret\w+)\b'),
    ('history_chronicle', r'\b(histor\w+|cr[oó]nic\w+|annal\w+|diari\w*|diarii|comment\w+|rerum|gesta\w*|vit[ae]|vida|vite|relaci[oó]n\w*|relazione|relatione|viaje|viaggio\w*|navigat\w+|storia|istor\w+)\b'),
    ('science_medicine', r'\b(medic\w+|anatom\w+|plant\w+|herb\w+|phys\w+|fisic\w+|astronom\w+|mathemat\w+|geometr\w+|chym\w+|chem\w+|botan\w+|animal\w+|corpor\w+|morb\w+|pest\w+|agricoltura|sidereus|cometa\w*|ephemerid\w+|magia|mechanic\w+|scienz\w+|arith\w+)\b'),
    ('oration_dialogue', r'\b(orati\w+|oraci\w+|orazion\w+|discurso\w*|discorso\w*|dialog\w+|colloqui\w+|coloquio\w*|disputat\w+|declamat\w+|panegyr\w+)\b'),
    ('fiction_satire', r'\b(novel\w+|cuento\w*|novella\w*|sue[nñ]os|satir\w+|facet\w+|lazarillo|persiles|quijote|guzm[aá]n)\b'),
    ('treatise_essay', r'\b(trattato|tratado|traite|galateo|arte|ars|institut\w+|elementa)\b'),
]
OCR_COLLECTIONS = {'NOSCEMUS': 0.10, 'EMLAP': 0.10}        # Latin OOV ceiling per OCR collection (other Latin: 0.14)
TRIM_COLLECTIONS = {'NOSCEMUS', 'EMLAP'}                  # full-book OCR: front matter (4 %) and back matter (8 %) cut, as in dev3
MIN_CAPS = {'OSTA': 2}                                   # proper names are written in lower case in this paleographic transcription
# symbols that are editorial or OCR marks, not text (section and pilcrow signs are real text and stay)
STRAY = re.compile(r'[@`^~*#$%+_=\u00a4\u25a0\u25a1]')
EOL_EQ = re.compile(r'(?<=[A-Za-z\u00c0-\u00ff])=\s+(?=[a-z\u00e0-\u00ff])')           # end-of-line hyphen written "=" (BDPN, Bartoli)
OCR_JUNK = re.compile(r'Digitized by|Google|§\s*\d+\.\s*\d+')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def stem_of(name: str) -> str:
    b = name.replace('\\', '/').split('/')[-1]
    b = re.sub(r'\.(xml|txt|json|jsonl|odt|epub|zip|html)$', '', b, flags=re.I)
    b = re.sub(r'\.(xml|txt)$', '', b, flags=re.I)
    return re.sub(r'\.(P5|TEI-P5)$', '', b)


def norm(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', fold(s))


# ----------------------------------------------------------------------------------------------- 8-gram hashing (64 bit, deterministic)
_PK = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x94D049BB133111EB,
                0xBF58476D1CE4E5B9], dtype=np.uint64) | np.uint64(1)
_VOC: dict = {}
_WORD = re.compile(r'\w+', re.U)


def _wid(w: str) -> int:
    v = _VOC.get(w)
    if v is None:
        if len(_VOC) > 2_500_000:
            _VOC.clear()
        v = int.from_bytes(hashlib.blake2b(fold(w).encode(), digest_size=8).digest(), 'little')
        _VOC[w] = v
    return v


def gram_hashes(text: str, n: int = 8) -> np.ndarray:
    ws = _WORD.findall(text)
    m = len(ws)
    if m < n:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((_wid(w) for w in ws), dtype=np.uint64, count=m)
    h = np.zeros(m - n + 1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        for k in range(n):
            h += ids[k:m - n + 1 + k] * _PK[k]
    return h


def in_sorted(h: np.ndarray, E: np.ndarray) -> np.ndarray:
    if len(h) == 0 or len(E) == 0:
        return np.zeros(len(h), dtype=bool)
    i = np.searchsorted(E, h)
    i[i >= len(E)] = 0
    return E[i] == h


# ----------------------------------------------------------------------------------------------- stage excl
def iter_json(path: Path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def cmd_excl(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sources = {
        'gold_v2': [p for p in sorted((NER / 'v2/gold2').glob('*.jsonl'))] + [NER / 'gold_seed_v1.jsonl'],
        'dev3': [NER / 'v2/renner/dev3/passages.jsonl', NER / 'v2/renner/dev3/gold_dev3.jsonl'] + sorted((NER / 'v2/renner/dev3/annotation').glob('*.jsonl')),
        'test_v2': [NER / 'v2/test2/test_v2.jsonl'] + sorted((NER / 'v2/test2').glob('cand_*.jsonl')) + sorted((NER / 'v2/test2').glob('judged_*.jsonl')),
    }
    docs = []
    summ = {}
    for src, files in sources.items():
        grams = []
        seen_text = set()
        nfiles = nrec = 0
        for p in files:
            if not p.exists():
                continue
            nfiles += 1
            for d in iter_json(p):
                t = d.get('text') or d.get('passage_text')
                if isinstance(t, str) and t and t not in seen_text:
                    seen_text.add(t)
                    grams.append(gram_hashes(t))
                    nrec += 1
                if d.get('doc'):
                    docs.append({'source': src, 'collection': d.get('collection') or d.get('edition') or '', 'doc': str(d['doc']), 'lang': d.get('lang') or ''})
        arr = np.unique(np.concatenate(grams)) if grams else np.empty(0, dtype=np.uint64)
        np.save(work / f'excl_grams_{src}.npy', arr)
        summ[src] = {'files': nfiles, 'distinct_texts': nrec, 'distinct_8grams': int(len(arr))}
        log(src, summ[src])
    # document names
    ud = {}
    for d in docs:
        ud[(d['source'], d['collection'], d['doc'])] = d
    json.dump(list(ud.values()), open(work / 'excl_docs.json', 'w'), ensure_ascii=False)
    summ['n_doc_names'] = len(ud)
    json.dump(summ, open(work / 'excl_summary.json', 'w'), indent=1)


def load_excl_grams(work: Path):
    parts = {s: np.load(work / f'excl_grams_{s}.npy') for s in ('gold_v2', 'dev3', 'test_v2')}
    allg = np.unique(np.concatenate(list(parts.values())))
    return parts, allg


# ----------------------------------------------------------------------------------------------- stage salamanca
def cmd_salamanca(a):
    """Re-read Salamanca with the dev3 reader (expanded forms). Writes texts/<lang>/salamanca_dev3reader.jsonl.gz and docs_parts/salamanca_dev3reader.jsonl."""
    sys.path.insert(0, str(NER / 'v2/renner/dev3/scripts'))
    import ner3_dev3_build as D  # noqa: E402
    out = Path(a.extract)
    la, es = D.load_salamanca(Path(a.root))
    rows = []
    for lang, docs in (('la', la), ('es', es)):
        d_ = out / 'texts' / lang
        d_.mkdir(parents=True, exist_ok=True)
        with gzip.open(d_ / 'salamanca_dev3reader.jsonl.gz', 'wt', encoding='utf-8') as f:
            for d in docs:
                did = f'Salamanca/{d.doc}'
                f.write(json.dumps({'id': did, 't': d.text}, ensure_ascii=False) + '\n')
                rows.append({'doc_id': did, 'collection': 'Salamanca', 'member': d.doc, 'task': 'salamanca_dev3reader', 'lang_db': {'la': 'lat', 'es': 'spa'}[lang], 'lang': lang, 'year': d.year,
                             'date_confidence': 'tei_source_date', 'tier': 'T1', 'licence_tag': 'CC-BY-4.0', 'db_words': 0, 'n_words': len(d.text.split()), 'n_tok': 0, 'n_chars': len(d.text),
                             'n_dropcap': 0, 'status': 'ok', 'note': 'dev3 reader', 'title': d.title, 'author': d.author, 'date': d.date})
    (out / 'docs_parts').mkdir(exist_ok=True)
    with open(out / 'docs_parts' / 'salamanca_dev3reader.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    log('salamanca', len(rows), 'documents', Counter(r['lang'] for r in rows))


# ----------------------------------------------------------------------------------------------- stage inv
def precision_of(conf: str) -> str:
    c = (conf or '').lower()
    if c in ('exact', 'stated') or 'correspaction' in c or 'creation_date' in c:
        return 'exact'
    if 'inferred' in c and c not in ('inferred', 'ws_inferred', 'inferred_per_work'):
        return 'work_inferred'
    if 'query' in c:
        return 'author'
    if c == 'corpuscorp_year_header':
        return 'edition'
    if c in ('edition_date', 'tei_source_date', 'tei_header', 'jsonl', 'collection'):
        return 'edition'
    if c in ('inferred', 'ws_inferred', 'wikisource_es_curated', 'wikisource_it_curated', 'inferred_per_work'):
        return 'work_inferred'
    if 'author' in c or 'lifespan' in c or 'lookup' in c:
        return 'author'
    if 'period' in c or 'century' in c:
        return 'century'
    if c in ('hsms_opdt',):
        return 'edition'
    return c or 'unknown'


def guess_genre(coll: str, stem: str, title: str = '') -> tuple[str, str]:
    t = re.sub(r'[_\-]+', ' ', f'{title} {stem}').lower()
    if coll not in ('NOSCEMUS', 'OSTA', 'EMLAP'):
        for g, pat in GENRE_RULES:
            if re.search(pat, t):
                return g, 'title_heuristic'
    return GENRE_DEFAULT.get(coll, 'other_prose'), 'collection_default'


def cmd_inv(a):
    import pandas as pd
    import sqlite3
    work = Path(a.work)
    rows = []
    for ex in a.extract:
        for p in sorted((Path(ex) / 'docs_parts').glob('*.jsonl')):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                r['_extract'] = str(ex)
                if r.get('task') == 'extra':
                    r['task'] = 'extra.' + r['collection']
                rows.append(r)
    df = pd.DataFrame(rows)
    df['v1_seen'] = False
    if a.tei:
        t = pd.read_parquet(Path(a.tei) / 'docs_read.parquet')
        t = t[(t['status'] == 'ok') & t['lang'].isin(LANGS) & ~t['gold'].astype(bool)].copy()
        t['task'] = None
        t['v1_seen'] = t['v1_doc'].astype(bool)
        t['_extract'] = str(a.tei)
        df = pd.concat([df, t[[c for c in t.columns if c in df.columns]]], ignore_index=True)
        log('TEI-edition rows added', len(t), t.groupby(['collection', 'lang']).size().to_dict())
    # Salamanca: the dev3 reader replaces the first reading
    sal_new = df[df['task'] == 'salamanca_dev3reader']
    if len(sal_new):
        df = df[~((df['collection'] == 'Salamanca') & (df['task'] != 'salamanca_dev3reader'))]
    df = df.drop_duplicates('doc_id', keep='last')
    log('rows read', len(df), df['status'].value_counts().to_dict())
    status_all = df['status'].value_counts().to_dict()
    # catalogue columns
    db = sqlite3.connect(a.db)
    cat = {}
    for r in db.execute('select collection, filename, edition_type, transmission_medium, origin, date_confidence, year, wordcount, transmission_evidence from texts where is_dup=0'):
        cat.setdefault(f'{r[0]}/{stem_of(r[1])}', r)
    ed, tm, og, mo = [], [], [], []
    for did in df['doc_id']:
        r = cat.get(did)
        ed.append(r[2] if r else '')
        tm.append(r[3] if r else '')
        og.append(r[4] if r else '')
        mo.append(bool(r and re.search(r'modern', r[8] or '', re.I)))
    df['edition_type'], df['transmission_medium'], df['origin'], df['modern_transcription_flag'] = ed, tm, og, mo
    df = df[(df['status'] == 'ok') & (df['lang'].isin(LANGS))].copy()
    reasons = Counter()
    keep = []
    exd = json.load(open(work / 'excl_docs.json'))
    by_lang_norms = defaultdict(list)
    for d in exd:
        n = norm(d['doc'])
        if len(n) >= 6:
            by_lang_norms[d['lang'] or '*'].append((n, d))
    flat_norms = [x for v in by_lang_norms.values() for x in v]
    exact_norm = defaultdict(list)
    for n, d in flat_norms:
        exact_norm[n].append(d)
    name_hits = {}
    for r in df.itertuples():
        stem = r.doc_id.split('/', 1)[1]
        coll = r.collection
        why = None
        if coll in EXCLUDED_COLLECTIONS:
            why = 'collection:' + coll
        elif re.search(r'translat(ion|ed)\b', (r.origin or ''), re.I) and coll not in ('NOSCEMUS',):
            why = 'translation_flag'
        if why is None:
            ns = norm(stem)
            hit = None
            if ns in exact_norm:
                hit = exact_norm[ns][0]
            elif len(ns) >= 8:
                for n, d in flat_norms:
                    if len(n) >= 8 and (n in ns or ns in n):
                        hit = d
                        break
            if hit:
                why = f"doc_name:{hit['source']}"
                name_hits[r.doc_id] = hit
        if why:
            reasons[why] += 1
            continue
        keep.append(r.Index)
    out = df.loc[keep].copy()
    out['v1_edition_collection'] = out['collection'].isin(V1_TRAIN_DEV_EDITIONS)
    out['stem'] = out['doc_id'].str.split('/', n=1).str[1]
    out['date_precision'] = [precision_of(c) for c in out['date_confidence']]
    gs = [guess_genre(c, s, (t if isinstance(t, str) else '')) for c, s, t in zip(out['collection'], out['stem'], out.get('title', [''] * len(out)))]
    out['genre'] = [g for g, _ in gs]
    out['genre_source'] = [s for _, s in gs]
    out.to_parquet(work / 'docs.parquet', index=False)
    json.dump({'status_all_rows': status_all, 'excluded_at_inv': dict(reasons), 'name_hits': {k: v for k, v in list(name_hits.items())}, 'kept': int(len(out)),
               'kept_by_lang': out['lang'].value_counts().to_dict()}, open(work / 'inv_summary.json', 'w'), indent=1, ensure_ascii=False)
    log('docs kept', len(out), dict(reasons))
    print(out.groupby(['lang', 'collection']).agg(docs=('doc_id', 'size'), words=('n_words', 'sum')).to_string())


# ----------------------------------------------------------------------------------------------- stage lex
def read_shard(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            yield d['id'], d['t']


def cmd_lex(a):
    import pandas as pd
    work = Path(a.work)
    df = pd.read_parquet(work / 'docs.parquet')
    clean = df[(df['lang'] == 'la') & (df['collection'].isin(['LatinEEBO', 'LatinWikisource', 'Salamanca']))]
    want = set(clean['doc_id'])
    lex = Counter()
    for ex in a.extract:
        for p in sorted((Path(ex) / 'texts' / 'la').glob('*.jsonl.gz')):
            for did, t in read_shard(p):
                if did in want:
                    lex.update(fold(w) for w in re.findall(rf'[{ALPHA}]+', t[:500000]))
    import pickle
    pickle.dump(lex, open(work / 'lex_la.pkl', 'wb'))
    log('lexicon', len(lex), 'types from', len(want), 'documents')


# ----------------------------------------------------------------------------------------------- stage cut
G: dict = {}


def oov_rate(seg, lex):
    ws = [fold(w) for w in re.findall(rf'[{ALPHA}]+', seg)]
    ws = [w for w in ws if len(w) >= 4]
    return sum(lex[w] < 2 for w in ws) / max(1, len(ws))


def is_verse2(raw_seg: str) -> bool:
    """Verse: short lines (median < 55 characters, >= 5 lines) of which most start with a capital. Hard-wrapped prose has short lines too, but they start in lower case."""
    lines = [l.strip() for l in raw_seg.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    lens = sorted(len(l) for l in lines)
    if lens[len(lens) // 2] >= 55:
        return False
    cap = sum(1 for l in lines if l[0].isupper() or l[0] in '\u00bf\u00a1"\u00ab')
    return cap / len(lines) >= 0.6


def draw_passage(text: str, rng: random.Random, lang: str, lo=120, hi=200, tries=80, min_caps=4, min_stop=0.12, extra_check=None):
    """ner2_passages.draw_passage (sentence start to sentence end, 120-200 words, same filters) with the capitalised-word minimum and the stop-word share as parameters,
    and the verse test of ``is_verse2``. Returns (offset, passage, is_verse) or None."""
    flat = text.replace('\n', ' ')
    starts = [0] + [m.end() for m in SENT.finditer(flat)]
    if len(starts) < 3:
        return None
    for _ in range(tries):
        j = rng.randrange(len(starts))
        s = starts[j]
        k = j
        end = s
        while k < len(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(flat)
            k += 1
            if len(flat[s:end].split()) >= lo:
                break
        seg_raw = text[s:end]
        seg = re.sub(r'\s+', ' ', flat[s:end]).strip()
        ok, _ = passage_ok(seg, lang, lo, hi, min_caps=min_caps, min_stop=min_stop)
        if not ok:
            continue
        if extra_check is not None and not extra_check(seg):
            continue
        if not re.search(r'[.!?:;\u00bb\u201d"\)]\s*$', seg):
            continue
        return s, seg, is_verse2(seg_raw)
    return None


def osta_norm(t: str) -> str:
    """OSTA paleographic transcription: empty editorial brackets dropped, n~ -> n-tilde, c' -> c-cedilla, other tildes dropped, one-to-three-letter [X] supplied letters kept."""
    t = re.sub(r'\[\s*\]|\(\s*\)', ' ', t)
    t = re.sub(r'([nN])~', lambda m: '\u00f1' if m.group(1) == 'n' else '\u00d1', t)
    t = re.sub(r"([cC])'(?=[aeiou\u00e1\u00e9\u00ed\u00f3\u00fa])", lambda m: '\u00e7' if m.group(1) == 'c' else '\u00c7', t)
    t = t.replace('~', '')
    t = re.sub(r'\[([^\[\]]{1,3})\]', r'\1', t)
    return re.sub(r'[ \t]+', ' ', t)


def raw_of_passage(text: str, s: int, seg: str) -> str:
    n = len(seg.split())
    end = s
    k = 0
    for m in re.finditer(r'\S+', text[s:s + 6 * len(seg) + 200]):
        k += 1
        end = s + m.end()
        if k >= n:
            break
    return text[s:end]


def cut_shard(task):
    ex, lang, shard = task
    work = Path(G['work'])
    out_path = work / 'cand' / f'{"tei" if lang == "*" else lang}.{Path(shard).name.replace(".jsonl.gz", "")}.jsonl'
    out_path.parent.mkdir(exist_ok=True)
    df = G['docs']
    meta = G['meta']
    parts, allg = G['excl']
    lex = G['lex']
    st = Counter()
    n_out = 0
    t0 = time.time()
    with open(out_path, 'w', encoding='utf-8') as fo:
        for did, text in read_shard(Path(shard)):
            m = meta.get(did)
            if m is None or (m['task'] and m['task'] != Path(shard).name[:-len('.jsonl.gz')]):
                continue
            lang_doc = m['lang']
            if lang != '*' and lang_doc != lang:
                continue
            st['docs_seen'] += 1
            coll = m['collection']
            if coll == 'OSTA':
                text = osta_norm(text)
            text = EOL_EQ.sub('', text)
            if coll in TRIM_COLLECTIONS:
                n = len(text)
                text = text[int(n * 0.04):int(n * 0.92)]
                text = text[text.find('\n') + 1:] if '\n' in text[:2000] else text
            # document-level 8-gram check against every excluded text
            h = gram_hashes(text)
            hit = in_sorted(h, allg)
            nm = int(hit.sum())
            frac = nm / max(1, len(h))
            if nm >= 25 or frac >= 0.01:
                srcs = {s: int(in_sorted(h, e).sum()) for s, e in parts.items()}
                fo.write(json.dumps({'_doc_flagged': did, 'collection': coll, 'matched_8grams': nm, 'doc_8grams': int(len(h)), 'frac': frac, 'by_source': srcs}) + '\n')
                st['doc_8gram_excluded'] += 1
                continue
            words = len(text.split())
            q = min(DOC_CAP, words // MIN_WORDS_PER_PASSAGE_SLOT)
            if q < 1 and words >= 250:
                q = 1
            if q < 1:
                st['too_short'] += 1
                continue
            rng = random.Random(f'{SEED}/{did}')
            L = len(text)
            bounds = [int(L * i / q) for i in range(q + 1)]
            for i in range(1, q):                       # move each boundary to the next white space
                j = text.find(' ', bounds[i])
                bounds[i] = j if j >= 0 else bounds[i]
            lim = OCR_COLLECTIONS.get(coll, 0.14)
            junk_lim = 1
            chk = None
            if lang_doc == 'la':
                chk = (lambda seg, _l=lim: not STRAY.search(seg) and oov_rate(seg, lex) <= _l and not OCR_JUNK.search(seg) and seg.count('¶') <= junk_lim)
            elif coll == 'OSTA':
                chk = (lambda seg: not STRAY.search(seg) and not re.search(r'[\[\]()]', seg))
            else:
                chk = (lambda seg: not STRAY.search(seg) and not OCR_JUNK.search(seg))
            got = 0
            taken = []                                     # (start, end) of the passages already cut in this document
            kw = dict(min_caps=MIN_CAPS.get(coll, 4), min_stop=0.09 if lang_doc == 'la' else 0.12, extra_check=chk)

            def emit(base, chunk, r):
                nonlocal got, n_out
                s_, seg, verse = r
                off = base + s_
                end = off + len(raw_of_passage(chunk, s_, seg))
                if any(off < e and b < end for b, e in taken):
                    return False
                gp = gram_hashes(seg)
                if int(in_sorted(gp, allg).sum()) >= max(2, int(0.01 * len(gp))):
                    st['passage_8gram_excluded'] += 1
                    return False
                taken.append((off, end))
                fo.write(json.dumps({'doc_id': did, 'lang': lang_doc, 'offset_in_doc': off, 'text': seg, 'n_words': len(seg.split()), 'verse': bool(verse)}, ensure_ascii=False) + '\n')
                got += 1
                n_out += 1
                return True

            for i in range(q):
                chunk = text[bounds[i]:bounds[i + 1]]
                if len(chunk.split()) < 160:
                    continue
                r = draw_passage(chunk, rng, lang_doc, tries=100, **kw)
                if not r:
                    st['no_passage_in_chunk'] += 1
                    continue
                emit(bounds[i], chunk, r)
            # chunks that gave nothing (index, tables, damaged OCR): a few more draws from random windows of the whole document
            miss = q - got
            for _ in range(miss * 3 if miss > 0 else 0):
                if got >= q:
                    break
                p0 = rng.randrange(0, max(1, L - 4000))
                j0 = text.find(' ', p0)
                p0 = j0 if j0 >= 0 else p0
                win = text[p0:p0 + 5000]
                r = draw_passage(win, rng, lang_doc, tries=40, **kw)
                if r and emit(p0, win, r):
                    st['passages_from_fallback'] += 1
            st['passages'] += got
            st['docs_with_passages'] += 1 if got else 0
    log('cut', lang, Path(shard).name, dict(st), f'{time.time() - t0:.0f}s')
    return dict(st)


def _init_cut(work, docs, excl, lex):
    G['work'] = work
    G['docs'] = docs
    G['meta'] = {r['doc_id']: r for r in docs.to_dict('records')}
    G['excl'] = excl
    G['lex'] = lex


def cmd_cut(a):
    import pandas as pd
    import pickle
    work = Path(a.work)
    docs = pd.read_parquet(work / 'docs.parquet')
    excl = load_excl_grams(work)
    lex = pickle.load(open(work / 'lex_la.pkl', 'rb'))
    tasks = []
    for ex in a.extract:
        for lang in LANGS:
            if a.langs and lang not in a.langs:
                continue
            for p in sorted((Path(ex) / 'texts' / lang).glob('*.jsonl.gz')):
                if not a.shards or any(x in p.name for x in a.shards):
                    tasks.append((ex, lang, str(p)))
    for p in sorted((Path(a.tei) / 'texts').glob('*.jsonl.gz')) if a.tei else []:
        if not a.shards or any(x in p.name for x in a.shards):
            tasks.append((a.tei, '*', str(p)))
    log(len(tasks), 'shards')
    with mp.get_context('fork').Pool(a.procs, initializer=_init_cut, initargs=(str(work), docs, excl, lex)) as pool:
        res = list(pool.imap_unordered(cut_shard, tasks, chunksize=1))
    tot = Counter()
    for r in res:
        tot.update(r)
    json.dump(dict(tot), open(work / 'cut_summary.json', 'w'), indent=1)
    log('cut done', dict(tot))


# ----------------------------------------------------------------------------------------------- stage select
def period_bin25(y: int) -> int:
    return min(10, max(0, (int(y) - 1450) // 25))


def period_bin50(y: int) -> int:
    return min(4, max(0, (int(y) - 1450) // 50))


def bin50_label(b: int) -> str:
    return ('1450-1499', '1500-1549', '1550-1599', '1600-1649', '1650-1700')[b]


DRAMA_LABEL = re.compile(r'(?<![A-Za-z\u00c0-\u00ff])[A-Z\u00c0-\u00dd]{3,}(?![A-Za-z\u00c0-\u00ff])|\[[A-Z\u00c0-\u00dd ]{3,}\]')


def is_drama(text: str) -> bool:
    return len(DRAMA_LABEL.findall(text)) >= 3


def bin_label(b: int) -> str:
    lo = 1450 + 25 * b
    return f'{lo}-{lo + 24}' if b < 10 else '1700'


def cmd_select(a):
    import pandas as pd
    work = Path(a.work)
    docs = pd.read_parquet(work / 'docs.parquet')
    meta = {r['doc_id']: r for r in docs.to_dict('records')}
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    flagged = []
    summary = {}
    for lang in LANGS:
        cands = []
        for p in sorted((work / 'cand').glob('*.jsonl')):
            for line in open(p, encoding='utf-8'):
                d = json.loads(line)
                if '_doc_flagged' in d:
                    if lang == 'es':
                        flagged.append(d)
                    continue
                if d['lang'] == lang and d['doc_id'] in meta:
                    cands.append(d)
        rng = random.Random(f'{SEED}/select/{lang}')
        log(lang, 'candidates', len(cands))
        # passage dedup: 1/8 sample of the 8-gram hashes, shared >= 5 % (and >= 2) with an accepted passage -> dropped
        rng.shuffle(cands)
        seen: dict = {}
        kept = []
        ndup = 0
        for c in cands:
            h = gram_hashes(c['text'])
            hs = h[(h % np.uint64(8)) == 0]
            ids = set(int(x) for x in hs)
            shared = sum(1 for x in ids if x in seen)
            if shared >= 2 and shared >= 0.05 * max(1, len(ids)):
                ndup += 1
                continue
            for x in ids:
                seen[x] = 1
            kept.append(c)
        cands = kept
        log(lang, 'after dedup', len(cands), 'duplicates', ndup)
        for c in cands:
            m = meta[c['doc_id']]
            c['collection'] = m['collection']
            c['year'] = int(m['year'])
            c['bin25'] = period_bin25(m['year'])
            c['bin'] = period_bin50(m['year'])
            c['drama'] = is_drama(c['text'])
            c['nonprose'] = bool(c['verse']) or c['drama']
            c['genre'] = m['genre']
        # order: rank inside the document first, then random (every document gets one passage before any gets two)
        by_doc = defaultdict(list)
        for c in cands:
            by_doc[c['doc_id']].append(c)
        order = []
        for did, lst in by_doc.items():
            rng.shuffle(lst)
            for r, c in enumerate(lst):
                order.append((r, rng.random(), c))
        order.sort(key=lambda x: (x[0], x[1]))
        def run_sel(T, f):
            """Greedy pass in the order above. Mandated caps: document (30) and collection (25 % of T). Balance caps (50-year bin, genre, verse and drama), each
            multiplied by f (f = None: off)."""
            cap_c = int(COLL_SHARE * T)
            cap_b = int(min(1.0, BIN_SHARE * f) * T) if f else T
            cap_g = int(min(1.0, GENRE_SHARE * f) * T) if f else T
            cap_v = int(min(1.0, VERSE_SHARE * f) * T) if f else T
            cc, cb, cg, cd = Counter(), Counter(), Counter(), Counter()
            nverse = 0
            out = []
            for r, _, c in order:
                if len(out) >= T:
                    break
                if cd[c['doc_id']] >= DOC_CAP or cc[c['collection']] >= cap_c or cb[c['bin']] >= cap_b or cg[c['genre']] >= cap_g or (c['nonprose'] and nverse >= cap_v):
                    continue
                nverse += 1 if c['nonprose'] else 0
                out.append(c)
                cd[c['doc_id']] += 1
                cc[c['collection']] += 1
                cb[c['bin']] += 1
                cg[c['genre']] += 1
            return out

        # 1. largest size T for which the mandated caps (document 30, collection 25 % of the final size) can be met
        T = a.target
        while True:
            sel = run_sel(T, None)
            if len(sel) >= T or T <= 100:
                break
            log(lang, f'T={T} gives {len(sel)} -> lower the base of the share caps')
            T = len(sel)
        # 2. same size with the balance caps, relaxed step by step only as far as needed
        relax = 1.0
        while True:
            sel = run_sel(T, relax)
            if len(sel) >= T or relax > 3:
                break
            relax *= 1.1
        if len(sel) < T:
            sel = run_sel(T, None)
            relax = None
        summary[lang] = {'balance_cap_factor': relax}
        log(lang, 'selected', len(sel), 'final base', T)
        # final check of the shares against the real size
        N = len(sel)
        summary[lang].update({'target': a.target, 'candidates_after_dedup': len(cands), 'duplicates_dropped': ndup, 'selected': N, 'share_base': T,
                         'max_collection_share': max(Counter(c['collection'] for c in sel).values()) / N})
        # ids: index of the passage among all the passages of its document (pool + reserve), in text order
        idx = {}
        for did, lst in by_doc.items():
            for k, c in enumerate(sorted(lst, key=lambda c: c['offset_in_doc']), 1):
                idx[id(c)] = k

        def rec_of(c, extra=None):
            m = meta[c['doc_id']]
            toks, starts = tokenize(c['text'])
            title = m.get('title') if isinstance(m.get('title'), str) and m.get('title') else m['stem']
            author = m.get('author') if isinstance(m.get('author'), str) and m.get('author') else ''
            rec = {'id': f"silver/{lang}/{c['doc_id']}#{idx[id(c)]}", 'lang': lang, 'collection': m['collection'], 'doc': m['stem'], 'doc_id': c['doc_id'],
                   'title': title, 'author': author, 'date': str(int(m['year'])), 'year': int(m['year']), 'date_precision': m['date_precision'],
                   'date_basis': m['date_confidence'], 'period_bin_25y': bin_label(c['bin25']), 'period_bin_50y': bin50_label(c['bin']), 'genre': m['genre'],
                   'genre_source': m['genre_source'], 'edition_type': m.get('edition_type') or '', 'transmission_medium': m.get('transmission_medium') or '',
                   'licence_note': f"{m.get('licence_tag') or 'n/a'} ({m['collection']}, text archive {m.get('tier') or 'n/a'})",
                   'offset_in_doc': c['offset_in_doc'], 'n_words': c['n_words'], 'text': c['text'], 'tokens': toks, 'token_starts': starts, 'seed': False,
                   'form': 'verse' if c['verse'] else 'drama' if c['drama'] else 'prose', 'modern_transcription_flag': bool(m.get('modern_transcription_flag')),
                   'v1_train_dev_edition_collection': bool(m.get('v1_edition_collection')), 'v1_train_dev_edition_doc': bool(m.get('v1_seen')), 'silver': True}
            if extra:
                rec.update(extra)
            return rec

        per_doc_n = Counter(c['doc_id'] for c in sel)
        path = outdir / f'pool_{lang}.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for c in sorted(sel, key=lambda c: (c['doc_id'], c['offset_in_doc'])):
                f.write(json.dumps(rec_of(c), ensure_ascii=False) + '\n')
        log(lang, 'written', path)
        # reserve: eligible passages (same exclusions, same document cap) that the share caps kept out of the balanced pool
        chosen = {id(c) for c in sel}
        reserve = [c for c in cands if id(c) not in chosen]
        ff = relax or 99.0
        cap_c, cap_b, cap_g, cap_v = int(COLL_SHARE * T), int(min(1.0, BIN_SHARE * ff) * T), int(min(1.0, GENRE_SHARE * ff) * T), int(min(1.0, VERSE_SHARE * ff) * T)
        cc, cb, cg = Counter(c['collection'] for c in sel), Counter(c['bin'] for c in sel), Counter(c['genre'] for c in sel)
        nv = sum(1 for c in sel if c['nonprose'])
        blocked = Counter()
        rpath = outdir / f'pool_{lang}_reserve.jsonl'
        with open(rpath, 'w', encoding='utf-8') as f:
            for c in sorted(reserve, key=lambda c: (c['doc_id'], c['offset_in_doc'])):
                why = [n for n, bad in (('collection_cap', cc[c['collection']] >= cap_c), ('period_cap', cb[c['bin']] >= cap_b), ('genre_cap', cg[c['genre']] >= cap_g),
                                        ('verse_drama_cap', c['nonprose'] and nv >= cap_v)) if bad] or ['size']
                blocked[why[0]] += 1
                f.write(json.dumps(rec_of(c, {'reserve': True, 'blocked_by': why}), ensure_ascii=False) + '\n')
        summary[lang]['reserve'] = {'passages': len(reserve), 'blocked_first_by': dict(blocked), 'by_collection': dict(Counter(c['collection'] for c in reserve).most_common())}
        if not reserve:
            rpath.unlink()
        log(lang, 'reserve', len(reserve), dict(blocked))
        for key, fld in (('by_collection', 'collection'), ('by_bin25', 'bin25'), ('by_bin50', 'bin'), ('by_genre', 'genre')):
            summary[lang][key] = dict(Counter(c[fld] for c in sel).most_common())
        summary[lang]['by_date_precision'] = dict(Counter(meta[c['doc_id']]['date_precision'] for c in sel).most_common())
        summary[lang]['by_form'] = dict(Counter('verse' if c['verse'] else 'drama' if c['drama'] else 'prose' for c in sel).most_common())
        summary[lang]['verse_passages'] = sum(1 for c in sel if c['verse'])
        summary[lang]['drama_passages'] = sum(1 for c in sel if c['drama'] and not c['verse'])
        summary[lang]['v1_seen_doc_passages'] = sum(1 for c in sel if meta[c['doc_id']].get('v1_seen'))
        summary[lang]['docs_used'] = len(per_doc_n)
        summary[lang]['max_per_doc'] = max(per_doc_n.values()) if per_doc_n else 0
        summary[lang]['words_median'] = int(np.median([c['n_words'] for c in sel])) if sel else 0
        summary[lang]['candidates_by_collection'] = dict(Counter(c['collection'] for c in cands).most_common())
    json.dump(summary, open(outdir / 'pool_summary.json', 'w'), indent=1, ensure_ascii=False, default=int)
    json.dump(flagged, open(work / 'docs_flagged_8gram.json', 'w'), indent=1, ensure_ascii=False)
    log('done')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('excl', 'salamanca', 'inv', 'lex', 'cut', 'select'):
        p = sub.add_parser(name)
        p.add_argument('--work', required=True)
        p.add_argument('--extract', nargs='*', default=[])
        p.add_argument('--db', default='')
        p.add_argument('--root', default='')
        p.add_argument('--procs', type=int, default=2)
        p.add_argument('--langs', nargs='*', default=[])
        p.add_argument('--shards', nargs='*', default=[])
        p.add_argument('--tei', default='')
        p.add_argument('--verse-share', type=float, default=0.20)
        p.add_argument('--out', default='')
        p.add_argument('--target', type=int, default=TARGET)
    a = ap.parse_args()
    if a.cmd == 'salamanca':
        a.extract = a.extract[0]
    {'excl': cmd_excl, 'salamanca': cmd_salamanca, 'inv': cmd_inv, 'lex': cmd_lex, 'cut': cmd_cut, 'select': cmd_select}[a.cmd](a)


if __name__ == '__main__':
    main()
