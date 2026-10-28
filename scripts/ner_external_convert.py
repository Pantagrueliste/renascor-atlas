"""Convert the public NER datasets fetched by ner_external_fetch.py to our JSONL format (tokens + BIO, LOC / PER only).

    python -I scripts/ner_external_convert.py [--root ~/renascor-atlas-data/ner/external] [--only a,b] [--list]

Reads  <root>/<name>/raw/...        (untrusted data: parsed with the standard library, pyarrow for parquet; nothing is executed)
Writes <root>/<name>/<split>.jsonl  one record per segment (see ratlas.ner.external for the fields), counts.json, and <root>/CONVERTED.tsv
Their own train / dev / test splits are kept; the sets without a split get split='all' and a deterministic split_suggested.
Every record carries licence, licence_flag (open | nc | nc-sa | gpl | unstated | synthetic) and quality (gold | reviewed | silver | synthetic).
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from ratlas.ner import external as E

csv.field_size_limit(1 << 28)

LOC_NAMES = ['LOC', 'GPE', 'GEO', 'LOCATION', 'NP-Spatial', 'LOC_NAME']
PER_NAMES = ['PER', 'PERS', 'PRS', 'PERSON', 'PERSONA', 'NP-Person', 'PER_NAME']
TOKRE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.U)


def tokenize(text: str) -> tuple[list[str], list[int], list[int]]:
    toks, st, en = [], [], []
    for m in TOKRE.finditer(text):
        toks.append(m.group())
        st.append(m.start())
        en.append(m.end())
    return toks, st, en


def char_to_tok_spans(st: list[int], en: list[int], cspans: list[tuple[int, int, str]]) -> tuple[list[tuple[int, int, str]], int]:
    """Character spans -> token spans (a token touched by the span is inside it).  Returns (spans, number of spans whose edge cut a token)."""
    out, cut = [], 0
    for a, b, lab in cspans:
        idx = [i for i in range(len(st)) if st[i] < b and en[i] > a]
        if not idx:
            continue
        if st[idx[0]] != a or en[idx[-1]] != b:
            cut += 1
        out.append((idx[0], idx[-1] + 1, lab))
    return out, cut


def spans_to_bio(n: int, spans: list[tuple[int, int, str]]) -> list[str]:
    bio = ['O'] * n
    for s, e, lab in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if any(bio[k] != 'O' for k in range(s, e)):
            continue
        for k in range(s, e):
            bio[k] = ('B-' if k == s else 'I-') + lab
    return bio


def emit(out: E.Out, *, base: str, source: str, subset: str, split: str, lang: str, tokens: list[str], bio: list[str], nsa: Optional[list[bool]] = None,
         stops: Optional[list[bool]] = None, loc: Iterable[str] = LOC_NAMES, per: Iterable[str] = PER_NAMES, meta: Optional[dict] = None,
         suggest_key: Optional[str] = None, limit: int = E.MAX_TOKENS, quality: Optional[str] = None, licence: Optional[str] = None,
         flag: Optional[str] = None, **kw) -> int:
    """Write one segment (cut into chunks of <= limit tokens).  Returns the number of records written."""
    if not tokens:
        return 0
    n = 0
    for k, (a, b) in enumerate(E.chunk(len(tokens), bio, stops, limit)):
        m = dict(meta or {})
        if suggest_key is not None:
            m['split_suggested'] = E.suggested_split(f'{suggest_key}#{k // 25}')     # blocks of 25 chunks (<= 5 000 tokens): a file without sentence breaks is not one unit
        rec = E.make_record(rid=f'{base}#{k}' if k else base, source=source, subset=subset, split=split, lang=lang, tokens=tokens[a:b], bio=bio[a:b],
                            no_space_after=(nsa[a:b] if nsa else None), loc_types=loc, per_types=per, meta=m, **kw)
        out.write(rec, licence, flag, quality)
        n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------------------------------------------------
def read_columns(path: Path, sep: str = '\t', tok: int = 0, tag: int = 1, new_sentence: Optional[Callable[[list[str]], bool]] = None,
                 last_tag: bool = False) -> Iterator[tuple[list[str], list[str]]]:
    """Token / tag columns, sentences separated by blank lines (or by ``new_sentence(fields)`` returning True on the first token of a sentence)."""
    toks, tags = [], []
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line.strip():
                if toks:
                    yield toks, tags
                    toks, tags = [], []
                continue
            fields = line.split(sep) if sep else line.split()
            if len(fields) <= max(tok, tag):
                continue
            if last_tag and len(fields) > 2:        # a token that holds spaces: the tag is the last field
                fields = ['_'.join(fields[:-1]), fields[-1]]
            if new_sentence and toks and new_sentence(fields):
                yield toks, tags
                toks, tags = [], []
            toks.append(fields[tok])
            tags.append(fields[tag])
    if toks:
        yield toks, tags


def read_hipe(path: Path, tagcol: str = 'NE-COARSE-LIT') -> Iterator[dict]:
    """HIPE TSV: one dict per sentence (EndOfSentence in MISC, a blank line, or a new document): doc, meta, tokens, tags, nsa, stops."""
    header, meta, doc = None, {}, None
    toks, tags, nsa, stops = [], [], [], []
    seq = 0

    def flush():
        nonlocal toks, tags, nsa, stops, seq
        if toks:
            seq += 1
            r = {'doc': doc, 'meta': dict(meta), 'tokens': toks, 'tags': tags, 'nsa': nsa, 'stops': stops, 'seq': seq}
            toks, tags, nsa, stops = [], [], [], []
            return r
        return None

    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if header is None:
                header = line.split('\t')
                ti, ci = header.index('TOKEN'), header.index(tagcol)
                mi = header.index('MISC') if 'MISC' in header else None
                continue
            if line.startswith('#'):
                m = re.match(r'#\s*(?:hipe2022:)?(\w+)\s*=\s*"?(.*?)"?\s*$', line)
                if m:
                    k, v = m.group(1), m.group(2)
                    if k == 'document_id':
                        r = flush()
                        if r:
                            yield r
                        doc, meta, seq = v.split(' (see ')[0], {}, 0
                    else:
                        meta[k] = v
                continue
            if not line.strip():
                r = flush()
                if r:
                    yield r
                continue
            fields = line.split('\t')
            if len(fields) <= max(ti, ci):
                continue
            misc = fields[mi] if mi is not None and mi < len(fields) else '_'
            toks.append(fields[ti])
            t = fields[ci]
            tags.append('O' if t in ('_', '') else t)
            nsa.append('NoSpaceAfter' in misc)
            stop = 'EndOfSentence' in misc
            stops.append(stop)
            if stop:
                r = flush()
                if r:
                    yield r
    r = flush()
    if r:
        yield r


def hipe_year(meta: dict) -> Optional[int]:
    for k in ('date', 'decade'):
        m = re.match(r'(\d{4})', meta.get(k, '') or meta.get('newseye:' + k, ''))
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------------------------------------------------
# topres19th annotates places only; in ajmc the places are annotated sparsely (43 / 45 / 24 'loc' against about 900 'pers' per language, HIPE-2022 overview table 5)
HIPE_NO_PER = {'topres19th'}
HIPE_SPARSE_LOC = {'ajmc'}


def conv_hipe2022(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'hipe2022/raw').glob('HIPE-2022-data-*'))))
    lic = {'hipe2020': ('CC BY-NC-SA 4.0', 'nc-sa'), 'letemps': ('CC BY-NC-SA 4.0', 'nc-sa'), 'topres19th': ('CC BY-NC-SA 4.0', 'nc-sa'),
           'newseye': ('CC BY 4.0', 'open'), 'sonar': ('CC BY 4.0', 'open'), 'ajmc': ('CC BY 4.0', 'open')}
    for p in sorted(base.glob('data/v2.1/*/*/HIPE-2022-v2.1-*.tsv')):
        m = re.match(r'HIPE-2022-v2\.1-(\w+)-(train|dev2|dev|test)-(\w+)\.tsv$', p.name)
        if not m:        # test_ELmasked / test_allmasked and ocr-correction tables are not used
            continue
        ds, sp, lang = m.groups()
        L, flag = lic[ds]
        for s in read_hipe(p):
            emit(out, base=f'hipe2022:{ds}:{lang}:{sp}:{s["doc"]}:{s["seq"]}', source='hipe2022', subset=ds, split=sp, lang=lang, tokens=s['tokens'], bio=s['tags'],
                 nsa=s['nsa'], stops=s['stops'], loc=['loc'], per=['pers', 'per'], meta={'doc': s['doc'], 'year': hipe_year(s['meta'])},
                 per_valid=ds not in HIPE_NO_PER, loc_valid=ds not in HIPE_SPARSE_LOC)


def _with_licence(out: E.Out, table: dict[str, tuple[str, str]]):
    """Patch Out.write so that each record gets the licence of its subset."""
    orig = out.write

    def w(rec, licence=None, flag=None, quality=None):
        L, F = table.get(rec['subset'], (out.licence, out.flag))
        orig(rec, licence or L, flag or F, quality)
    out.write = w


def conv_europeana(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'europeana/raw').glob('ner-corpora-*'))))
    for d in sorted(base.glob('enp_*.bio')):
        f = d / d.name
        sub = d.name[:-4]                      # enp_FR.bnf
        lang = sub[4:6].lower()
        for i, (toks, tags) in enumerate(read_columns(f, sep=' ', tok=0, tag=1, last_tag=True)):
            if any(t.startswith(('url:', 'file:')) or '\t' in t for t in toks):      # metadata lines of the file, not text
                continue
            tags = [t.upper() if t[:2].upper() in ('B-', 'I-') else t for t in tags]
            emit(out, base=f'europeana:{sub}:{i}', source='europeana', subset=sub, split='all', lang=lang, tokens=toks, bio=tags, loc=['LOC'], per=['PER'],
                 suggest_key=f'{sub}:{i}')


def conv_kind(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'kind/raw').glob('KIND-*'))))
    for p in sorted(base.glob('dataset/*_*.tsv')):
        sub, sp = p.stem.rsplit('_', 1)
        fl = 'nc-sa' if sub == 'degasperi' else 'nc'
        L = 'CC BY-NC-SA 4.0' if sub == 'degasperi' else 'CC BY-NC 4.0'
        q = 'silver' if sub == 'moro' else 'gold'
        for i, (toks, tags) in enumerate(read_columns(p, sep='\t')):
            for k, (a, b) in enumerate(E.chunk(len(toks), tags)):
                r = E.make_record(rid=f'kind:{sub}:{sp}:{i}#{k}', source='kind', subset=sub, split=sp, lang='it', tokens=toks[a:b], bio=tags[a:b],
                                  loc_types=['LOC'], per_types=['PER'])
                out.write(r, L, fl, q)


def conv_reichsanzeiger(root: Path, out: E.Out):
    import xml.etree.ElementTree as ET
    base = next(iter(sorted((root / 'reichsanzeiger/raw').glob('reichsanzeiger-nlp-*'))))
    for p in sorted(base.glob('data/curated/*.xmi')):
        raw = p.read_bytes().replace(b'<?xml version="1.1"', b'<?xml version="1.0"', 1)
        tree = ET.fromstring(raw)
        text, toks, ents, sents = '', [], [], []
        for el in tree.iter():
            tag = el.tag.rsplit('}', 1)[-1]
            if tag == 'Sofa':
                text = el.get('sofaString') or ''
            elif tag == 'Token':
                toks.append((int(el.get('begin')), int(el.get('end'))))
            elif tag == 'NamedEntity':
                ents.append((int(el.get('begin')), int(el.get('end')), el.get('value') or ''))
            elif tag == 'Sentence':
                sents.append((int(el.get('begin')), int(el.get('end'))))
        toks.sort()
        st, en = [a for a, _ in toks], [b for _, b in toks]
        words = [text[a:b] for a, b in toks]
        cs = []
        for a, b, v in ents:
            vl = v.lower()
            lab = 'LOC' if (vl.startswith('loc.') and vl != 'loc.add') else ('PER' if vl.startswith('per.') else None)
            if lab:
                cs.append((a, b, lab))
            else:
                out.dropped[v] += 1
        tsp, cut = char_to_tok_spans(st, en, cs)
        bio = spans_to_bio(len(toks), tsp)
        year = int(p.stem[:4]) if p.stem[:4].isdigit() else None
        # sentences -> segments
        sents.sort()
        for j, (sa, sb) in enumerate(sents):
            idx = [i for i in range(len(toks)) if st[i] >= sa and en[i] <= sb]
            if not idx:
                continue
            i0, i1 = idx[0], idx[-1] + 1
            nsa = [en[i] == st[i + 1] if i + 1 < len(toks) else True for i in range(i0, i1)]
            emit(out, base=f'reichsanzeiger:{p.stem}:{j}', source='reichsanzeiger', subset='reichsanzeiger', split='all', lang='de', tokens=words[i0:i1],
                 bio=bio[i0:i1], nsa=nsa, loc=['LOC'], per=['PER'], meta={'doc': p.stem, 'year': year}, suggest_key=f'ra:{p.stem}')


def conv_geoedda(root: Path, out: E.Out):
    for sp0, fn in (('train', 'train.jsonl'), ('dev', 'val.jsonl'), ('test', 'test.jsonl')):
        for i, line in enumerate(open(root / 'geoedda/raw' / fn, encoding='utf-8')):
            d = json.loads(line)
            toks = [t['text'] for t in d['tokens']]
            nsa = [not t.get('ws', True) for t in d['tokens']]
            sp = []
            for s in d['spans']:
                if s['label'] in ('NP-Spatial', 'NP-Person'):
                    sp.append((s['token_start'], s['token_end'] + 1, 'LOC' if s['label'] == 'NP-Spatial' else 'PER'))
                else:
                    out.dropped[s['label']] += 1
            emit(out, base=f'geoedda:{sp0}:{i}', source='geoedda', subset='edda', split=sp0, lang='fr', tokens=toks, bio=spans_to_bio(len(toks), sp), nsa=nsa,
                 loc=['LOC'], per=['PER'], meta={'doc': f'{d["meta"].get("volume")}-{d["meta"].get("article")}', 'year': 1765})


def conv_globalise(root: Path, out: E.Out):
    for sp0, fn in (('train', 'train.csv'), ('dev', 'validation.csv'), ('test', 'test.csv')):
        with open(root / 'globalise/raw' / fn, encoding='utf-8', newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                toks, labs = ast.literal_eval(row['tokens']), ast.literal_eval(row['labels'])
                emit(out, base=f'globalise:{sp0}:{i}', source='globalise', subset='voc', split=sp0, lang='nl', tokens=toks, bio=labs, loc=['LOC_NAME'],
                     per=['PER_NAME'], meta={'year': None})


def conv_earlymodern_german(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'earlymodern_german/raw').glob('NER-eval-corpus_*'))))
    f = next(iter(sorted(base.glob('files HIPE scorer format/0_*.tsv'))))
    for s in read_hipe(f):
        emit(out, base=f'emg:{s["doc"]}:{s["seq"]}', source='earlymodern_german', subset=(s['doc'] or 'x')[:6], split='test', lang='de', tokens=s['tokens'],
             bio=s['tags'], nsa=None, stops=s['stops'], loc=['loc'], per=['pers', 'per'], meta={'doc': s['doc']})


def conv_sdhk(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'sdhk_old_swedish/raw').glob('sdhk-ner-old-swedish-*')))) / 'data'
    man = json.loads((base / 'internal_test_set/split_manifest.json').read_text())
    where = {}
    for k, v in man.items():
        if isinstance(v, list) and v and isinstance(v[0], str) and v[0].endswith('.conll'):
            for n in v:
                where[n] = {'train': 'train', 'val': 'dev', 'dev': 'dev', 'validation': 'dev', 'test': 'test'}.get(k, k)
    for p in sorted((base / '1380_1382_dataset').glob('*.conll')):
        sp = where.get(p.name, 'train')
        for i, (toks, tags) in enumerate(read_columns(p, sep='', tok=0, tag=1)):
            emit(out, base=f'sdhk:{p.stem}:{i}', source='sdhk_old_swedish', subset='charters-1380-82', split=sp, lang='sv', tokens=toks, bio=tags,
                 loc=['Place', 'Location', 'Loc', 'Plats', 'LOC'], per=['Person', 'Pers', 'PER'], meta={'doc': p.stem, 'year': 1381}, quality='silver')
    for p in sorted((base / 'expert_gold_test').glob('*.conll')):
        for i, (toks, tags) in enumerate(read_columns(p, sep='', tok=0, tag=1)):
            emit(out, base=f'sdhk:{p.stem}:{i}', source='sdhk_old_swedish', subset='charters-expert-gold', split='test_gold', lang='sv', tokens=toks, bio=tags,
                 loc=['Place', 'Location', 'Loc', 'Plats', 'LOC'], per=['Person', 'Pers', 'PER'], meta={'doc': p.stem}, quality='gold')


def conv_herodotos(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'latin_herodotos/raw').glob('Herodotos-Project-*'))))
    for p in sorted(base.glob('Annotation_1-1-19/*.crf')):
        sp = 'test' if p.stem == 'GWtest' else 'train'
        for i, (toks, tags) in enumerate(read_columns(p, sep='\t', tok=1, tag=0)):
            emit(out, base=f'herodotos:{p.stem}:{i}', source='latin_herodotos', subset=p.stem, split=sp, lang='la', tokens=toks, bio=tags, loc=['GEO'], per=['PRS'],
                 meta={'doc': p.stem})


def conv_lasla(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'latin_lasla/raw').glob('Ner-Latin-RANLP-*'))))
    for p in sorted(base.glob('Latin_Gold_Data/*_GOLD.tsv')):
        sub = p.stem.replace('_GOLD', '')
        for i, (toks, tags) in enumerate(read_columns(p, sep='\t', tok=1, tag=4, new_sentence=lambda f: f[0] == '1')):
            emit(out, base=f'lasla:{sub}:{i}', source='latin_lasla', subset=sub, split='test', lang='la', tokens=toks, bio=tags, loc=['GEO', 'LOC'],
                 per=['PER', 'PERS'], meta={'doc': sub})


def conv_latincy(root: Path, out: E.Out):
    base = next(iter(sorted((root / 'latincy_ner/raw').glob('latincy-ner-*'))))
    for sp in ('train', 'dev'):
        for i, (toks, tags) in enumerate(read_columns(base / f'assets/processed/bio/{sp}.tsv', sep='\t')):
            emit(out, base=f'latincy:{sp}:{i}', source='latincy_ner', subset='primer', split=sp, lang='la', tokens=toks, bio=tags, loc=['LOC', 'GPE'],
                 per=['PERSON'], quality='reviewed')


def conv_wikineural(root: Path, out: E.Out):
    import pyarrow.parquet as pq
    names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    for p in sorted((root / 'wikineural/raw/data').glob('*.parquet')):
        m = re.match(r'(train|val|test)_(\w+)-', p.name)
        sp0, lang = m.groups()
        sp = 'dev' if sp0 == 'val' else sp0
        t = pq.read_table(p, columns=['tokens', 'ner_tags'])
        toks, tags = t.column('tokens').to_pylist(), t.column('ner_tags').to_pylist()
        for i in range(len(toks)):
            emit(out, base=f'wikineural:{lang}:{sp}:{i}', source='wikineural', subset='wikineural', split=sp, lang=lang, tokens=toks[i],
                 bio=[names[x] for x in tags[i]], loc=['LOC'], per=['PER'])


MULTINERD = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-ANIM', 'I-ANIM', 'B-BIO', 'I-BIO', 'B-CEL', 'I-CEL', 'B-DIS', 'I-DIS', 'B-EVE', 'I-EVE',
             'B-FOOD', 'I-FOOD', 'B-INST', 'I-INST', 'B-MEDIA', 'I-MEDIA', 'B-MYTH', 'I-MYTH', 'B-PLANT', 'I-PLANT', 'B-TIME', 'I-TIME', 'B-VEHI', 'I-VEHI']


def conv_multinerd(root: Path, out: E.Out):
    for p in sorted((root / 'multinerd/raw').glob('*/*_*.jsonl')):
        m = re.match(r'(train|val|test)_(\w+)\.jsonl$', p.name)
        sp0, lang = m.groups()
        if lang == 'zh':           # no word spacing: not useful for the European target
            continue
        sp = 'dev' if sp0 == 'val' else sp0
        for i, line in enumerate(open(p, encoding='utf-8')):
            d = json.loads(line)
            emit(out, base=f'multinerd:{lang}:{sp}:{i}', source='multinerd', subset='multinerd', split=sp, lang=lang, tokens=d['tokens'],
                 bio=[MULTINERD[x] for x in d['ner_tags']], loc=['LOC'], per=['PER'])


def conv_wikiann(root: Path, out: E.Out):
    import pyarrow.parquet as pq
    names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    for d in sorted(p for p in (root / 'wikiann/raw').iterdir() if p.is_dir()):
        lang = d.name
        for f in sorted(d.glob('*.parquet')):
            sp0 = f.name.split('-')[0]
            sp = 'dev' if sp0 == 'validation' else sp0
            t = pq.read_table(f, columns=['tokens', 'ner_tags'])
            toks, tags = t.column('tokens').to_pylist(), t.column('ner_tags').to_pylist()
            for i in range(len(toks)):
                emit(out, base=f'wikiann:{lang}:{sp}:{i}', source='wikiann', subset='wikiann', split=sp, lang=lang, tokens=toks[i], bio=[names[x] for x in tags[i]],
                     loc=['LOC'], per=['PER'])


PILE_LOC = {'location', 'locations', 'place', 'places', 'city', 'cities', 'country', 'countries', 'state', 'region', 'province', 'town', 'county', 'village',
            'geographical location', 'geographic location', 'geopolitical entity', 'gpe', 'nation', 'continent', 'capital', 'capital city', 'city name',
            'country name', 'place name', 'location name', 'state name', 'geographical region', 'geographic region', 'territory', 'municipality'}
PILE_LOC_BROAD = {'location', 'locations', 'place', 'places', 'geographical location', 'geographic location', 'geopolitical entity', 'gpe'}
PILE_PER = {'person', 'people', 'persons', 'individual', 'person name', 'personal name', 'name of person', 'human', 'full name'}


def conv_pile_ner(root: Path, out: E.Out):
    data = json.loads((root / 'pile_ner/raw/train.json').read_text(encoding='utf-8'))
    types = Counter()
    for d in data:
        conv = d['conversations']
        text = conv[0]['value']
        if not text.startswith('Text: '):
            continue
        text = text[6:]
        found: dict[str, list[str]] = {}
        for k in range(2, len(conv) - 1, 2):
            q = conv[k]['value']
            m = re.match(r'What describes (.+?) in the text\?', q)
            if not m:
                continue
            ty = m.group(1).strip().lower()
            types[ty] += 1
            try:
                ans = json.loads(conv[k + 1]['value'])
            except Exception:  # noqa: BLE001
                continue
            if ty in PILE_LOC:
                found.setdefault('LOC', []).extend(a for a in ans if isinstance(a, str))
            elif ty in PILE_PER:
                found.setdefault('PER', []).extend(a for a in ans if isinstance(a, str))
        qs = [c['value'].lower() for c in conv[2::2]]
        # LOC is exhaustive only when a BROAD place type was asked ('country' alone says nothing about the cities); PER likewise needs 'person'
        asked_loc = any(q.startswith(f'what describes {t} in') for q in qs for t in PILE_LOC_BROAD)
        asked_per = any(q.startswith(f'what describes {t} in') for q in qs for t in PILE_PER)
        if not asked_loc and not asked_per:
            continue
        toks, st, en = tokenize(text)
        if len(toks) > 400 or len(toks) < 3:
            continue
        cs = []
        for lab, names in found.items():
            for nm in sorted(set(names), key=len, reverse=True):
                if not nm.strip():
                    continue
                for m in re.finditer(r'(?<!\w)' + re.escape(nm) + r'(?!\w)', text):
                    cs.append((m.start(), m.end(), lab))
        tsp, _ = char_to_tok_spans(st, en, cs)
        nsa = [en[i] == st[i + 1] if i + 1 < len(toks) else True for i in range(len(toks))]
        bio = spans_to_bio(len(toks), tsp)
        emit(out, base=f'pile:{d["id"]}', source='pile_ner', subset='pile-ner-type', split='all', lang='en', tokens=toks, bio=bio, nsa=nsa, loc=['LOC'], per=['PER'],
             loc_valid=asked_loc, per_valid=asked_per, suggest_key=d['id'], quality='synthetic')
    (out.dir / 'pile_types_top.json').write_text(json.dumps(types.most_common(300), ensure_ascii=False, indent=0))


def conv_hist_pile(root: Path, out: E.Out):
    for p in sorted((root / 'hist_pile_ner/raw/hipe_iob').glob('train-*.tsv')):
        lang = p.stem.split('-', 1)[1]
        for s in read_hipe(p):
            emit(out, base=f'histpile:{lang}:{s["doc"]}:{s["seq"]}', source='hist_pile_ner', subset='europeana-gpt5mini', split='all', lang=lang, tokens=s['tokens'],
                 bio=s['tags'], nsa=s['nsa'], stops=s['stops'], loc=['loc'], per=['pers', 'per'], meta={'doc': s['doc']}, suggest_key=s['doc'] or '', quality='synthetic')


GL_LOC = {'location', 'locations', 'place', 'places', 'city', 'country', 'state', 'region', 'province', 'town', 'county', 'village', 'geographical location',
          'geographic location', 'geopolitical entity', 'gpe', 'nation', 'continent', 'capital', 'territory', 'municipality', 'city name', 'country name', 'place name',
          'location name', 'island', 'district', 'ort', 'stadt', 'land', 'lieu', 'ville', 'pays', 'lugar', 'luogo', 'citta', 'città'}
GL_PER = {'person', 'people', 'persons', 'individual', 'person name', 'personal name', 'name of person', 'human', 'full name', 'persona', 'personne', 'person_name',
          'personen', 'osoba', 'persoon'}


def _norm_type(t: str) -> str:
    return re.sub(r'\s+', ' ', t.strip().lower())


def conv_gliner_json(root: Path, out: E.Out, name: str, rel: str, lang: str):
    """GLiNER training format: tokenized_text + ner [[start, end (inclusive), type]].  Only the types that were annotated are listed: a record is
    valid for LOC (PER) only when it holds at least one place (person) span, i.e. these sets give positive evidence, not exhaustive annotation."""
    data = json.loads((root / name / 'raw' / rel).read_text(encoding='utf-8'))
    for i, d in enumerate(data):
        toks = d['tokenized_text']
        sp, has = [], set()
        for s, e, ty in d.get('ner', []):
            t = _norm_type(ty)
            lab = 'LOC' if t in GL_LOC else ('PER' if t in GL_PER else None)
            if lab and 0 <= s <= e < len(toks):
                sp.append((s, e + 1, lab))
                has.add(lab)
            else:
                out.dropped[t] += 1
        if not has:
            continue
        emit(out, base=f'{name}:{i}', source=name, subset=name, split='all', lang=lang, tokens=toks, bio=spans_to_bio(len(toks), sp), loc=['LOC'], per=['PER'],
             loc_valid='LOC' in has, per_valid='PER' in has, suggest_key=f'{name}:{i}', quality='synthetic')


def conv_pile_mistral(root: Path, out: E.Out):
    conv_gliner_json(root, out, 'pile_mistral', 'data.json', 'en')


def conv_gliner_multi_synth(root: Path, out: E.Out):
    conv_gliner_json(root, out, 'gliner_multi_synth', 'european_dataset.json', 'mul')


MED_DIR = 'medieval_charters'


def conv_medieval(root: Path, out: E.Out):
    """Torres Aguilar 2022 (Zenodo 6463699): see conv_medieval_zip; written once the zip has been inspected (scripts/ner_external_medieval.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not put the script directory on sys.path
    from ner_external_medieval import convert  # noqa: PLC0415
    convert(root, out, emit=emit, spans_to_bio=spans_to_bio)


DATASETS: dict[str, tuple[Callable, str, str, str]] = {
    # name: (function, licence, flag, quality)
    'hipe2022': (conv_hipe2022, 'CC BY-NC-SA 4.0 / CC BY 4.0 per subset', 'nc-sa', 'gold'),
    'europeana': (conv_europeana, 'CC0 1.0', 'open', 'gold'),
    'reichsanzeiger': (conv_reichsanzeiger, 'CC0 1.0', 'open', 'gold'),
    'kind': (conv_kind, 'CC BY-NC 4.0 / CC BY-NC-SA 4.0', 'nc', 'gold'),
    'geoedda': (conv_geoedda, 'CC BY-NC 4.0', 'nc', 'gold'),
    'globalise': (conv_globalise, 'MIT', 'open', 'gold'),
    'earlymodern_german': (conv_earlymodern_german, 'none stated', 'unstated', 'gold'),
    'sdhk_old_swedish': (conv_sdhk, 'CC BY 4.0', 'open', 'silver'),
    'latin_herodotos': (conv_herodotos, 'AGPL-3.0', 'gpl', 'gold'),
    'latin_lasla': (conv_lasla, 'CC BY-NC-SA', 'nc-sa', 'gold'),
    'latincy_ner': (conv_latincy, 'CC BY-NC-SA 4.0', 'nc-sa', 'reviewed'),
    'wikineural': (conv_wikineural, 'CC BY-NC-SA 4.0', 'nc-sa', 'silver'),
    'multinerd': (conv_multinerd, 'CC BY-NC-SA 4.0', 'nc-sa', 'silver'),
    'wikiann': (conv_wikiann, 'unclear (PAN-X / WikiANN)', 'unstated', 'silver'),
    'pile_ner': (conv_pile_ner, 'none stated; ChatGPT labels', 'synthetic', 'synthetic'),
    'pile_mistral': (conv_pile_mistral, 'Apache-2.0 (card); Mistral-7B labels', 'synthetic', 'synthetic'),
    'gliner_multi_synth': (conv_gliner_multi_synth, 'none stated; synthetic', 'synthetic', 'synthetic'),
    'hist_pile_ner': (conv_hist_pile, 'none stated; GPT-5-mini labels over CC0 text', 'synthetic', 'synthetic'),
    'medieval_charters': (conv_medieval, 'CC BY 4.0', 'open', 'gold'),
}

HIPE_LICENCE = {'hipe2020': ('CC BY-NC-SA 4.0', 'nc-sa'), 'letemps': ('CC BY-NC-SA 4.0', 'nc-sa'), 'topres19th': ('CC BY-NC-SA 4.0', 'nc-sa'),
                'newseye': ('CC BY 4.0', 'open'), 'sonar': ('CC BY 4.0', 'open'), 'ajmc': ('CC BY 4.0', 'open')}


def write_summary(root: Path) -> Path:
    """CONVERTED.tsv from the counts.json of every converted dataset directory."""
    hdr = ['dataset', 'split', 'subset', 'lang', 'licence_flag', 'quality', 'records', 'tokens', 'LOC', 'PER', 'loc_invalid_records', 'per_invalid_records']
    f = root / 'CONVERTED.tsv'
    with open(f, 'w') as fh:
        fh.write('\t'.join(hdr) + '\n')
        for d in sorted(p for p in root.iterdir() if (p / 'counts.json').exists()):
            for r in json.loads((d / 'counts.json').read_text())['rows']:
                fh.write('\t'.join(str(r[h]) for h in hdr) + '\n')
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path.home() / 'renascor-atlas-data/ner/external'))
    ap.add_argument('--only', default='')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        print('\n'.join(DATASETS))
        return 0
    root = Path(a.root)
    names = [n for n in a.only.split(',') if n] or list(DATASETS)
    allrows = []
    for n in names:
        fn, lic, flag, q = DATASETS[n]
        out = E.Out(root, n, lic, flag, q)
        if n == 'hipe2022':
            _with_licence(out, HIPE_LICENCE)
        print(f'== {n}', flush=True)
        try:
            fn(root, out)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'   FAILED {n}: {e}', file=sys.stderr, flush=True)
        rows = out.close()
        allrows += rows
        t = Counter()
        for r in rows:
            t['records'] += r['records']
            t['tokens'] += r['tokens']
            t['LOC'] += r['LOC']
            t['PER'] += r['PER']
        print(f'   {dict(t)}  dropped source labels: {dict(out.dropped.most_common(8))}', flush=True)
    f = write_summary(root)
    print('wrote', f)
    return 0


if __name__ == '__main__':
    sys.exit(main())
