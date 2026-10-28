"""External comparison: convert the public test sets that are NOT in the RenNER training mixes to our JSONL (tokens + BIO, LOC / PER only).

    python -I scripts/ner3_extcmp_convert.py --name neolatiner --raw DIR --out OUT_DIR
    names: neolatiner (HF mhvdr/neolatiner-dataset, test.parquet), evalatin2026 (LT4HALA 2026 NER test, HIPE TSV), eneide (Zenodo 17407356, DZ + AMD test),
           grandtour (Zenodo 23056599, data/testData.json, spaCy-style JSON)

Label maps (everything else -> O; licences kept as metadata only):
    neolatiner  GEO -> LOC, PRS -> PER, GRP -> O
    evalatin    place -> LOC, person -> PER (NE-COARSE-LIT); collective, creature, work, time, language, misc -> O
    eneide      LOC -> LOC, PER -> PER; ORG, WORK -> O (silver, semi-automatic)
    grandtour   LOC -> LOC, PER -> PER; other labels -> O
Reads untrusted data with the standard library and pyarrow only. Writes OUT_DIR/<name>__<subset>.jsonl and OUT_DIR/<name>.counts.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
sys.path.insert(0, str(HERE))
from ratlas.ner import external as E  # noqa: E402
import ner_external_convert as C  # noqa: E402

csv.field_size_limit(1 << 28)


class Sink:
    """E.Out-like writer: one file per subset, test split only."""

    def __init__(self, out: Path, name: str, licence: str, flag: str, quality: str):
        self.out, self.name, self.licence, self.flag, self.quality = out, name, licence, flag, quality
        self.fh = {}
        self.count = Counter()

    def write(self, rec, licence=None, flag=None, quality=None):
        rec['licence'] = licence or self.licence
        rec['licence_flag'] = flag or self.flag
        rec['quality'] = quality or self.quality
        rec['edition'] = f"{self.name}/{rec['subset']}"
        key = rec['subset']
        if key not in self.fh:
            self.fh[key] = open(self.out / f'{self.name}__{key}.jsonl', 'w', encoding='utf-8')
        self.fh[key].write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.count[(key, 'records')] += 1
        self.count[(key, 'tokens')] += len(rec['tokens'])
        self.count[(key, 'LOC')] += rec['n_loc']
        self.count[(key, 'PER')] += rec['n_per']

    def close(self):
        for f in self.fh.values():
            f.close()
        d = {}
        for (k, f), v in self.count.items():
            d.setdefault(k, {})[f] = v
        (self.out / f'{self.name}.counts.json').write_text(json.dumps(d, indent=1))
        print(json.dumps(d, indent=1))


def conv_neolatiner(raw: Path, out: Sink):
    import pyarrow.parquet as pq
    names = ['O', 'B-PRS', 'I-PRS', 'B-GRP', 'I-GRP', 'B-GEO', 'I-GEO']
    for i, r in enumerate(pq.read_table(raw / 'test.parquet').to_pylist()):
        C.emit(out, base=f'neolatiner:test:{i}', source='neolatiner', subset='test', split='test', lang='la', tokens=list(r['tokens']),
               bio=[names[x] for x in r['ner_tags']], loc=['GEO'], per=['PRS'], meta={'doc': 'neolatiner'})


def conv_evalatin(raw: Path, out: Sink):
    for s in C.read_hipe(raw / 'evalatin2026-ner-v0.1-test.tsv', 'NE-COARSE-LIT'):
        C.emit(out, base=f"evalatin2026:{s['doc']}:{s['seq']}", source='evalatin2026', subset='test', split='test', lang='la', tokens=s['tokens'], bio=s['tags'],
               nsa=s['nsa'], stops=s['stops'], loc=['place'], per=['person'], meta={'doc': s['doc'], 'title': s['meta'].get('title')},
               licence=s['meta'].get('license'))


def conv_eneide(raw: Path, out: Sink):
    base = raw / 'ENEIDE-data' / 'v1.0'
    for sub in ('DZ', 'AMD'):
        anns = {}
        for a in csv.DictReader(open(base / sub / 'annotations_test.csv', encoding='utf-8')):
            anns.setdefault(a['doc_id'], []).append((int(a['start_pos']), int(a['end_pos']), a['type']))
        for i, p in enumerate(csv.DictReader(open(base / sub / 'paragraphs_test.csv', encoding='utf-8'))):
            text = p['text']
            toks, st, en = C.tokenize(text)
            sp, _ = C.char_to_tok_spans(st, en, [x for x in anns.get(p['doc_id'], []) if x[2] in ('LOC', 'PER')])
            nsa = [en[k] == st[k + 1] for k in range(len(toks) - 1)] + [False]
            C.emit(out, base=f'eneide:{sub}:{i}', source='eneide', subset=sub, split='test', lang='it', tokens=toks, bio=C.spans_to_bio(len(toks), sp), nsa=nsa,
                   loc=['LOC'], per=['PER'], meta={'doc': p['doc_id'], 'date': p.get('publication_date')})


def conv_grandtour(raw: Path, out: Sink):
    """data/testData.json: list of [text, {"entities": [[start, end, label], ...]}] (spaCy training JSON); other layouts are reported."""
    root = next(iter(sorted(raw.glob('**/data/testData.json'))))
    data = json.loads(root.read_text(encoding='utf-8'))
    items = data if isinstance(data, list) else data.get('annotations') or data.get('data') or []
    labs = Counter()
    for i, it in enumerate(items):
        if isinstance(it, dict):
            text = it.get('text', '')
            ents = it.get('entities') or (it.get('label') or [])
        else:
            text, ann = it[0], it[1]
            ents = ann.get('entities', []) if isinstance(ann, dict) else ann
        cs = []
        for e in ents:
            if isinstance(e, dict):
                a, b, lab = e.get('start'), e.get('end'), e.get('label')
            else:
                a, b, lab = e[0], e[1], e[2]
            labs[lab] += 1
            if lab in ('LOC', 'PER'):
                cs.append((int(a), int(b), lab))
        toks, st, en = C.tokenize(text)
        sp, _ = C.char_to_tok_spans(st, en, cs)
        nsa = [en[k] == st[k + 1] for k in range(len(toks) - 1)] + [False]
        C.emit(out, base=f'grandtour:test:{i}', source='grandtour', subset='test', split='test', lang='de', tokens=toks, bio=C.spans_to_bio(len(toks), sp),
               nsa=nsa, loc=['LOC'], per=['PER'], meta={'doc': f'grandtour:{i}'})
    print('grandtour labels', dict(labs))


SETS = {
    'neolatiner': (conv_neolatiner, 'no licence field on the dataset card', 'unstated', 'gold'),
    'evalatin2026': (conv_evalatin, 'CC-BY (file header)', 'open', 'gold'),
    'eneide': (conv_eneide, 'CC BY-NC-SA 4.0', 'nc-sa', 'silver'),
    'grandtour': (conv_grandtour, 'MIT (Zenodo record)', 'open', 'gold'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, choices=sorted(SETS))
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fn, lic, flag, q = SETS[a.name]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Sink(out, a.name, lic, flag, q)
    fn(Path(a.raw), s)
    s.close()


if __name__ == '__main__':
    main()
