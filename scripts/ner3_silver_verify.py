"""Independent leak check of the silver pools: string 8-grams (ner2_passages.fingerprints, folded words) of every pool passage against every text of gold v2 (all files),
the gold seed, dev3 and test_v2, plus a document-name check. Different code path from the 64-bit hashes of ner3_silver_pool.py.

    python scripts/ner3_silver_verify.py --pool-dir DIR --out report.json
    python scripts/ner3_silver_verify.py --file balanced_es=final/silver_balanced_es.jsonl [--file ...] \
        --ref v1_dev=data_v1/loc/dev.jsonl --ref v1_dev=data_v1/per/dev.jsonl --ref v1_test=... --details shared.jsonl --out report.json

--file checks the given files (silver training files) instead of the pool files of --pool-dir. --ref adds reference sets (name=path,
repeatable; same name = one set), e.g. the v1 dev and test splits. --details lists every checked passage that shares at least one 8-gram
with a reference set (id, set, count, examples). The exclusion check looks for London Lives, MAP (Map of Early Modern London) and the private
collections (CELT, CIDTC) in the collection, document and id fields.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from ratlas.ner.ner2_passages import fingerprints, fold  # noqa: E402

NER = Path('/Users/clem/Projects/Papers/RQ/derived/atlas/ner')


def norm(s):
    return re.sub(r'[^0-9a-z]', '', fold(s))


def iter_json(p):
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


EXCLUDED = {'london_lives': re.compile(r'london_?lives|londonlives|\boldbailey', re.I),
            'map': re.compile(r'map_of_early_modern_london|\bmoeml\b|mapoflondon|\bMAP_', re.I),
            'private': re.compile(r'\bcelt\b|\bcidtc\b|(^|[/_])celt[/_]|(^|[/_])cidtc[/_]', re.I)}


def exclusion_hits(d):
    fields = ' '.join(str(d.get(k, '')) for k in ('collection', 'doc', 'doc_id', 'id', 'edition', 'source'))
    return [k for k, rx in EXCLUDED.items() if rx.search(fields)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool-dir')
    ap.add_argument('--file', action='append', default=[], help='name=path of a file to check (instead of --pool-dir)')
    ap.add_argument('--ref', action='append', default=[], help='extra reference set name=path (repeatable)')
    ap.add_argument('--details', help='JSONL of the checked passages sharing an 8-gram with a reference set')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    files = {
        'gold_v2': sorted((NER / 'v2/gold2').glob('*.jsonl')) + [NER / 'gold_seed_v1.jsonl'],
        'dev3': [NER / 'v2/renner/dev3/passages.jsonl', NER / 'v2/renner/dev3/gold_dev3.jsonl'] + sorted((NER / 'v2/renner/dev3/annotation').glob('*.jsonl')),
        'test_v2': [NER / 'v2/test2/test_v2.jsonl'] + sorted((NER / 'v2/test2').glob('cand_*.jsonl')) + sorted((NER / 'v2/test2').glob('judged_*.jsonl')),
    }
    refs = set()
    for spec in a.ref:
        name, _, path = spec.partition('=')
        if not Path(path).exists():
            raise FileNotFoundError(path)
        files.setdefault(name, []).append(Path(path))
        refs.add(name)
    grams = {}
    docs = defaultdict(set)
    for src, fl in files.items():
        g = set()
        seen = set()
        for p in fl:
            if not p.exists():
                continue
            for d in iter_json(p):
                t = d.get('text')
                if isinstance(t, str) and t not in seen:
                    seen.add(t)
                    g |= fingerprints(t)
                if d.get('doc'):
                    n = norm(str(d['doc']))
                    if len(n) >= 6:
                        docs[src].add(n)
        grams[src] = g
        print(src, len(seen), 'texts', len(g), '8-grams', len(docs[src]), 'doc names', flush=True)
    rep = {}
    if a.file:
        targets = [(spec.partition('=')[0], Path(spec.partition('=')[2])) for spec in a.file]
    else:
        targets = [(lang + suf, Path(a.pool_dir) / f'pool_{lang}{suf}.jsonl') for lang in ('es', 'it', 'la') for suf in ('', '_reserve')]
    det = open(a.details, 'w', encoding='utf-8') if a.details else None
    for name, path in targets:
        if not path.exists() or path.stat().st_size == 0:
            continue
        n = 0
        excl = Counter()
        colls = Counter()
        worst = Counter()
        any_shared = Counter()
        ge1 = Counter()
        name_hits = []
        max_share = {k: 0.0 for k in grams}
        for d in iter_json(path):
            n += 1
            colls[d.get('collection', '?')] += 1
            for k in exclusion_hits(d):
                excl[k] += 1
            fp = fingerprints(d['text'])
            for src, g in grams.items():
                common = fp & g
                sh = len(common)
                if sh:
                    any_shared[src] += 1
                    if det:
                        det.write(json.dumps({'file': name, 'id': d['id'], 'set': src, 'shared': sh, 'of': len(fp), 'examples': sorted(common)[:5]},
                                             ensure_ascii=False) + '\n')
                share = sh / max(1, len(fp))
                max_share[src] = max(max_share[src], share)
                if share >= 0.01:
                    ge1[src] += 1
            nm = norm(d.get('doc', ''))
            for src, names in docs.items():
                if src in refs:                       # large reference sets: exact document names only
                    if nm in names:
                        name_hits.append((d['id'], src, nm))
                    continue
                for x in names:
                    if nm == x or (len(nm) >= 8 and len(x) >= 8 and (x in nm or nm in x)):
                        name_hits.append((d['id'], src, x))
                        break
        rep[name] = {'file': str(path), 'passages': n, 'passages_sharing_any_8gram': dict(any_shared), 'passages_ge_1pct': dict(ge1), 'max_share': max_share,
                     'doc_name_hits': len(name_hits), 'doc_name_hits_by_set': dict(Counter(s for _, s, _ in name_hits)), 'doc_name_hit_examples': name_hits[:10],
                     'exclusion_hits': dict(excl), 'collections': dict(colls.most_common())}
        print(name, {k: v for k, v in rep[name].items() if k != 'collections'}, flush=True)
    if det:
        det.close()
    json.dump(rep, open(a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
