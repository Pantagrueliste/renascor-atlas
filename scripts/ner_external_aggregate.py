"""Aggregate the converted external NER datasets into training views (train / dev merged, every test set kept apart).

    python -I scripts/ner_external_aggregate.py [--root ~/renascor-atlas-data/ner/external] [--cap 0.25]

Reads  <root>/<name>/{train,dev,dev2,test,test_gold,all}.jsonl   (scripts/ner_external_convert.py)
Writes <root>/aggregate/<view>/{train,dev}.jsonl, <root>/aggregate/<view>/test/<source>__<subset>__<lang>.jsonl, <root>/aggregate/<view>/MANIFEST.tsv

Views
  hist_gold      historical corpora with gold / reviewed labels (all licences: private use only)
  hist_open      historical corpora whose licence_flag is "open" (CC0 / CC BY / MIT): the part a later published model may keep (gold + silver)
  modern_silver  WikiNEuRal, MultiNERD, WikiANN, Pile-NER-type, Pile-mistral, GLiNER multilingual synthetic: modern, silver or synthetic labels, capped per (source, language)
  all_private    hist_gold + hist_open + synthetic historical + modern_silver (everything above), private use only
Rules (deterministic, hash based)
  * train / dev: the dataset's own split; unsplit sets ("all") follow split_suggested (80 / 10 / 10); "dev2" joins dev; test and test_gold never enter train / dev;
  * no source may exceed ``--cap`` of the train tokens of the view (water-filling, the records of a capped source are sampled by hash);
  * modern sources keep at most ``--modern-records`` train records per (source, language) (8000 in all_private, so that the historical sets are not drowned)
    and an eighth of that in dev; their test sets are kept whole, one file per (source, language);
  * records with loc_valid or per_valid False are kept (the view loss must mask the invalid label: fields loc_valid / per_valid).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HIST = {'hipe2022', 'europeana', 'reichsanzeiger', 'kind', 'geoedda', 'globalise', 'earlymodern_german', 'sdhk_old_swedish', 'latin_herodotos', 'latin_lasla',
        'latincy_ner', 'medieval_charters'}
HIST_SYNTH = {'hist_pile_ner'}
MODERN = {'wikineural', 'multinerd', 'wikiann', 'pile_ner', 'pile_mistral', 'gliner_multi_synth'}
# KIND: only the historical chapter (De Gasperi 1901-1954) counts as historical
HIST_SUBSETS = {'kind': {'degasperi'}}

VIEWS = {
    'hist_gold': dict(sources=HIST, quality={'gold', 'reviewed'}, flags=None),
    'hist_open': dict(sources=HIST, quality={'gold', 'reviewed', 'silver'}, flags={'open'}),
    'modern_silver': dict(sources=MODERN, quality={'silver', 'synthetic', 'gold'}, flags=None),
    'all_private': dict(sources=HIST | HIST_SYNTH | MODERN, quality={'gold', 'reviewed', 'silver', 'synthetic'}, flags=None, modern_records=8000),
}


def h32(s: str) -> float:
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) / 2 ** 32


def cap_tokens(sizes: dict, cap: float) -> dict:
    """Water-filling: the largest groups are cut to a common ceiling so that no group exceeds ``cap`` of the total (when feasible)."""
    total = sum(sizes.values())
    if not total or len(sizes) * cap < 1:
        return dict(sizes)
    allow = dict(sizes)
    for _ in range(50):
        tot = sum(allow.values())
        limit = cap * tot
        over = [k for k, v in allow.items() if v > limit + 1e-9]
        if not over:
            break
        for k in over:
            allow[k] = limit
    return allow


def records(path: Path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def where(rec: dict) -> str:
    sp = rec['split']
    if sp == 'all':
        return rec.get('split_suggested', 'train')
    return {'dev2': 'dev', 'val': 'dev', 'validation': 'dev'}.get(sp, sp)   # test, test_gold, test_ood are kept as they are


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path.home() / 'renascor-atlas-data/ner/external'))
    ap.add_argument('--cap', type=float, default=0.25)
    ap.add_argument('--modern-records', type=int, default=30000)
    ap.add_argument('--views', default='')
    a = ap.parse_args()
    root = Path(a.root)
    names = [v for v in a.views.split(',') if v] or list(VIEWS)
    for view in names:
        spec = VIEWS[view]
        outdir = root / 'aggregate' / view
        (outdir / 'test').mkdir(parents=True, exist_ok=True)
        # pass 1: sizes of the train groups (source) and the per (source, lang) record counters for the modern sets
        sizes: Counter = Counter()
        files = []
        for d in sorted(spec['sources']):
            for p in sorted((root / d).glob('*.jsonl')) if (root / d).is_dir() else []:
                files.append((d, p))
        keep = lambda r: (r['quality'] in spec['quality'] and (spec['flags'] is None or r['licence_flag'] in spec['flags'])  # noqa: E731
                          and (r['source'] not in HIST_SUBSETS or r['subset'] in HIST_SUBSETS[r['source']] or r['source'] in MODERN))
        per_lang: Counter = Counter()
        per_lang_dev: Counter = Counter()
        for d, p in files:
            for r in records(p):
                if keep(r) and where(r) == 'dev' and r['source'] in MODERN:
                    per_lang_dev[(r['source'], r['lang'])] += 1
                if keep(r) and where(r) == 'train':
                    if r['source'] in MODERN:
                        per_lang[(r['source'], r['lang'])] += 1
                    sizes[r['source']] += len(r['tokens'])
        # modern per-language record cap changes the sizes: recompute with the cap applied as a keep probability
        mod_n = spec.get('modern_records', a.modern_records)
        mod_p = {k: min(1.0, mod_n / n) for k, n in per_lang.items()}
        dev_p = {k: min(1.0, (mod_n // 8) / n) for k, n in per_lang_dev.items()}
        sizes2: Counter = Counter()
        for d, p in files:
            for r in records(p):
                if keep(r) and where(r) == 'train':
                    if r['source'] in MODERN and h32(r['id'] + '|m') >= mod_p[(r['source'], r['lang'])]:
                        continue
                    sizes2[r['source']] += len(r['tokens'])
        allow = cap_tokens(dict(sizes2), a.cap)
        prob = {k: min(1.0, allow[k] / v) for k, v in sizes2.items() if v}
        # pass 2: write
        fh = {sp: open(outdir / f'{sp}.jsonl', 'w', encoding='utf-8') for sp in ('train', 'dev')}
        tests: dict[str, object] = {}
        stat: Counter = Counter()
        for d, p in files:
            for r in records(p):
                if not keep(r):
                    continue
                sp = where(r)
                if sp.startswith('test'):
                    key = f'{r["source"]}__{r["subset"]}__{r["lang"]}' + ('' if sp == 'test' else '__' + sp[5:])
                    if key not in tests:
                        tests[key] = open(outdir / 'test' / f'{key}.jsonl', 'w', encoding='utf-8')
                    tests[key].write(json.dumps(r, ensure_ascii=False) + '\n')
                    stat[('test:' + key, 'records')] += 1
                    stat[('test:' + key, 'tokens')] += len(r['tokens'])
                    stat[('test:' + key, 'LOC')] += r['n_loc']
                    stat[('test:' + key, 'PER')] += r['n_per']
                    continue
                if sp not in fh:
                    continue
                if sp == 'dev' and r['source'] in MODERN and h32(r['id'] + '|d') >= dev_p[(r['source'], r['lang'])]:
                    continue
                if sp == 'train':
                    if r['source'] in MODERN and h32(r['id'] + '|m') >= mod_p[(r['source'], r['lang'])]:
                        continue
                    if h32(r['id'] + '|c') >= prob.get(r['source'], 1.0):
                        continue
                fh[sp].write(json.dumps(r, ensure_ascii=False) + '\n')
                k = f'{sp}:{r["source"]}'
                stat[(k, 'records')] += 1
                stat[(k, 'tokens')] += len(r['tokens'])
                stat[(k, 'LOC')] += r['n_loc']
                stat[(k, 'PER')] += r['n_per']
                if not r['loc_valid']:
                    stat[(k, 'loc_invalid')] += 1
                if not r['per_valid']:
                    stat[(k, 'per_invalid')] += 1
        for f in list(fh.values()) + list(tests.values()):
            f.close()
        keys = sorted({k for k, _ in stat})
        with open(outdir / 'MANIFEST.tsv', 'w') as m:
            m.write('part\trecords\ttokens\tLOC\tPER\tloc_invalid\tper_invalid\n')
            for k in keys:
                m.write('\t'.join([k] + [str(stat[(k, c)]) for c in ('records', 'tokens', 'LOC', 'PER', 'loc_invalid', 'per_invalid')]) + '\n')
        tr = sum(v for (k, c), v in stat.items() if k.startswith('train:') and c == 'tokens')
        print(f'{view}: train {tr} tokens, sources {dict((k[6:], stat[(k, "tokens")]) for k in keys if k.startswith("train:"))}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
