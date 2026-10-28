"""NER v2, large encoders: training windows per silver passage, for the wall-time estimate of v1 + silver (CPU, a few seconds per tokenizer).

    python scripts/ner3_enc_windows.py --sample SILVER_SAMPLE.jsonl --tok NAME=PATH [--tok ...] [--trust NAME] --out windows.json

Each record of the sample is tokenised and cut into windows exactly as ratlas.ner.train does (max_len 256, no overlap in training).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from ratlas.ner import nerdata as nd
from ratlas.ner.train import load_tokenizer, wrapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--tok', action='append', required=True)
    ap.add_argument('--force-bos', action='append', default=[])
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    recs = nd.read_slim(a.sample)
    res = {}
    for spec in a.tok:
        name, _, path = spec.partition('=')
        tok = load_tokenizer(path, False)
        prefix, suffix = wrapping(tok, name in a.force_bos)
        limit = a.max_len - len(prefix) - len(suffix)
        enc = nd.encode_records(tok, recs)
        rows = nd.build_windows(enc, limit, 0)
        by = defaultdict(lambda: {'passages': 0, 'words': 0, 'subwords': 0, 'windows': 0})
        for i, m in enumerate(enc.meta):
            d = by[m['lang']]
            d['passages'] += 1
            d['words'] += int(enc.word_off[i + 1] - enc.word_off[i])
            d['subwords'] += int(enc.sub_off[i + 1] - enc.sub_off[i])
        for r in rows:
            by[enc.meta[int(r[0])]['lang']]['windows'] += 1
        tot = {k: sum(v[k] for v in by.values()) for k in ('passages', 'words', 'subwords', 'windows')}
        for d in list(by.values()) + [tot]:
            d['words_per_passage'] = round(d['words'] / d['passages'], 1)
            d['subwords_per_word'] = round(d['subwords'] / d['words'], 3)
            d['windows_per_passage'] = round(d['windows'] / d['passages'], 3)
        res[name] = {'by_lang': dict(by), 'all': tot, 'limit': limit}
        print(name, json.dumps(tot), flush=True)
    open(a.out, 'w').write(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
