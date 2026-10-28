"""RenNER variant without CELT and CIDTC: per-seed copy of the modern_silver sample, so that the variant sees exactly the same modern texts
as the lc25-lv4 run of the same seed. train.py samples modern_silver to 6 M words with the seed (nerdata.sample_records, a shuffle of the
record list): dropping even one record from the file would change the whole sample. Here: the sample of the ORIGINAL file is drawn with the
same function and seed, the records listed in --drop (ids) are removed, and the kept lines are written in the order train.py would load
them (sorted by id, as sample_records returns them). Read without a cap, the copy gives the same records (minus the dropped ones).

    python scripts/ner3_noT3_modern_presample.py --src modern_silver/train.jsonl --cap 6000000 --seed 1 --drop dropped.jsonl --out OUT --report R
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner import nerdata as nd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--cap', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--drop', required=True, help='dropped.jsonl of ner3_noT3_filter.py (rows with file=modern_silver)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    drop = {r['id'] for r in map(json.loads, open(a.drop, encoding='utf-8')) if r['file'] == 'modern_silver'}
    lines, recs = [], []
    with nd.open_text(a.src) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            d = json.loads(s)
            recs.append({'id': d['id'], 'tokens': d['tokens'], '_i': len(lines)})
            lines.append(s)
    samp = nd.sample_records(recs, a.cap, a.seed)
    hit = [r['id'] for r in samp if r['id'] in drop]
    n = 0
    with open(a.out, 'w', encoding='utf-8') as g:
        for r in samp:
            if r['id'] in drop:
                continue
            g.write(lines[r['_i']] + '\n'); n += 1
    rep = {'seed': a.seed, 'records_file': len(recs), 'sample_records': len(samp), 'sample_words': sum(len(r['tokens']) for r in samp),
           'dropped_in_sample': hit, 'written': n, 'written_words': sum(len(r['tokens']) for r in samp if r['id'] not in drop)}
    json.dump(rep, open(a.report, 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
