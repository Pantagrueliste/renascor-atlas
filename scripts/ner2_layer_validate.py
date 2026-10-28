"""Check that the numpy inference of ratlas.ner.layerpred reproduces the v1 independent inference (scripts/ner_audit_predict.py): same spans, same confidences.

    python scripts/ner2_layer_validate.py --model-dir MODELS/r3_xlmr_large_H --data seed_passages.jsonl --ref pred_seed/r3_xlmr_large_H.jsonl.gz [--limit N]

Each record of --data is run as one document with a single passage (size = infinity) so that the windows are exactly those of the v1 inference. Prints the
share of identical span sets, the number of spans only in one side, and the maximum confidence difference (CPU fp32 against the bf16 reference: expect ~1e-2).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    import torch
    from ratlas.ner import layerpred as LP
    dev = torch.device(a.device)
    b = LP.load_bundle(a.model_dir, dev)
    recs = []
    with open(a.data, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
                if a.limit and len(recs) >= a.limit:
                    break
    ref = {}
    with gzip.open(a.ref, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ref[d['id']] = {(x['s'], x['e']): x['conf'] for x in d['LOC']}
    texts = [r['text'] for r in recs]
    pack = LP.prepare_pack(b.ti, list(range(len(recs))), texts, [None] * len(recs), size=10 ** 9, stride=10 ** 9)
    lab, pm, pp = LP.run_batches(b, pack)
    sp = LP.pack_spans(pack, lab, pm, pp)
    mine = {}
    for d, s, e, c in zip(sp['doc'], sp['w_start'], sp['w_end'], sp['conf']):
        mine.setdefault(int(d), {})[(int(s), int(e))] = float(c)
    same = only_m = only_r = 0
    dmax = 0.0
    n_rec_same = 0
    for i, r in enumerate(recs):
        m, v = mine.get(i, {}), ref[r['id']]
        n_rec_same += set(m) == set(v)
        for k in set(m) & set(v):
            same += 1
            dmax = max(dmax, abs(m[k] - v[k]))
        only_m += len(set(m) - set(v))
        only_r += len(set(v) - set(m))
    print(f'records {len(recs)}  identical span sets {n_rec_same}  spans common {same}  only-mine {only_m}  only-v1 {only_r}  max |conf diff| {dmax:.4f}')


if __name__ == '__main__':
    main()
