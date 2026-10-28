"""NER v2: tokenise evaluation sets into the cache of ratlas.ner.train (same cache key as its final evaluation), on CPU before the GPU job.

    python scripts/ner3_train_encode_eval.py --model MODEL [--trust-remote-code] --cache CACHE --set name=path ...
"""
import argparse
import time

from ratlas.ner import nerdata as nd
from ratlas.ner import train as T

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--trust-remote-code', action='store_true')
ap.add_argument('--cache', required=True)
ap.add_argument('--max-len', type=int, default=256)
ap.add_argument('--set', action='append', default=[])
a = ap.parse_args()
tok = T.load_tokenizer(a.model, a.trust_remote_code)
for name, path, _ in T.parse_named(a.set):
    t0 = time.time()
    files = T.expand(path)
    recs = []
    for f in files:
        recs += nd.read_slim(f)
    try:
        enc = T.encode_cached(tok, a.model, recs, files, f'eval|{a.max_len}', a.cache)
    except ValueError as e:      # report the records whose tags are not integers, then go on with the next set
        bad = [(r.get('id'), f) for f in files for r in nd.read_slim(f)
               if any(not isinstance(t, int) for t in (r.get('ner_tags_loc') or []) + (r.get('ner_tags_per') or []))]
        print(f'[encode-eval] {a.model} {name}: FAILED {e}; {len(bad)} records with non-integer tags, e.g. {bad[:3]}', flush=True)
        continue
    print(f'[encode-eval] {a.model} {name}: {len(files)} files, {len(enc)} records, {time.time() - t0:.0f}s', flush=True)
