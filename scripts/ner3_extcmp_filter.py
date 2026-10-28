"""External comparison: keep the predicted spans with conf >= THRESHOLD (GLiNER-type models were run at 0.3, the threshold is chosen on dev3).

    python3 scripts/ner3_extcmp_filter.py IN.jsonl.gz OUT.jsonl.gz 0.7
"""
import gzip
import json
import sys

src, dst, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
with gzip.open(src, 'rt', encoding='utf-8') as f, gzip.open(dst, 'wt', encoding='utf-8') as g:
    for line in f:
        d = json.loads(line)
        for k in ('LOC', 'PER'):
            d[k] = [x for x in d.get(k, []) if x.get('conf', 1.0) >= t]
        g.write(json.dumps(d, ensure_ascii=False) + '\n')
