"""RenNER v2.1: apply the frozen post-processing rules (src/ratlas/ner/postproc.py) to a prediction file of ner_audit_predict.py.

    python scripts/ner3_v21_apply_pp.py --data passages.jsonl --pred pred.jsonl.gz --rules drop_generic,extend_loc_adj,trim_punct,split_title --out pred_pp.jsonl.gz
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas.ner import postproc as PP  # noqa: E402


def opn(p, mode='rt'):
    return gzip.open(p, mode, encoding='utf-8') if str(p).endswith('.gz') else open(p, mode[0], encoding='utf-8')


ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--pred', required=True)
ap.add_argument('--rules', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rules = tuple(r for r in a.rules.split(',') if r)
assert all(r in PP.RULES for r in rules), rules
meta = {}
with opn(a.data) as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            meta[r['id']] = (r['tokens'], r.get('lang'))
n = ch = 0
with opn(a.pred) as f, opn(a.out, 'wt') as g:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        toks, lang = meta[d['id']]
        new = PP.apply_pred_line(d, toks, lang, rules)
        for lab in ('LOC', 'PER'):
            ch += len({(x['s'], x['e']) for x in d.get(lab) or []} ^ {(x['s'], x['e']) for x in new[lab]})
        g.write(json.dumps(new, ensure_ascii=False) + '\n')
        n += 1
print(f'{a.pred}: {n} lines, {ch} spans changed', file=sys.stderr)
