"""RenNER rescore under the owner rules: write the teacher's dev3 output with the silver post-processing (+pp: final '.' dropped,
names propagated; ner3_teacher_bakeoff.strip_final_dot / propagate), one file per prompt mix, so that ner3_eval_v2_dev3.py can score
several teachers in one run (its --teacher option takes one teacher only).
    python scripts/ner3_rescore_teacher_pp.py --passages dev3/passages.jsonl --out OUT.jsonl LANGS=raw.jsonl [LANGS=raw.jsonl ...]
LANGS is a comma list (es,it,la); only dev3 ids of those languages are kept. Gold v2 and gold v3 are never read.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ner3_teacher_score as S  # noqa: E402

_spec = importlib.util.spec_from_file_location('bk', HERE / 'ner3_teacher_bakeoff.py')
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passages', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('src', nargs='+')
    a = ap.parse_args()
    ps = S.load(a.passages)
    toks = {r['id']: r['tokens'] for r in ps}
    lang = {r['id']: r['lang'] for r in ps}
    t = {}
    for s in a.src:
        lg, p = s.split('=', 1)
        lgs = set(lg.split(','))
        raw = {d['id']: S.to_pred(d) for d in S.load(p) if lang.get(d['id']) in lgs}
        t.update(bk.propagate(bk.strip_final_dot(raw, toks), toks))
    with open(a.out, 'w', encoding='utf-8') as f:
        for i in toks:
            if i in t:
                d = t[i]
                f.write(json.dumps({'id': i, 'LOC': d['LOC'], 'PER': d['PER']}, ensure_ascii=False) + '\n')
    print(a.out, len(t), 'of', len(toks))


if __name__ == '__main__':
    main()
