"""Plain-LLM baseline, variant "+prop": name propagation only (ner3_teacher_bakeoff.propagate: a predicted name is also marked at its
other occurrences in the passage). The teacher's +pp also drops final dots (a RenNER convention); +prop does not.

    python scripts/ner3_plain_llm_prop.py --passages P.jsonl [--passages ...] --pred raw__plain.jsonl --out raw__plain+prop.jsonl
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
    ap.add_argument('--passages', action='append', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    toks = {}
    for p in a.passages:
        toks.update({r['id']: r['tokens'] for r in S.load(p)})
    raw = {d['id']: S.to_pred(d) for d in S.load(a.pred)}
    out = bk.propagate(raw, toks)
    with open(a.out, 'w', encoding='utf-8') as f:
        for i, d in out.items():
            f.write(json.dumps({'id': i, 'LOC': d['LOC'], 'PER': d['PER']}, ensure_ascii=False) + '\n')
    added = sum(len(out[i][l]) - len(raw[i].get(l, [])) for i in out for l in ('LOC', 'PER'))
    print(f'{a.pred}: {len(out)} passages, {added} spans added by propagation -> {a.out}')


if __name__ == '__main__':
    main()
