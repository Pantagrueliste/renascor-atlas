"""Build silver files from raw teacher outputs (renner_llm_teacher.py): post-processing +pp and, if several teachers, a vote.

    python scripts/ner3_silver_combine.py --passages pool_or_sample.jsonl --raw a.jsonl [--raw b.jsonl ...] [--vote 2] --out silver.jsonl [--stats s.json]

One --raw is one teacher. Its value may be a file, a glob pattern, or several of them separated by ',' (outputs of one teacher split over
jobs and GPU shards: la_j01__gemma4_31b_v3.jsonl,la_j01__gemma4_31b_v3.s*.jsonl); the lines are merged by passage id, a good line winning over
an error line. --stats writes per-teacher counts: ok, invalid_json, error, missing, empty answers, truncated answers (finish_reason "length"),
and anchoring failures (entity strings of the answer not found in the passage, field "unmatched" of the raw line).

+pp (as in the dev3 bake-off, ner3_teacher_bakeoff.py): per teacher, a final '.' token is dropped from every span, then a name
found once is marked at its other occurrences in the passage. With several --raw files, a span is kept when at least --vote
teachers give it (same start, end and label); overlapping survivors are resolved by votes, then length. With one --raw file, its
+pp spans are kept as they are.

Output: the passage record (all pool fields: id, lang, text, tokens, token_starts, metadata) plus
  spans: [{start, end, label, text, char_start, char_end, votes}]   (end-exclusive token indices, like gold v2)
  teacher: {models, prompt_files, vote, pp}
  teacher_status: {model: "ok" | "invalid_json" | "missing" | "error"}
Passages with no usable answer from enough teachers are written with "silver_ok": false (spans from what is available).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('bk', HERE / 'ner3_teacher_bakeoff.py')
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)


def raw_files(spec):
    """A file, a glob pattern, or several of them separated by ','. Sorted, without duplicates; a pattern that matches nothing is an error."""
    files = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        hits = sorted(glob.glob(part)) if any(c in part for c in '*?[') else [part]
        if not hits:
            raise FileNotFoundError(part)
        files += [h for h in hits if h not in files]
    return files


def _rank(d):
    """Line quality: a usable answer beats invalid JSON, which beats a transport error."""
    e = d.get('error')
    return 2 if e is None else (1 if e == 'invalid_json' else 0)


def load_raw(spec):
    out = {}
    for path in raw_files(spec):
        for l in open(path, encoding='utf8'):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d['id'] in out and _rank(out[d['id']]) >= _rank(d):
                continue                  # keep an earlier line unless the new one is better (good > invalid_json > error)
            out[d['id']] = d
    return out


def pp_one(d, toks):
    pred = {d['id']: bk.sc.to_pred(d)}
    t = {d['id']: toks}
    return bk.propagate(bk.strip_final_dot(pred, t), t)[d['id']]


def resolve(cands):
    """cands: list of (s, e, label, votes) -> non-overlapping list, more votes first, then longer."""
    keep = []
    for s, e, lab, v in sorted(cands, key=lambda x: (-x[3], -(x[1] - x[0]), x[0])):
        if all(e <= a or b <= s for a, b, _, _ in keep):
            keep.append((s, e, lab, v))
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passages', required=True)
    ap.add_argument('--raw', action='append', required=True)
    ap.add_argument('--vote', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt-file', default='scripts/renner_prompt_rules_en.md + scripts/renner_fewshot_v1train.jsonl')
    ap.add_argument('--only-done', action='store_true', help='write only passages answered by every teacher (partial runs)')
    ap.add_argument('--stats', help='JSON file with the per-teacher counts (validity, empty, truncated, anchoring failures)')
    a = ap.parse_args()
    raws = [load_raw(p) for p in a.raw]
    k = a.vote or (1 if len(raws) == 1 else (len(raws) // 2 + 1))
    models = []
    for r in raws:
        models.append(next((d['model'] for d in r.values()), '?'))
    n = n_bad = 0
    st = {m: {'ok': 0, 'invalid_json': 0, 'error': 0, 'missing': 0, 'empty_answer': 0, 'truncated': 0,
              'entities_answered': 0, 'unmatched_entities': 0, 'passages_with_unmatched': 0} for m in models}
    with open(a.out, 'w', encoding='utf8') as f:
        for l in open(a.passages, encoding='utf8'):
            p = json.loads(l)
            toks = p['tokens']
            ends = [s + len(t) for s, t in zip(p['token_starts'], toks)]
            status, cnt = {}, defaultdict(int)
            for m, r in zip(models, raws):
                d = r.get(p['id'])
                if d is None:
                    status[m] = 'missing'
                    st[m]['missing'] += 1
                    continue
                status[m] = 'ok' if d.get('error') is None else ('invalid_json' if d['error'] == 'invalid_json' else 'error')
                st[m][status[m]] += 1
                st[m]['truncated'] += d.get('finish_reason') == 'length'
                if status[m] != 'ok':
                    continue
                um = len(d.get('unmatched') or [])
                st[m]['entities_answered'] += len(d.get('spans') or []) + um
                st[m]['unmatched_entities'] += um
                st[m]['passages_with_unmatched'] += um > 0
                st[m]['empty_answer'] += not d.get('spans') and not um
                q = pp_one(d, toks)
                for lab in ('LOC', 'PER'):
                    for x in q.get(lab, []):
                        cnt[(x['s'], x['e'], lab)] += 1
            if a.only_done and any(v == 'missing' or v == 'error' for v in status.values()):
                continue
            good = sum(v == 'ok' for v in status.values())
            spans = resolve([(s, e, lab, v) for (s, e, lab), v in cnt.items() if v >= min(k, max(good, 1))])
            rec = dict(p)
            rec['spans'] = [{'start': s, 'end': e, 'label': lab, 'text': ' '.join(toks[s:e]),
                             'char_start': p['token_starts'][s], 'char_end': ends[e - 1], 'votes': v} for s, e, lab, v in spans]
            rec['teacher'] = {'models': models, 'prompt_files': a.prompt_file, 'vote': f'{k}/{len(models)}', 'pp': 'strip_final_dot+propagate'}
            rec['teacher_status'] = status
            rec['silver_ok'] = good >= k
            n_bad += not rec['silver_ok']
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
    print(f'{n} passages written to {a.out}; {n_bad} without enough usable teacher answers (silver_ok false); vote {k}/{len(models)}')
    if a.stats:
        json.dump({'passages': n, 'silver_ok_false': n_bad, 'vote': f'{k}/{len(models)}', 'raw': [raw_files(r) for r in a.raw], 'teachers': st},
                  open(a.stats, 'w'), indent=1)


if __name__ == '__main__':
    main()
