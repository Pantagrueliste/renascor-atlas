"""Summarise the Ollama teacher bake-off on dev3: scores (via ner3_teacher_score), ensembles, bootstrap intervals, run statistics.

    python scripts/ner3_teacher_bakeoff.py --dir RQ/derived/atlas/ner/v2/renner/teacher_ollama --gold dev3/gold_dev3.jsonl \
        [--best3 a,b,c] [--best2 a,b] [--bootstrap 1000]

Reads <dir>/pred/<teacher>__<prompt>.jsonl (output of renner_llm_teacher.py) and <dir>/logs/<teacher>__<prompt>.log; writes
<dir>/scores_all.json, <dir>/scores_sure.json, <dir>/run_stats.json and prints markdown tables. dev3 only: never gold v2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('sc', HERE / 'ner3_teacher_score.py')
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)
LANGS = ['es', 'it', 'la', 'all']


def run_stats(path: Path, log: Path, gold_ids):
    rs = {}
    for l in open(path, encoding='utf-8'):
        d = json.loads(l)
        rs[d['id']] = d                  # the last line of an id wins (retry pass)
    rs = {i: d for i, d in rs.items() if i in gold_ids}
    vals = list(rs.values())
    n_ent = sum(len(d['spans']) + len(d['unmatched']) for d in vals)
    secs = [d['seconds'] for d in vals if d.get('error') is None]
    walls = [int(x) for x in re.findall(r'finished: \d+ passages, \d+ errors, (\d+) s wall', log.read_text())] if log.exists() else []
    firsts = re.findall(r'finished: (\d+) passages', log.read_text()) if log.exists() else []
    st = {
        'passages': len(vals), 'missing': len(gold_ids - set(rs)),
        'transport_errors': sum(1 for d in vals if d.get('error') not in (None, 'invalid_json')),
        'invalid_json': sum(1 for d in vals if d.get('error') == 'invalid_json'),
        'retried_calls': sum(1 for d in vals if (d.get('attempts') or 1) > 1),
        'entities_returned': n_ent, 'unmatched': sum(len(d['unmatched']) for d in vals),
        'sec_per_passage_mean': round(statistics.mean(secs), 2) if secs else None,
        'sec_per_passage_median': round(statistics.median(secs), 2) if secs else None,
        'wall_s_first_pass': walls[0] if walls else None, 'first_pass_n': int(firsts[0]) if firsts else None,
        'prompt_tokens_mean': round(statistics.mean([d['prompt_tokens'] for d in vals if d.get('prompt_tokens')]), 0) if any(d.get('prompt_tokens') for d in vals) else None,
        'eval_tokens_mean': round(statistics.mean([d['eval_tokens'] for d in vals if d.get('eval_tokens')]), 0) if any(d.get('eval_tokens') for d in vals) else None,
    }
    if st['wall_s_first_pass'] and st['first_pass_n']:
        st['passages_per_hour'] = round(st['first_pass_n'] / st['wall_s_first_pass'] * 3600)
    return st, {i: sc.to_pred(d) for i, d in rs.items()}


def strip_final_dot(pred, toks):
    """Post-processing: drop a final '.' token from every span (dev3 keeps abbreviation dots outside the span)."""
    out = {}
    for i, d in pred.items():
        t = toks.get(i, [])
        nd = {'id': i}
        for lab in ('LOC', 'PER'):
            sp = set()
            for x in d.get(lab, []):
                s, e = x['s'], x['e']
                while e - s > 1 and e <= len(t) and t[e - 1] == '.':
                    e -= 1
                sp.add((s, e))
            nd[lab] = [{'s': s, 'e': e} for s, e in sorted(sp)]
        out[i] = nd
    return out


def propagate(pred, toks):
    """Post-processing: a name found once is marked at its other occurrences in the passage (same tokens, same label),
    where no predicted span already overlaps. LLMs often list a repeated name only once."""
    out = {}
    for i, d in pred.items():
        t = toks.get(i, [])
        taken = [(x['s'], x['e']) for lab in ('LOC', 'PER') for x in d.get(lab, [])]
        nd = {'id': i, 'LOC': list(d.get('LOC', [])), 'PER': list(d.get('PER', []))}
        for lab in ('LOC', 'PER'):
            for x in d.get(lab, []):
                seq = t[x['s']:x['e']]
                n = len(seq)
                if not n or not any(c.isalpha() for c in ''.join(seq)):
                    continue
                for k in range(0, len(t) - n + 1):
                    if t[k:k + n] == seq and not any(k < b and a < k + n for a, b in taken):
                        nd[lab].append({'s': k, 'e': k + n})
                        taken.append((k, k + n))
        out[i] = nd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--best3')
    ap.add_argument('--best2')
    ap.add_argument('--extra-combine', action='append', default=[])
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--pp', action='store_true', help='also score each run with the final-dot post-processing (suffix +pp)')
    ap.add_argument('--only', help='comma-separated run names to keep (after loading)')
    a = ap.parse_args()
    assert 'final_gold_v2' not in a.gold
    d = Path(a.dir)
    gold = sc.load(a.gold)
    gids = {r['id'] for r in gold}
    preds, stats = {}, {}
    for f in sorted((d / 'pred').glob('*.jsonl')):
        name = f.stem
        stats[name], preds[name] = run_stats(f, d / 'logs' / f'{name}.log', gids)
    if a.only:
        keep = set(a.only.split(','))
        preds = {k: v for k, v in preds.items() if k in keep}
    if a.pp:
        toks = {r['id']: r['tokens'] for r in gold}
        for n in list(preds):
            preds[n + '+dot'] = strip_final_dot(preds[n], toks)
            preds[n + '+pp'] = propagate(preds[n + '+dot'], toks)
    specs = []
    if a.best3:
        specs.append('vote:' + a.best3 + '@2')
    if a.best2:
        specs.append('union:' + a.best2)
    specs += a.extra_combine
    for s in specs:
        preds[s] = sc.combine(s, preds)
    out = {}
    for tag, sure in (('all', False), ('sure', True)):
        res = {}
        for n, p in preds.items():
            counts = sc.per_passage_counts(gold, p, accept_alt=True, sure_only=sure)
            s = sc.score(gold, p, counts=counts)
            ci = sc.bootstrap(gold, counts, a.bootstrap) if a.bootstrap else {}
            res[n] = {'|'.join(k): {'p': v[0], 'r': v[1], 'f1': v[2], 'ci_f1': ci.get(k, [None] * 3)[2] if ci else None,
                                    'ci_p': ci[k][0] if ci else None, 'ci_r': ci[k][1] if ci else None} for k, v in s.items()}
        out[tag] = res
        json.dump(res, open(d / f'scores_{tag}.json', 'w'), indent=1)
    json.dump(stats, open(d / 'run_stats.json', 'w'), indent=1)
    # markdown
    for tag in ('all', 'sure'):
        print(f'\n### dev3, {tag} spans, long form accepted (strict F1 [95% CI] / overlap F1)\n')
        print('| teacher | lang | LOC P | LOC R | LOC F1 strict | LOC F1 overlap | PER P | PER R | PER F1 strict | PER F1 overlap |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for n in preds:
            for lang in LANGS:
                r = out[tag][n]
                c = lambda lab, m: r[f'{lang}|{lab}|{m}']
                f = lambda x: f"{x['f1']:.3f}" + (f" [{x['ci_f1'][0]:.2f}-{x['ci_f1'][1]:.2f}]" if x['ci_f1'] else '')
                print(f"| {n} | {lang} | {c('LOC','strict')['p']:.3f} | {c('LOC','strict')['r']:.3f} | {f(c('LOC','strict'))} | {c('LOC','overlap')['f1']:.3f} "
                      f"| {c('PER','strict')['p']:.3f} | {c('PER','strict')['r']:.3f} | {f(c('PER','strict'))} | {c('PER','overlap')['f1']:.3f} |")
    print('\n### runs\n')
    keys = ['passages', 'missing', 'transport_errors', 'invalid_json', 'retried_calls', 'entities_returned', 'unmatched',
            'sec_per_passage_mean', 'sec_per_passage_median', 'wall_s_first_pass', 'passages_per_hour', 'prompt_tokens_mean', 'eval_tokens_mean']
    print('| run | ' + ' | '.join(keys) + ' |')
    print('|---|' + '---|' * len(keys))
    for n, s in stats.items():
        print(f'| {n} | ' + ' | '.join(str(s.get(k)) for k in keys) + ' |')


if __name__ == '__main__':
    main()
