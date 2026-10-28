"""Summary of the paired scores (dev3, editor bench main editions, test_v2) of a variant against its reference, seed by seed.
    python scripts/ner3_noT3_summary.py --scores DIR --new noT3 --ref lv4 [--json out.json]
Reads DIR/dev3.json, DIR/bench_s{1,2,3}.json, DIR/testv2_s{1,2,3}.json (outputs of the scripts of ner3_lv4_score_mac.sh). Prints the reference and
variant values per seed, the paired difference per seed with its 95 % interval, and the means. "sure" = the interval excludes 0.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--scores', required=True)
ap.add_argument('--new', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--json')
a = ap.parse_args()
S = Path(a.scores)
rows = []


def add(name, vals_ref, vals_new, diffs):
    rows.append({'measure': name, 'ref': vals_ref, 'new': vals_new, 'diff': [d[0] for d in diffs], 'ci': [d[1] for d in diffs]})


d3 = json.load(open(S / 'dev3.json'))
sysd = d3['systems']; pd = d3['paired']
def dkey(k): return f'{a.new}@s{k} - {a.ref}@s{k}'
add('dev3 selection', [sysd[f'{a.ref}@s{k}']['selection_score'] for k in (1, 2, 3)], [sysd[f'{a.new}@s{k}']['selection_score'] for k in (1, 2, 3)],
    [(pd[dkey(k)]['selection']['diff'], pd[dkey(k)]['selection']['ci']) for k in (1, 2, 3)])
for lg in ('la', 'es', 'it', 'all'):
    for lab in ('LOC', 'PER'):
        m = f'{lg}|{lab}|strict'
        add(f'dev3 {lg} {lab} F1', [sysd[f'{a.ref}@s{k}']['all'][m]['f1'] for k in (1, 2, 3)], [sysd[f'{a.new}@s{k}']['all'][m]['f1'] for k in (1, 2, 3)],
            [(pd[dkey(k)][m]['diff'], pd[dkey(k)][m]['ci']) for k in (1, 2, 3)])

B = {k: json.load(open(S / f'bench_s{k}.json')) for k in (1, 2, 3)}
def bval(k, sysn, lab, mode, grp, what):
    g = B[k]['scores'][f'{sysn}@s{k}'][lab][mode][grp]
    return g['recall']['r'] if what == 'r' else g['exhaustive'][what]
def bdiff(k, lab, mode, key):
    x = B[k]['paired_vs_ref']['diffs'][f'{a.new}@s{k}'][lab][mode][key]
    return x['diff'], x['ci95']
for name, lab, mode, grp, what in [('bench main LOC recall (editor, strict)', 'LOC_editor', 'strict', 'main', 'r'),
                                   ('bench main LOC F1 (v2 view, strict)', 'LOC_v2', 'strict', 'main', 'f1'),
                                   ('bench main PER recall (rule)', 'PER', 'rule', 'main', 'r'),
                                   ('bench main PER F1 (rule)', 'PER', 'rule', 'main', 'f1'),
                                   ('bench main la LOC recall (editor, strict)', 'LOC_editor', 'strict', 'main:lang=la', 'r'),
                                   ('bench main la LOC F1 (v2 view, strict)', 'LOC_v2', 'strict', 'main:lang=la', 'f1'),
                                   ('bench main la PER recall (rule)', 'PER', 'rule', 'main:lang=la', 'r'),
                                   ('bench main de LOC recall (editor, strict)', 'LOC_editor', 'strict', 'main:lang=de', 'r'),
                                   ('bench main de PER recall (rule)', 'PER', 'rule', 'main:lang=de', 'r')]:
    key = f'{grp}:{what}'
    try:
        add(name, [bval(k, a.ref, lab, mode, grp, what) for k in (1, 2, 3)], [bval(k, a.new, lab, mode, grp, what) for k in (1, 2, 3)],
            [bdiff(k, lab, mode, key) for k in (1, 2, 3)])
    except KeyError as e:
        print('missing', name, e)

T = {k: json.load(open(S / f'testv2_s{k}.json')) for k in (1, 2, 3)}
for sub in ('all', 'lang=la', 'lang=de', 'lang=fr', 'lang=en'):
    def tdiff(k):
        p = T[k]['paired_vs_ref']
        p = p.get(f'{a.new}@s{k}', p)
        x = p[sub] if sub in p else p.get('f1', {}).get(sub)
        return (x.get('diff', x.get('f1_diff')), x.get('ci95', x.get('f1_diff_ci95')))
    try:
        add(f'test_v2 LOC F1 {sub}', [T[k]['scores'][f'{a.ref}@s{k}'][sub]['f1'] for k in (1, 2, 3)],
            [T[k]['scores'][f'{a.new}@s{k}'][sub]['f1'] for k in (1, 2, 3)], [tdiff(k) for k in (1, 2, 3)])
    except (KeyError, AttributeError, TypeError) as e:
        print('missing', sub, e, json.dumps(T[1]['paired_vs_ref'])[:600])

for r in rows:
    sure = sum(1 for c in r['ci'] if c and (c[0] > 0 or c[1] < 0))
    r['sure'] = sure
    mr, mn = sum(r['ref']) / 3, sum(r['new']) / 3
    print(f"{r['measure']:44s} ref {' / '.join(f'{x:.4f}' for x in r['ref'])} (m {mr:.4f}) | new {' / '.join(f'{x:.4f}' for x in r['new'])} (m {mn:.4f}) | "
          f"diff {' ; '.join(f'{d:+.4f} [{c[0]:+.4f};{c[1]:+.4f}]' for d, c in zip(r['diff'], r['ci']))} | mean {sum(r['diff']) / 3:+.4f} | sure {sure}/3")
if a.json:
    json.dump(rows, open(a.json, 'w'), indent=1)
