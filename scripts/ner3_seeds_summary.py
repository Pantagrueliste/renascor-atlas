"""RenNER seeds 1-3: mean +- sd (sample sd, n=3) of each metric, and the paired differences within each seed (bootstrap CIs from the
scorers). Reads the outputs of scripts/ner3_seeds_score_mac.sh seeds.  python scripts/ner3_seeds_summary.py SCORES_DIR > summary.md
"""
import json
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1])
SYS = ['r3_mdeberta_HM', 'v2_mdeberta_Sfull', 'v21_mdeberta_lc25']
LAB = {'r3_mdeberta_HM': 'v1 (r3_mdeberta_HM)', 'v2_mdeberta_Sfull': 'v2 (v2_mdeberta_Sfull)', 'v21_mdeberta_lc25': 'v2.1 (v21_mdeberta_lc25)'}
dev = json.load(open(D / 'dev3.json'))
tv = {k: json.load(open(D / f'testv2_s{k}.json')) for k in (1, 2, 3)}
tvl = {k: json.load(open(D / f'testv2_lc_s{k}.json')) for k in (1, 2, 3)}
bm = {k: json.load(open(D / f'bench_main_s{k}.json')) for k in (1, 2, 3)}
bl = {k: json.load(open(D / f'bench_lc_s{k}.json')) for k in (1, 2, 3)}


def dev3_m(n):
    r = dev['systems'][n]
    out = {'dev3 sel.': r['selection_score']}
    for key, lab in (('all|LOC|strict', 'dev3 LOC'), ('all|PER|strict', 'dev3 PER'), ('la|LOC|strict', 'dev3 la LOC'), ('la|PER|strict', 'dev3 la PER')):
        out[lab] = r['all'][key]['f1']
    return out


def bench_m(k, n):
    s = bm[k]['scores'][n]
    return {'bench LOC R': s['LOC_editor']['strict']['main']['recall']['r'],
            'bench F1 LOC v2': s['LOC_v2']['strict']['main']['exhaustive']['f1'],
            'bench PER R': s['PER']['rule']['main']['recall']['r'],
            'bench F1 PER': s['PER']['rule']['main']['exhaustive']['f1'],
            'bench fr LOC R': s['LOC_editor']['strict']['main:lang=fr']['recall']['r'],
            'bench fr PER R': s['PER']['rule']['main:lang=fr']['recall']['r']}


def tv_m(k, n):
    s = tv[k]['scores'][n]
    return {'test_v2': s['all']['f1'], 'test_v2 w/o Kf': s['no_koenigsfelden']['f1'], 'test_v2 fr': s['lang=fr']['f1'], 'test_v2 la': s['lang=la']['f1']}


vals = {}
for n in SYS:
    for k in (1, 2, 3):
        m = {}
        m.update(dev3_m(f'{n}@s{k}'))
        m.update(bench_m(k, f'{n}@s{k}'))
        m.update(tv_m(k, f'{n}@s{k}'))
        vals[(n, k)] = m
METRICS = list(vals[(SYS[0], 1)])


def f(x):
    return f'{x:.3f}'.replace('.', ',')


def fs(x):
    return (f'{x:+.3f}').replace('.', ',').replace('-', '−')


out = {'per_seed': {f'{n}@s{k}': v for (n, k), v in vals.items()}, 'mean_sd': {}, 'paired': {}}
print('## Moyenne ± écart-type (3 graines)\n')
print('| mesure | ' + ' | '.join(LAB[n] for n in SYS) + ' |')
print('|---|' + '---:|' * len(SYS))
for met in METRICS:
    row = []
    for n in SYS:
        xs = [vals[(n, k)][met] for k in (1, 2, 3)]
        mu, sd = st.mean(xs), st.stdev(xs)
        out['mean_sd'].setdefault(n, {})[met] = {'mean': mu, 'sd': sd, 'seeds': xs}
        row.append(f'{f(mu)} ± {f(sd)}')
    print(f'| {met} | ' + ' | '.join(row) + ' |')
print('\n## Par graine\n')
print('| mesure | ' + ' | '.join(f'{LAB[n].split()[0]} s{k}' for n in SYS for k in (1, 2, 3)) + ' |')
print('|---|' + '---:|' * 9)
for met in METRICS:
    print(f'| {met} | ' + ' | '.join(f(vals[(n, k)][met]) for n in SYS for k in (1, 2, 3)) + ' |')


def ci(d, a='ci'):
    return f"{fs(d['diff'])} [{fs(d[a][0])} ; {fs(d[a][1])}]"


def sure(d, a='ci'):
    return d[a][0] > 0


print('\n## Écarts appariés, dans chaque graine (bootstrap)\n')
COMP = [('v2_mdeberta_Sfull', 'r3_mdeberta_HM'), ('v21_mdeberta_lc25', 'r3_mdeberta_HM'), ('v21_mdeberta_lc25', 'v2_mdeberta_Sfull')]
for a, b in COMP:
    print(f'\n### {LAB[a]} − {LAB[b]}\n')
    print('| mesure | graine 1 | graine 2 | graine 3 | IC > 0 |')
    print('|---|---|---|---|---:|')
    rows = {}
    for k in (1, 2, 3):
        pd = dev['paired'][f'{a}@s{k} - {b}@s{k}']
        rows.setdefault('dev3 sel.', []).append((ci(pd['selection']), sure(pd['selection'])))
        rows.setdefault('dev3 LOC', []).append((ci(pd['all|LOC|strict']), sure(pd['all|LOC|strict'])))
        rows.setdefault('dev3 PER', []).append((ci(pd['all|PER|strict']), sure(pd['all|PER|strict'])))
        rows.setdefault('dev3 la LOC', []).append((ci(pd['la|LOC|strict']), sure(pd['la|LOC|strict'])))
        rows.setdefault('dev3 la PER', []).append((ci(pd['la|PER|strict']), sure(pd['la|PER|strict'])))
        T = (tv if b == 'r3_mdeberta_HM' else tvl)[k]['paired_vs_ref'][f'{a}@s{k}']
        for key, lab in (('all', 'test_v2'), ('no_koenigsfelden', 'test_v2 w/o Kf'), ('lang=fr', 'test_v2 fr'), ('lang=la', 'test_v2 la')):
            rows.setdefault(lab, []).append((ci(T[key], 'ci95'), sure(T[key], 'ci95')))
        Bd = (bm if b == 'r3_mdeberta_HM' else bl)[k]['paired_vs_ref']['diffs'][f'{a}@s{k}']
        for v, mo, key, lab in (('LOC_editor', 'strict', 'main:r', 'bench LOC R'), ('LOC_v2', 'strict', 'main:f1', 'bench F1 LOC v2'),
                                ('PER', 'rule', 'main:r', 'bench PER R'), ('PER', 'rule', 'main:f1', 'bench F1 PER')):
            x = Bd[v][mo][key]
            rows.setdefault(lab, []).append((ci(x, 'ci95'), sure(x, 'ci95')))
    for lab, xs in rows.items():
        out['paired'].setdefault(f'{a} - {b}', {})[lab] = [x[0] for x in xs]
        print(f'| {lab} | ' + ' | '.join(x[0] for x in xs) + f' | {sum(x[1] for x in xs)}/3 |')
json.dump(out, open(D / 'seeds_summary.json', 'w'), ensure_ascii=False, indent=1)
