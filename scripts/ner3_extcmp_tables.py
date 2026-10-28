"""External comparison: markdown tables of N/v2/external_comparison.md from the score files of that folder (no new prediction, no gold v2 / v3).

    python3 scripts/ner3_extcmp_tables.py --dir N/v2/external_comparison --teacher-dev3 N/v2/renner/train_v2/eval/scores_dev3.json > tables.md
"""
import argparse
import json
from pathlib import Path

NAMES = {
    'v2_xlmrxl_Sfull': '**RenNER v2 XL** (v2_xlmrxl_Sfull)', 'v2_mdeberta_Sfull': '**RenNER v2 base** (v2_mdeberta_Sfull)',
    'r3_mdeberta_HM': 'RenNER v1 (r3_mdeberta_HM)', 'teacher_gemma4_31b_pp': 'maître gemma-4-31B +pp (LLM, nos consignes)',
    'zs_flair_ner_de_large': 'flair ner-german-large', 'zs_flair_ner_en_large': 'flair ner-english-large', 'zs_flair_ner_multi': 'flair ner-multi',
    'zs_xlmr_conll03': 'XLM-R large CoNLL-03 (en)', 'zs_neolatiner': 'NeoLatiNER (XLM-R large, latin moderne)',
    'zs_davlan_xlmr_large': 'Davlan XLM-R large NER-hrl', 'zs_gliner_x_large': 'GLiNER-X large (seuil 0,7)',
    'zs_gliner_multi_v21': 'GLiNER multi v2.1 (seuil 0,7)', 'zs_gliner25_multi': 'GLiNER2.5 multi (seuil 0,7)',
    'zs_impresso_mmbert': 'impresso mmBERT HIPE', 'zs_impresso_hmbert': 'impresso hmBERT HIPE', 'zs_wikineural': 'WikiNEuRal multilingue',
    'zs_aendriu_it': 'BERT NER italien historique (aendriu)', 'zs_medieval_latin': 'medieval-latin-ner (ERCDiDip)',
    'zs_hmbench_hipe2020_de': 'hmBench flair HIPE-2020 de', 'zs_hmbench_hipe2020_fr': 'hmBench flair HIPE-2020 fr',
    'zs_hmbench_newseye_fr': 'hmBench flair NewsEye fr', 'zs_hmbench_topres_en': 'hmBench flair TopRes19th en',
    'zs_spacy_lg': 'spaCy *_core_*_lg 3.8 + LatinCy (par langue)', 'zs_grandtour': 'Grand Tour NER (spaCy, allemand moderne)',
    'zs_dell_en': 'Dell historical newspaper NER', 'zs_gliner_large_v25': 'GLiNER large v2.5', 'zs_gliner_x_base': 'GLiNER-X base',
    'zs_gliner2_multi': 'GLiNER2 multi', 'zs_nuner_zero': 'NuNER Zero', 'zs_sauerkraut_gliner': 'SauerkrautLM-GLiNER',
}
OURS = ['v2_xlmrxl_Sfull', 'v2_mdeberta_Sfull', 'r3_mdeberta_HM']


def n(x):
    return NAMES.get(x, x)


def f(x, k=3):
    return '–' if x is None else f'{x:.{k}f}'.replace('.', ',')


def ci(c, k=3):
    return f'[{f(c[0], k)}–{f(c[1], k)}]'


def dif(d, k=3):
    return f"{'+' if d['diff'] >= 0 else '−'}{f(abs(d['diff']), k)} [{f(d['ci'][0] if 'ci' in d else d['ci95'][0], k)} ; {f(d['ci'][1] if 'ci' in d else d['ci95'][1], k)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--teacher-dev3', required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    out = []
    # ------------------------------------------------------------ dev3
    d3 = json.loads((D / 'scores_dev3_all.json').read_text())
    t3 = json.loads(Path(a.teacher_dev3).read_text())['systems']['teacher_gemma4_31b_pp']
    sysd = dict(d3['systems'])
    sysd['teacher_gemma4_31b_pp'] = t3
    order = sorted(sysd, key=lambda s: -sysd[s]['selection_score'])
    out.append('### T-dev3. dev3 (358 passages es/it/la), F1 strict, tous les empans\n')
    out.append('| système | score de choix | es LOC | es PER | it LOC | it PER | la LOC | la PER | tout LOC [IC] | tout PER [IC] | tout LOC chev. | tout PER chev. |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in order:
        x = sysd[s]['all']
        g = lambda k: x[k]['f1']
        out.append(f"| {n(s)} | {f(sysd[s]['selection_score'])} | " + ' | '.join(f(g(f'{lg}|{lab}|strict')) for lg in ('es', 'it', 'la') for lab in ('LOC', 'PER'))
                   + f" | {f(g('all|LOC|strict'))} {ci(x['all|LOC|strict']['f1_ci'])} | {f(g('all|PER|strict'))} {ci(x['all|PER|strict']['f1_ci'])}"
                   + f" | {f(g('all|LOC|overlap'))} | {f(g('all|PER|overlap'))} |")
    out.append('\n### T-dev3-paired. Écarts appariés sur dev3 (F1 strict ; même tirage bootstrap)\n')
    out.append('| comparaison | score de choix | tout LOC | tout PER | la LOC | la PER |')
    out.append('|---|---|---|---|---|---|')
    for k, v in d3['paired'].items():
        sel = v.get('selection') or v.get('selection_score')
        out.append(f"| {k} | {dif(sel) if sel else '–'} | {dif(v['all|LOC|strict'])} | {dif(v['all|PER|strict'])} | {dif(v['la|LOC|strict'])} | {dif(v['la|PER|strict'])} |")
    # ------------------------------------------------------------ bench
    b = json.loads((D / 'scores_bench_refXL.json').read_text())
    sc = b['scores']
    pv = b['paired_vs_ref']['diffs']

    def br(s, view, mode, grp, key='recall'):
        e = sc[s][view][mode].get(grp)
        return None if not e else e[key]

    border = sorted(sc, key=lambda s: -(br(s, 'PER', 'rule', 'main')['r'] + br(s, 'LOC_editor', 'strict', 'main')['r']))
    out.append('\n### T-bench. Banc des éditeurs (17 956 passages) : rappel (score principal), F1 là où l\'édition est exhaustive\n')
    out.append('| système | LOC éd. R [IC] | R de | R la | R fr | R en | F1 LOC éd. (exh.) | F1 LOC v2 | PER règle R [IC] | R de | R la | R fr | R en | R es | R it | F1 PER règle (exh.) | XL − système : LOC R | XL − système : PER R |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|')
    for s in border:
        lr = br(s, 'LOC_editor', 'strict', 'main')
        pr = br(s, 'PER', 'rule', 'main')
        row = [f"{f(lr['r'] * 100, 1)} {ci([x * 100 for x in lr['r_ci95']], 0)}"]
        for lg in ('de', 'la', 'fr', 'en'):
            e = br(s, 'LOC_editor', 'strict', f'main:lang={lg}')
            row.append(f(e['r'] * 100, 1) if e else '–')
        row.append(f(sc[s]['LOC_editor']['strict']['main']['exhaustive']['f1'] * 100, 1))
        row.append(f(sc[s]['LOC_v2']['strict']['main']['exhaustive']['f1'] * 100, 1))
        row.append(f"{f(pr['r'] * 100, 1)} {ci([x * 100 for x in pr['r_ci95']], 0)}")
        for lg in ('de', 'la', 'fr', 'en', 'es', 'it'):
            e = br(s, 'PER', 'rule', f'main:lang={lg}')
            row.append(f(e['r'] * 100, 1) if e else '–')
        row.append(f(sc[s]['PER']['rule']['main']['exhaustive']['f1'] * 100, 1))
        if s in pv:
            dl = pv[s]['LOC_editor']['strict']['main:r']
            dp = pv[s]['PER']['rule']['main:r']
            neg = lambda d: {'diff': -d['diff'], 'ci': [-d['ci95'][1], -d['ci95'][0]]}
            row += [dif({'diff': neg(dl)['diff'] * 100, 'ci': [x * 100 for x in neg(dl)['ci']]}, 1),
                    dif({'diff': neg(dp)['diff'] * 100, 'ci': [x * 100 for x in neg(dp)['ci']]}, 1)]
        else:
            row += ['(réf.)', '(réf.)']
        out.append(f'| {n(s)} | ' + ' | '.join(row) + ' |')
    # ------------------------------------------------------------ test_v2
    t = json.loads((D / 'scores_testv2_refXL.json').read_text())
    ts = t['scores']
    tp = t['paired_vs_ref']
    out.append('\n### T-testv2. test_v2 (15 234 passages, 6 éditions), LOC F1 strict, or corrigé ; PER = or des éditeurs (secondaire)\n')
    out.append('| système | P | R | F1 [IC] | hors Königsfelden | de | en | fr | it | la | PER (éditeurs) | XL − système (F1 tout) |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for s in sorted(ts, key=lambda s: -ts[s]['all']['f1']):
        x = ts[s]
        g = lambda k: f(x[k]['f1']) if k in x else '–'
        d = tp.get(s, {}).get('all')
        dd = dif({'diff': -d['diff'], 'ci': [-d['ci95'][1], -d['ci95'][0]]}) if d else '(réf.)'
        out.append(f"| {n(s)} | {f(x['all']['p'])} | {f(x['all']['r'])} | {f(x['all']['f1'])} {ci(x['all']['f1_ci95'])} | {g('no_koenigsfelden')} | "
                   + ' | '.join(g(f'lang={lg}') for lg in ('de', 'en', 'fr', 'it', 'la')) + f" | {g('per_editor')} | {dd} |")
    # ------------------------------------------------------------ public sets
    for fn, title in (('scores_pub.json', 'tous les enregistrements'), ('scores_pub_clean.json', 'sans les enregistrements qui partagent ≥ 5 % de 8-grammes avec notre entraînement')):
        p = json.loads((D / fn).read_text())
        groups = [('neolatiner/test', 'NeoLatiNER test (la, XVIe-XVIIIe s.)'), ('evalatin2026/test', 'EvaLatin 2026 NER test (la classique)'),
                  ('eneide/DZ', 'ENEIDE Zibaldone (it, 1817-1832, argent)'), ('eneide/AMD', 'ENEIDE Moro (it, XXe s., argent)'),
                  ('grandtour/test', 'Grand Tour NER test (de, XVIIe s.)')]
        for lab in ('LOC', 'PER'):
            out.append(f'\n### T-pub-{lab} ({title}). F1 strict / trim / chevauchement\n')
            ng = [p['v2_xlmrxl_Sfull']['groups'][g].get(lab, {}).get('n_gold', 0) for g, _ in groups]
            out.append('| système | ' + ' | '.join(f'{t_} ({k})' for (_, t_), k in zip(groups, ng)) + ' |')
            out.append('|---|' + '---|' * len(groups))
            for s in sorted(p, key=lambda s: -sum((p[s]['groups'][g].get(lab, {}).get('trim', {}).get('f1') or 0) for g, _ in groups)):
                cells = []
                for g, _ in groups:
                    e = p[s]['groups'][g].get(lab)
                    cells.append(f"{f(e['strict']['f1'])} / **{f(e['trim']['f1'])}** / {f(e['overlap']['f1'])}" if e else '–')
                out.append(f'| {n(s)} | ' + ' | '.join(cells) + ' |')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
