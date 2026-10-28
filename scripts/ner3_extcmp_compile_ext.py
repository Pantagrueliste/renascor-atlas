"""External comparison: collect the strict F1 already computed by ratlas.ner.train (end of training) and by the v1 zero-shot runs on the external
test sets of hist_gold/test (sets ext_hipe, ext_hist, ext_charters of scripts/ner_sets.sh). No new prediction.

    python3 scripts/ner3_extcmp_compile_ext.py --runs-v1 N/runs --runs-v2 N/v2/renner/train_v2/hypatia/runs --out ext_existing.json --md ext_existing.md
"""
import argparse
import json
from pathlib import Path

OURS = ['v2_xlmrxl_Sfull', 'v2_mdeberta_Sfull', 'r3_mdeberta_HM']
SKIP = ('_1200', '_llm', '_hi')
GROUPS = [
    ('HIPE-2022 (presse, XIXe-XXe s.)', 'ext_hipe', ['hipe2022/hipe2020/de', 'hipe2022/hipe2020/fr', 'hipe2022/hipe2020/en', 'hipe2022/letemps/fr',
                                                   'hipe2022/newseye/de', 'hipe2022/newseye/fr', 'hipe2022/sonar/de', 'hipe2022/topres19th/en',
                                                   'hipe2022/ajmc/de', 'hipe2022/ajmc/fr', 'hipe2022/ajmc/en', 'hipe2022/newseye/fi', 'hipe2022/newseye/sv']),
    ('Allemand moderne précoce (test seul)', 'ext_hist', ['earlymodern_german/BE/de', 'earlymodern_german/LP/de', 'earlymodern_german/AD/de', 'earlymodern_german/NUL/de']),
    ('Latin', 'ext_hist', ['latin_lasla/Cicero_PhilippicaOratio_CicPhi01/la', 'latin_lasla/Juvenal_sat_1_3/la', 'latin_lasla/Tacitus_TacHistoriae_TacHist1/la',
                           'latin_herodotos/GWtest/la']),
    ('Autres historiques', 'ext_hist', ['kind/degasperi/it', 'geoedda/edda/fr', 'globalise/voc/nl', 'europeana/enp_FR.bnf/fr', 'europeana/enp_DE.lft/de',
                                        'europeana/enp_DE.sbb/de', 'europeana/enp_NL.kb/nl', 'reichsanzeiger/reichsanzeiger/de', 'sdhk_old_swedish/charters-expert-gold/sv']),
    ('Chartes médiévales', 'ext_charters', ['medieval_charters/CBMA/la', 'medieval_charters/CBMA/fr', 'medieval_charters/CDBE/fr', 'medieval_charters/CODCAR/es',
                                            'medieval_charters/CODEA/es', 'medieval_charters/CORHEN/es', 'medieval_charters/ESLONZA/es', 'medieval_charters/ESLONZA/la',
                                            'medieval_charters/NESLE/fr', 'medieval_charters/NESLE/la']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-v1', required=True)
    ap.add_argument('--runs-v2', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args()
    runs = {}
    for name in OURS:
        base = Path(a.runs_v2 if name.startswith('v2_') else a.runs_v1) / name / 'metrics.json'
        runs[name] = json.loads(base.read_text())
    for p in sorted(Path(a.runs_v1).glob('zs_*/metrics.json')):
        n = p.parent.name
        if any(s in n for s in SKIP):
            continue
        runs[n] = json.loads(p.read_text())
    res = {}
    for name, d in runs.items():
        out = {}
        for _, setname, subs in GROUPS:
            s = d.get('sets', {}).get(setname)
            if not s:
                continue
            for sub in subs:
                for lab in ('LOC', 'PER'):
                    e = s.get(lab, {}).get('by_edition', {}).get(sub)
                    if e and e.get('n_gold', 0) >= 20:
                        out[f'{sub}|{lab}'] = {'f1': e['f1'], 'p': e['p'], 'r': e['r'], 'n_gold': e['n_gold']}
        res[name] = {'threshold': d.get('selected_threshold'), 'scores': out}
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.md:
        L = []
        zs = [n for n in res if n.startswith('zs_')]
        for title, _, subs in GROUPS:
            L.append(f'\n#### {title}\n')
            L.append('| jeu | label | or | XL S-full | mDeBERTa S-full | v1 (r3_mdeberta_HM) | meilleur hors boîte (F1) | 2e hors boîte |')
            L.append('|---|---|---:|---:|---:|---:|---|---|')
            for sub in subs:
                for lab in ('LOC', 'PER'):
                    k = f'{sub}|{lab}'
                    if k not in res['v2_xlmrxl_Sfull']['scores']:
                        continue
                    ng = res['v2_xlmrxl_Sfull']['scores'][k]['n_gold']
                    cells = [f"{res[n]['scores'][k]['f1']:.3f}" if k in res[n]['scores'] else '–' for n in OURS]
                    best = sorted(((res[n]['scores'][k]['f1'], n) for n in zs if k in res[n]['scores']), reverse=True)
                    b1 = f'{best[0][1][3:]} {best[0][0]:.3f}' if best else '–'
                    b2 = f'{best[1][1][3:]} {best[1][0]:.3f}' if len(best) > 1 else '–'
                    L.append(f"| {sub.replace('hipe2022/', '').replace('medieval_charters/', '')} | {lab} | {ng} | " + ' | '.join(cells) + f' | {b1} | {b2} |')
        Path(a.md).write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
