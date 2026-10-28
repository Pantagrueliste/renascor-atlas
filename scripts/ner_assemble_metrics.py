"""Assemble RQ/derived/atlas/ner/metrics_v1.md from the metrics of all runs (tables from ratlas.ner.report + numbers read from the same files).

   python scripts/ner_assemble_metrics.py [RUNS_DIR] [OUT_MD]
"""
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ratlas.ner import report as rp  # noqa: E402

NER = Path('/Users/clem/Projects/Papers/RQ/derived/atlas/ner')
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else NER / 'runs'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else NER / 'metrics_v1.md'
TAB = NER / 'tables_v1'

runs = rp.load_runs(RUNS)
rp.main(['--runs', str(RUNS), '--out', str(TAB)])
tab = lambda n: (TAB / n).read_text() if (TAB / n).exists() else '(not available)'


def v(run, s, t, k='f1', *path):
    return rp.get(runs[run], s, t, 'overall', k) if not path else rp.get(runs[run], s, t, *path)


def f3(x):
    return '-' if x is None else f'{x:.3f}'


def boot(name, task='LOC'):
    b = json.loads((NER / 'bootstrap_test.json').read_text())['runs'] if (NER / 'bootstrap_test.json').exists() else {}
    x = b.get(name, {}).get(task)
    return None if not x else x


# --- selection of systems for the headline tables --------------------------------------------------------------
HEAD = [
    ('Dictionary', ['gazetteer_ours']),
    ('Off the shelf, token classifiers', ['zs_davlan_xlmr_large', 'zs_wikineural', 'zs_impresso_mmbert', 'zs_impresso_hmbert', 'zs_hmbench_hipe2020_fr', 'zs_flair_ner_en_large', 'zs_flair_ner_de_large', 'zs_flair_ner_multi', 'zs_dalembert_fr', 'zs_medieval_latin']),
    ('Off the shelf, GLiNER family (label prompts)', ['zs_gliner_x_large', 'zs_gliner_x_base', 'zs_gliner_multi_v21_location', 'zs_gliner_large_v25', 'zs_gliner2_multi', 'zs_gliner25_multi', 'zs_nuner_zero', 'zs_sauerkraut_gliner']),
    ('Fine-tuned on our data only (R2)', ['ft_xlmr_base_ours', 'ft_hmbert64k_ours', 'ft_hmteams_ours', 'ft_mmbert_base_ours', 'ft_eurobert210_ours', 'ft_mdeberta_ours', 'ft_xlmr_large_ours', 'ft_eurobert610_ours', 'ft_gliner_multi_ours_v2']),
    ('Fine-tuned on our data + external data (R3)', ['r3_xlmr_base_H', 'r3_xlmr_base_HM', 'r3_hmteams_H', 'r3_hmteams_HM', 'r3_mmbert_base_H', 'r3_mmbert_base_HM', 'r3_mdeberta_H', 'r3_mdeberta_HM', 'r3_xlmr_large_H', 'r3_hmteams_ext_stage1', 'r3_hmteams_twostage']),
]


def headline():
    rows = []
    for group, names in HEAD:
        rows.append([f'**{group}**'] + [''] * 11)
        for n in names:
            if n not in runs:
                continue
            r = runs[n]
            b = boot(n)
            ci = f" [{b['micro_ci95'][0]:.2f}-{b['micro_ci95'][1]:.2f}]" if b else ''
            rows.append([f'`{n}`', rp.model_of(n, r)[:48], f3(v(n, 'dev', 'LOC')), f3(v(n, 'test', 'LOC', 'p')), f3(v(n, 'test', 'LOC', 'r')),
                         f3(v(n, 'test', 'LOC')) + ci, f3(rp.get(r, 'test', 'LOC', 'macro_edition_f1')), f3(v(n, 'dev', 'PER')), f3(v(n, 'test', 'PER')),
                         f3(v(n, 'ext_hipe', 'LOC')), f3(v(n, 'ext_hist', 'LOC')), f3(v(n, 'ext_charters', 'LOC'))])
    return rp.md(['system', 'model', 'LOC dev F1', 'LOC test P', 'LOC test R', 'LOC test F1', 'LOC test macro-ed F1', 'PER dev F1', 'PER test F1', 'HIPE LOC F1',
                  'hist LOC F1', 'charters LOC F1'], rows)


def best(prefix_filter, key=('test', 'LOC')):
    c = [(v(n, *key), n) for n in runs if prefix_filter(n) and v(n, *key) is not None]
    return max(c) if c else (None, None)


CUR = {n: runs[n] for _, names in HEAD for n in names if n in runs}
CUR.update({n: runs[n] for n in ('zs_hmbench_hipe2020_de', 'zs_hmbench_newseye_fr', 'zs_hmbench_newseye_de', 'zs_hmbench_topres_en', 'zs_flair_ner_en_large', 'zs_flair_ner_de_large') if n in runs})
EDS6 = rp.TEST_ED_LOC
def bootstrap_table():
    fp = NER / 'bootstrap_test.json'
    if not fp.exists():
        return '(not available)'
    b = json.loads(fp.read_text())
    ref = b['reference']
    rows = []
    for n, x in sorted(b['runs'].items(), key=lambda kv: -kv[1]['LOC']['micro_f1']):
        L, P = x['LOC'], x['PER']
        rows.append([f'`{n}`', f"{L['micro_f1']:.3f} [{L['micro_ci95'][0]:.3f}, {L['micro_ci95'][1]:.3f}]", f"{L['diff_micro']:+.3f} [{L['diff_micro_ci95'][0]:+.3f}, {L['diff_micro_ci95'][1]:+.3f}]",
                     f"{L['macro_edition_f1']:.3f} [{L['macro_ci95'][0]:.3f}, {L['macro_ci95'][1]:.3f}]", f"{P['micro_f1']:.3f} [{P['micro_ci95'][0]:.3f}, {P['micro_ci95'][1]:.3f}]",
                     f"{P['diff_micro']:+.3f} [{P['diff_micro_ci95'][0]:+.3f}, {P['diff_micro_ci95'][1]:+.3f}]"])
    return rp.md(['run', 'LOC test F1', f'LOC diff vs {ref}', 'LOC macro-edition F1 [95% CI]', 'PER test F1 [95% CI]', f'PER diff vs {ref}'], rows)


text = (NER / 'metrics_v1_narrative.md').read_text() if (NER / 'metrics_v1_narrative.md').exists() else ''
parts = {
    'HEADLINE': headline(), 'MAIN': tab('main.md'), 'EDITION_LOC': rp.edition_table(CUR, 'LOC'),
    'EDITION_PER': rp.edition_table(CUR, 'PER', ['elan_crenum', 'elan_fnvr', 'hainhofer_reisen', 'kaiserin_eleonora_magdalena', 'koenigsfelden', 'sozinianische_briefwechsel']),
    'LANG_LOC': rp.lang_table(CUR, 'LOC'), 'LANG_PER': rp.lang_table(CUR, 'PER'), 'GAP': tab('gap.md'), 'MIX': tab('mix.md'),
    'HIPE_LOC': rp.ext_table(CUR, 'ext_hipe', 'LOC', rp.HIPE_SUBSETS), 'HIPE_PER': rp.ext_table(CUR, 'ext_hipe', 'PER', rp.HIPE_SUBSETS),
    'HIST_LOC': rp.ext_table(CUR, 'ext_hist', 'LOC'), 'HIST_PER': tab('hist_per.md'), 'CHARTERS': rp.ext_table(CUR, 'ext_charters', 'LOC'), 'PUBLISHED_HIPE': tab('published_hipe.md'), 'SEEDS': tab('seeds.md'), 'LLM360': tab('llm_llm.md'),
    'LLM1200': tab('llm_1200.md'), 'BOOTSTRAP': bootstrap_table(),
}
for k, val in parts.items():
    text = text.replace('{{' + k + '}}', val)
OUT.write_text(text)
print('wrote', OUT, len(text), 'characters')
