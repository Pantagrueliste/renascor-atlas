"""Tables of the test_v2 report (French) from scores_test_v2.json and the gold file: writes ``test_v2_counts.json`` and the markdown tables that
``scores_test_v2.md`` and ``LISEZMOI.md`` embed.

    python scripts/ner2_report_testv2.py --gold test_v2.jsonl --scores scores_test_v2.json --old-stats test_v2_stats.json --cand-summary cand_summary.json \
        --selfcheck selfcheck.json --systems systems.tsv --out-dir DIR

The script only formats numbers; the prose of the two documents is written around its output.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas.ner.ner2_cand import family                                  # noqa: E402
from ratlas.ner.spans import tags_to_spans                               # noqa: E402

EDS = ['elan_crenum', 'hainhofer_reisen', 'kaiserin_eleonora_magdalena', 'koenigsfelden', 'reed', 'sozinianische_briefwechsel']
LANGS = ['de', 'en', 'fr', 'it', 'la', 'und']
SHOW = ['r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_xlmr_large_H', 'r3_hmteams_HM', 'ft_xlmr_large_ours', 'ft_hmteams_ours', 'zs_flair_ner_en_large',
        'zs_flair_ner_de_large', 'zs_gliner_x_large', 'zs_davlan_xlmr_large', 'zs_impresso_mmbert', 'zs_llm_qwen35_9b']


def n3(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',')


def cell(o, d=3):
    lo, hi = o['f1_ci95']
    return f"{n3(o['f1'], d)} [{n3(lo, d)}–{n3(hi, d)}]"


def table(head, rows, align=None):
    align = align or ['l'] + ['r'] * (len(head) - 1)
    sep = ['---:' if a == 'r' else ':---' for a in align]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(sep) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def counts(gold_path: Path) -> dict:
    ed = defaultdict(Counter)
    lg = defaultdict(Counter)
    tot = Counter()
    n_pass_changed = Counter()
    with open(gold_path, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            if not r['loc_valid'] or r['edition'] == 'elan_fnvr':
                continue
            e, l = r['edition'], r.get('lang') or 'und'
            editor = tags_to_spans(r['ner_tags_loc_editor'])
            added = len(r['loc_added'])
            removed = len(r['loc_removed'])
            amb_cand = sum(1 for x in r['loc_ambiguous'] if x['what'] == 'candidate')
            amb_ed = sum(1 for x in r['loc_ambiguous'] if x['what'] == 'editor_span')
            mask = set(r['loc_mask'])
            kept = [sp for sp in map(tuple, r['loc_gold_v2']) if not any(t in mask for t in range(sp[0], sp[1]))]
            for d in (ed[e], lg[l], tot):
                d['passages'] += 1
                d['words'] += r['words']
                d['editor_spans'] += len(editor)
                d['added'] += added
                d['removed'] += removed
                d['ambiguous_candidates'] += amb_cand
                d['ambiguous_editor_spans'] += amb_ed
                d['gold_v2_scored'] += len(kept)
                d['masked_tokens'] += len(mask)
                d['passages_changed_by_judgements'] += bool(added or removed or r['loc_ambiguous'])
                d['passages_repaired_text'] += bool(r['changed'])
    return {'edition': {k: dict(v) for k, v in ed.items()}, 'language': {k: dict(v) for k, v in lg.items()}, 'total': dict(tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--old-stats', type=Path, required=True)
    ap.add_argument('--cand-summary', type=Path, required=True)
    ap.add_argument('--selfcheck', type=Path, required=True)
    ap.add_argument('--systems', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    sc = json.loads(a.scores.read_text(encoding='utf-8'))
    S = sc['scores']
    names = sorted(S, key=lambda n: -S[n]['v2']['all']['f1'])
    info = {}
    with open(a.systems, encoding='utf-8') as fh:
        hdr = fh.readline().rstrip('\n').split('\t')
        for line in fh:
            row = dict(zip(hdr, line.rstrip('\n').split('\t')))
            info[row['system']] = row
    cnt = counts(a.gold)
    (a.out_dir / 'test_v2_counts.json').write_text(json.dumps(cnt, indent=1, ensure_ascii=False), encoding='utf-8')
    md = {}

    def v1_of(n):
        if 'v1' in S[n]:
            return S[n]['v1']['all']['f1'], ''
        x = info.get(n, {}).get('v1_test_LOC_F1')
        return (float(x), '†') if x else (None, '')

    # --- table 1: all systems --------------------------------------------------------------------------------------------
    rows = []
    for n in names:
        v2 = S[n]['v2']
        v1, mark = v1_of(n)
        star = '*' if n == 'zs_llm_qwen35_9b' else ''
        rows.append([f'`{n}`{star}', {'ours': 'nous', 'offshelf': 'hors boîte', 'llm': 'LLM'}[family(n)], n3(v2['all']['p']), n3(v2['all']['r']), cell(v2['all']),
                     cell(v2['no_koenigsfelden']), f"{n3(v2['all']['macro_edition_f1'])} [{n3(v2['all']['macro_ci95'][0])}–{n3(v2['all']['macro_ci95'][1])}]",
                     (n3(v1) + mark) if v1 is not None else '', (n3(v2['all']['f1'] - v1) if v1 is not None else '')])
    md['main'] = table(['système', 'famille', 'P', 'R', 'F1 v2 [IC 95 %]', 'F1 v2 hors Königsfelden [IC]', 'F1 macro-éditions [IC]', 'F1 v1', 'v2 − v1'], rows,
                       ['l', 'l', 'r', 'r', 'r', 'r', 'r', 'r', 'r'])

    # --- table 2: decomposition v1 -> v2a -> v2 ------------------------------------------------------------------------------------
    rows = []
    for n in names:
        v1, mark = v1_of(n)
        a_ = S[n]['v2a']['all']['f1']
        b_ = S[n]['v2']['all']['f1']
        g = sc['paired_gain'][n]
        rows.append([f'`{n}`', (n3(v1) + mark) if v1 is not None else '', n3(a_), n3(b_), (n3(a_ - v1, 3) if v1 is not None else ''), n3(b_ - a_),
                     f"[{n3(g['gain_ci95'][0])}–{n3(g['gain_ci95'][1])}]"])
    md['decomp'] = table(['système', 'F1 v1', 'F1 v2a (texte réparé, or éditeurs)', 'F1 v2', 'réparation (v2a − v1)', 'or corrigé (v2 − v2a)', 'IC 95 % de v2 − v2a'], rows)

    # --- table 3/4: language / edition for the shown systems ---------------------------------------------------------------------
    rows = []
    for n in SHOW:
        rows.append([f'`{n}`'] + [cell(S[n]['v2'][f'lang={l}'], 3) for l in ['de', 'en', 'fr', 'it', 'la']])
    md['lang'] = table(['système'] + [f"{l} (n={cnt['language'][l]['gold_v2_scored']})" for l in ['de', 'en', 'fr', 'it', 'la']], rows)
    rows = []
    for n in SHOW:
        rows.append([f'`{n}`'] + [cell(S[n]['v2'][f'edition={e}'], 3) for e in EDS])
    md['edition'] = table(['système'] + [f"{e.split('_')[0]} (n={cnt['edition'][e]['gold_v2_scored']})" for e in EDS], rows)
    rows = []
    for n in SHOW:
        rows.append([f'`{n}`'] + [n3(S[n]['v1'][f'edition={e}']['f1']) + ' → ' + n3(S[n]['v2'][f'edition={e}']['f1']) if 'v1' in S[n] else n3(S[n]['v2a'][f'edition={e}']['f1']) + ' → ' + n3(S[n]['v2'][f'edition={e}']['f1']) + ' (v2a→v2)' for e in EDS])
    md['edition_v1v2'] = table(['système'] + [e.split('_')[0] for e in EDS], rows)
    rows = []
    for n in SHOW:
        rows.append([f'`{n}`'] + [(n3(S[n]['v1'][f'lang={l}']['f1']) + ' → ' if 'v1' in S[n] else '') + n3(S[n]['v2'][f'lang={l}']['f1']) for l in ['de', 'en', 'fr', 'it', 'la']])
    md['lang_v1v2'] = table(['système'] + ['de', 'en', 'fr', 'it', 'la'], rows)

    # --- drop-cap repair, by edition, mean of the 'ours' systems that have v1 predictions ----------------------------------------------
    ours = [n for n in names if family(n) == 'ours' and 'v1' in S[n]]
    off = [n for n in names if family(n) == 'offshelf' and 'v1' in S[n]]
    rows = []
    for e in EDS:
        k = f'edition={e}'
        rows.append([e, n3(sum(S[n]['v1'][k]['f1'] for n in ours) / len(ours)), n3(sum(S[n]['v2a'][k]['f1'] for n in ours) / len(ours)),
                     n3(sum(S[n]['v2a'][k]['f1'] - S[n]['v1'][k]['f1'] for n in ours) / len(ours), 3),
                     n3(sum(S[n]['v1'][k]['f1'] for n in off) / len(off)), n3(sum(S[n]['v2a'][k]['f1'] for n in off) / len(off)),
                     n3(sum(S[n]['v2a'][k]['f1'] - S[n]['v1'][k]['f1'] for n in off) / len(off), 3)])
    k = 'all'
    rows.append(['toutes', n3(sum(S[n]['v1'][k]['f1'] for n in ours) / len(ours)), n3(sum(S[n]['v2a'][k]['f1'] for n in ours) / len(ours)),
                 n3(sum(S[n]['v2a'][k]['f1'] - S[n]['v1'][k]['f1'] for n in ours) / len(ours)), n3(sum(S[n]['v1'][k]['f1'] for n in off) / len(off)),
                 n3(sum(S[n]['v2a'][k]['f1'] for n in off) / len(off)), n3(sum(S[n]['v2a'][k]['f1'] - S[n]['v1'][k]['f1'] for n in off) / len(off))])
    md['repair'] = table(['édition', f'nos modèles v1 (n={len(ours)})', 'v2a', 'écart', f'hors boîte v1 (n={len(off)})', 'v2a', 'écart'], rows)
    md['repair_n'] = (len(ours), len(off))

    # --- pool bias ---------------------------------------------------------------------------------------------------------------------
    fg = sc['family_gain']
    fate = sc['fp_fate']
    def fam_mean(fam, key):
        l = [fate[n][key] for n in names if family(n) == fam]
        return sum(l) / len(l)
    rows = []
    for f_, lab in (('ours', 'nos modèles (19)'), ('offshelf', 'hors boîte (22)')):
        rows.append([lab, n3(fg[f_]['mean_f1_v2a']), n3(fg[f_]['mean_f1_v2']), n3(fg[f_]['mean_gain']), f"[{n3(fg[f_]['ci95'][0])}–{n3(fg[f_]['ci95'][1])}]",
                     n3(fam_mean(f_, 'share_new_tp'), 2), n3(fam_mean(f_, 'share_masked'), 2), n3(fam_mean(f_, 'share_left'), 2)])
    md['family'] = table(['famille', 'F1 moyen v2a', 'F1 moyen v2', 'gain moyen', 'IC 95 %', 'faux positifs devenus vrais', 'faux positifs masqués (ambigus)', 'faux positifs restants'], rows)
    rows = []
    for n in SHOW:
        s = S[n]
        rows.append([f'`{n}`', n3(s['v2']['all']['f1']), n3(s['pool_minus_self']['all']['f1']), n3(s['pool_minus_family']['all']['f1']), n3(s['pool_only_ours']['all']['f1']),
                     n3(s['pool_only_off']['all']['f1'])])
    md['pool'] = table(['système', 'lot complet (v2)', 'sans lui-même', 'sans sa famille', 'seulement lieux trouvés par un des nôtres', 'seulement lieux trouvés par un hors boîte ou le LLM'], rows)
    rows = []
    for n in SHOW:
        s = S[n]
        rows.append([f'`{n}`', n3(s['v2_amb_neg']['all']['f1']), n3(s['v2']['all']['f1']), n3(s['v2_amb_pos']['all']['f1']),
                     n3(s['v2_amb_neg']['no_koenigsfelden']['f1']), n3(s['v2']['no_koenigsfelden']['f1']), n3(s['v2_amb_pos']['no_koenigsfelden']['f1'])])
    md['amb'] = table(['système', 'ambigus = non-lieux', 'v2 (ambigus exclus)', 'ambigus = lieux', 'idem hors Kö.', 'v2 hors Kö.', 'idem hors Kö.'], rows)
    # LLM sample
    rows = []
    for n in ['zs_llm_qwen35_9b', 'r3_mdeberta_HM', 'r3_xlmr_large_H', 'zs_flair_ner_en_large', 'zs_gliner_x_large']:
        o = S[n]['v2']['llm_sample']
        rows.append([f'`{n}`', n3(o['p']), n3(o['r']), cell(o)])
    md['llm'] = table(['système', 'P', 'R', 'F1 v2 sur les passages vus par le LLM [IC]'], rows)
    # PER
    rows = []
    for n in ['r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_xlmr_large_H', 'r3_hmteams_HM', 'zs_flair_ner_en_large']:
        rows.append([f'`{n}`', n3(S[n]['per_v1']['all']['f1']) if 'per_v1' in S[n] else '', n3(S[n]['per_v2']['all']['f1']), f"[{n3(S[n]['per_v2']['all']['f1_ci95'][0])}–{n3(S[n]['per_v2']['all']['f1_ci95'][1])}]"])
    md['per'] = table(['système', 'PER F1 v1 (mêmes passages)', 'PER F1 texte réparé', 'IC 95 %'], rows)
    # top-4 pairwise
    rows = [[k, n3(v['diff']), f"[{n3(v['ci95'][0])}–{n3(v['ci95'][1])}]"] for k, v in sc['paired_top4_v2'].items()]
    md['top4'] = table(['écart F1 v2', 'valeur', 'IC 95 % apparié'], rows)

    # --- counts tables for the datasheet ------------------------------------------------------------------------------------------
    old = json.loads(a.old_stats.read_text(encoding='utf-8'))
    rows = []
    for e in EDS:
        c = cnt['edition'][e]
        rows.append([e, c['passages'], c['editor_spans'], f"+{c['added']}", f"−{c['removed']}", c['ambiguous_candidates'], c['ambiguous_editor_spans'], c['gold_v2_scored'], c['masked_tokens']])
    t = cnt['total']
    rows.append(['**total**', t['passages'], t['editor_spans'], f"+{t['added']}", f"−{t['removed']}", t['ambiguous_candidates'], t['ambiguous_editor_spans'], t['gold_v2_scored'], t['masked_tokens']])
    md['counts_ed'] = table(['édition', 'passages notés', 'empans LOC éditeurs (v1)', 'ajoutés', 'retirés', 'ambigus (candidats)', 'ambigus (empans éditeurs)', 'empans LOC de l’or v2 (notés)', 'tokens masqués'], rows)
    rows = []
    for l in LANGS:
        c = cnt['language'].get(l)
        if not c:
            continue
        rows.append([l if l != 'und' else 'inconnue', c['passages'], c['editor_spans'], f"+{c['added']}", f"−{c['removed']}", c['ambiguous_candidates'], c['ambiguous_editor_spans'], c['gold_v2_scored'], c['masked_tokens']])
    md['counts_lang'] = table(['langue', 'passages notés', 'empans LOC éditeurs (v1)', 'ajoutés', 'retirés', 'ambigus (candidats)', 'ambigus (empans éditeurs)', 'empans LOC de l’or v2 (notés)', 'tokens masqués'], rows)
    rows = []
    for e in ['elan_crenum', 'hainhofer_reisen', 'kaiserin_eleonora_magdalena', 'koenigsfelden', 'reed', 'sozinianische_briefwechsel', 'elan_fnvr']:
        o = old[e]
        rows.append([e, o['passages_v1'], o['passages_changed'], o['detached_v1'], o['detached_v2'], o['flag_ws_repair'], o['flag_dehyph'], o['tokens_v1'], o['tokens_v2']])
    md['repair_counts'] = table(['édition', 'passages', 'passages au texte modifié', 'lettrines détachées v1', 'v2', 'passages réparés (espace dans choice/app)', 'passages réparés (trait d’union de fin de ligne)', 'tokens v1', 'tokens v2'], rows)
    cs = json.loads(a.cand_summary.read_text(encoding='utf-8'))
    tg = sum(cs[e]['gold_touched'] for e in EDS)
    ng = sum(cs[e]['gold_spans'] for e in EDS)
    md['pool_cover'] = (tg, ng)
    (a.out_dir / 'test_v2_tables.json').write_text(json.dumps(md, indent=1, ensure_ascii=False), encoding='utf-8')
    print('ok', tg, ng)


if __name__ == '__main__':
    main()
