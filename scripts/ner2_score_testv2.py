"""Rescore every system of the NER pool on test_v2 (LOC, strict spans), with document-level bootstrap intervals.

    python scripts/ner2_score_testv2.py --gold test_v2.jsonl --pool pool.jsonl.gz --pred-dir pred --systems systems.tsv \
        --v1-test v1_test.jsonl --v1-runs v1runs --v1-audit v1audit --out scores_test_v2.json [--boot 2000]

Scoring versions (LOC, token spans, exact boundaries, one label)
  v1            the v1 test: original text, editor gold, v1 predictions (only for the systems whose v1 predictions exist)
  v2a           repaired text, editor gold: isolates the effect of the drop-capital / line-break repair
  v2            repaired text, editor gold + 'add' - 'remove', 'ambiguous' tokens masked in gold AND predictions (the reference version)
  v2_amb_neg    like v2, but every ambiguous item is NOT a place (editor spans judged ambiguous leave the gold, candidates count as false positives)
  v2_amb_pos    like v2, but every ambiguous item IS a place (added to the gold)
  pool_*        v2 gold rebuilt from the clusters that a restricted set of systems found (bias check of the candidate pool):
                  minus_self    the pool without the scored system itself
                  minus_family  without the scored system's family (ours / off-the-shelf+LLM)
                  only_ours / only_off   clusters found by at least one 'ours' / by at least one other system
Subsets: all, no_koenigsfelden, language (de en fr it la und), edition, de_no_koenigsfelden, llm_sample (passages that the LLM saw).
Bootstrap: documents resampled with replacement (seed 1, 2000 draws, the same draws for every system and subset, as scripts/ner_bootstrap.py);
micro P / R / F1 and macro-over-editions F1.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas.ner.ner2_cand import family, read_preds                        # noqa: E402
from ratlas.ner.ner2_testgold import apply_changes                         # noqa: E402
from ratlas.ner.spans import tags_to_spans                                 # noqa: E402

NO_LOC = 'elan_fnvr'
KOE = 'koenigsfelden'


def fam2(name: str) -> str:
    return 'ours' if family(name) == 'ours' else 'off'


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def prf(c):
    tp, fp, fn = (float(x) for x in c)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def counts_of(gold: set, pred: set):
    tp = len(gold & pred)
    return tp, len(pred) - tp, len(gold) - tp


def drop(spans, mask: set):
    if not mask:
        return set(spans)
    return {sp for sp in spans if not any(t in mask for t in range(sp[0], sp[1]))}


def load_v1_preds(path: Path) -> dict:
    out = {}
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            out[d['id']] = {t: {(x['s'], x['e']) if isinstance(x, dict) else (x[0], x[1]) for x in d.get(t, ())} for t in ('LOC', 'PER')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--systems', type=Path, required=True)
    ap.add_argument('--v1-test', type=Path, required=True)
    ap.add_argument('--v1-runs', type=Path, required=True)
    ap.add_argument('--v1-audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()

    # ---- gold passages ------------------------------------------------------------------------------------------------
    recs = []
    with open(a.gold, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            if not r['loc_valid'] or r['edition'] == NO_LOC:
                continue
            recs.append({'id': r['id'], 'v1_id': r['v1_id'], 'edition': r['edition'], 'doc': r['doc'], 'lang': r.get('lang') or 'und',
                         'n': len(r['tokens']), 'editor': set(map(tuple, tags_to_spans(r['ner_tags_loc_editor']))),
                         'gold': set(map(tuple, r['loc_gold_v2'])), 'mask': set(r['loc_mask']), 'amb': r['loc_ambiguous'],
                         'per': set(tags_to_spans(r['ner_tags_per'])) if r.get('per_valid') else None})
    N = len(recs)
    ids = [r['id'] for r in recs]
    pos = {i: k for k, i in enumerate(ids)}
    docs = {}
    for r in recs:
        docs.setdefault((r['edition'], r['doc']), len(docs))
    doc_idx = np.array([docs[(r['edition'], r['doc'])] for r in recs])
    nd = len(docs)
    ed_of_doc = np.zeros(nd, dtype=int)
    eds = sorted({r['edition'] for r in recs})
    for (e, _), k in docs.items():
        ed_of_doc[k] = eds.index(e)
    edition = np.array([r['edition'] for r in recs])
    lang = np.array([r['lang'] for r in recs])
    langs = sorted(set(lang))
    pool = {}
    with gzip.open(a.pool, 'rt', encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            pool[d['id']] = {'add': d['add'], 'mask': d['mask'], 'remove': d['remove']}

    # ---- systems ------------------------------------------------------------------------------------------------------
    sysinfo = {}
    with open(a.systems, encoding='utf-8') as fh:
        hdr = fh.readline().rstrip('\n').split('\t')
        for line in fh:
            row = dict(zip(hdr, line.rstrip('\n').split('\t')))
            sysinfo[row['system']] = row
    names = sorted(p.name[:-len('.jsonl.gz')] for p in Path(a.pred_dir).glob('*.jsonl.gz'))
    print(len(names), 'systems', N, 'passages', nd, 'documents', file=sys.stderr)

    # ---- v1 gold ------------------------------------------------------------------------------------------------------
    v1 = {}
    with open(a.v1_test, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            if r['loc_valid'] and r['edition'] != NO_LOC:
                v1[r['id']] = {'gold': set(map(tuple, tags_to_spans(r['ner_tags_loc']))), 'per': set(tags_to_spans(r['ner_tags_per'])) if r.get('per_valid') else None}
    v1_pos = {r['v1_id']: pos[r['id']] for r in recs if r['v1_id'] in v1}

    # ---- restricted gold sets for the pool bias check -----------------------------------------------------------------------------
    def restricted(k: int, allowed) -> tuple[set, set]:
        """gold, mask for passage k when only the clusters found by a system with allowed(name) count."""
        ch = pool.get(ids[k])
        r = recs[k]
        if ch is None:
            return r['gold'], r['mask']
        c2 = {'add': [x for x in ch['add'] if any(allowed(s) for s in x['systems'])],
              'mask': [x for x in ch['mask'] if x['what'] == 'editor_span' or any(allowed(s) for s in x['systems'])],
              'remove': ch['remove']}
        res = apply_changes(sorted(r['editor']), c2, r['n'])
        return set(map(tuple, res['gold'])), set(res['mask'])

    bad = sum(1 for k in touched_ if restricted(k, lambda s: True) != (recs[k]['gold'], recs[k]['mask'])) if (touched_ := [k for k, i in enumerate(ids) if i in pool]) else 0
    assert bad == 0, f'restricted gold with all systems differs from v2 on {bad} passages'
    VARS = ['v2a', 'v2', 'v2_amb_neg', 'v2_amb_pos', 'pool_minus_self', 'pool_minus_family', 'pool_only_ours', 'pool_only_off']
    cnt: dict = {}                    # cnt[system][variant] -> (N, 3) int array
    cnt_per: dict = {}
    have: dict = {}
    edamb = [{(x['s'], x['e']) for x in r['amb'] if x['what'] == 'editor_span'} for r in recs]
    candamb = [{tuple(x['span']) for x in r['amb'] if x['what'] == 'candidate'} for r in recs]
    touched = [k for k, i in enumerate(ids) if i in pool]
    for name in names:
        preds = read_preds(a.pred_dir / f'{name}.jsonl.gz')
        hv = np.array([i in preds for i in ids])
        have[name] = hv
        c = {v: np.zeros((N, 3), dtype=np.int32) for v in VARS}
        cp = np.zeros((N, 3), dtype=np.int32)
        fam = fam2(name)
        for k, i in enumerate(ids):
            p = preds.get(i)
            if p is None:
                continue
            r = recs[k]
            P = {(x['s'], x['e']) for x in p['LOC']}
            c['v2a'][k] = counts_of(r['editor'], P)
            m = r['mask']
            c['v2'][k] = counts_of(drop(r['gold'], m), drop(P, m))
            c['v2_amb_neg'][k] = counts_of(r['gold'] - edamb[k], P)
            c['v2_amb_pos'][k] = counts_of(r['gold'] | edamb[k] | candamb[k], P)
            if r['per'] is not None:
                cp[k] = counts_of(r['per'], {(x['s'], x['e']) for x in p['PER']})
        # restricted pools: only passages with pool entries differ from v2
        for v, allowed in (('pool_minus_self', lambda s, name=name: s != name), ('pool_minus_family', lambda s, fam=fam: fam2(s) != fam),
                           ('pool_only_ours', lambda s: fam2(s) == 'ours'), ('pool_only_off', lambda s: fam2(s) != 'ours')):
            c[v][:] = c['v2']
            for k in touched:
                p = preds.get(ids[k])
                if p is None:
                    continue
                g, m = restricted(k, allowed)
                P = {(x['s'], x['e']) for x in p['LOC']}
                c[v][k] = counts_of(drop(g, m), drop(P, m))
        cnt[name] = c
        cnt_per[name] = cp
    # v1 (original text, original gold) for the systems with v1 predictions
    v1cnt, v1per = {}, {}
    for name in names:
        f = a.v1_runs / name / 'pred_test.jsonl.gz'
        if not f.exists():
            f = a.v1_audit / f'{name}__test.jsonl.gz'
        if not f.exists():
            continue
        preds = load_v1_preds(f)
        c = np.zeros((N, 3), dtype=np.int32)
        cp = np.zeros((N, 3), dtype=np.int32)
        for vid, k in v1_pos.items():
            p = preds.get(vid)
            if p is None:
                continue
            c[k] = counts_of(v1[vid]['gold'], p['LOC'])
            if v1[vid]['per'] is not None:
                cp[k] = counts_of(v1[vid]['per'], p['PER'])
        v1cnt[name] = c
        v1per[name] = cp
    print('v1 predictions for', len(v1cnt), 'systems', file=sys.stderr)

    # ---- subsets ------------------------------------------------------------------------------------------------------
    masks = {'all': np.ones(N, dtype=bool), 'no_koenigsfelden': edition != KOE}
    for l in langs:
        masks[f'lang={l}'] = lang == l
    masks['lang=de_no_koenigsfelden'] = (lang == 'de') & (edition != KOE)
    for e in eds:
        masks[f'edition={e}'] = edition == e
    llm = 'zs_llm_qwen35_9b'
    masks['llm_sample'] = have[llm] if llm in have else np.zeros(N, bool)
    masks['llm_sample_no_koenigsfelden'] = masks['llm_sample'] & (edition != KOE)

    rng = np.random.default_rng(1)
    B = a.boot
    idx = rng.integers(0, nd, size=(B, nd))
    W = np.zeros((B, nd))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=nd)
    cols_ed = [np.where(ed_of_doc == e)[0] for e in range(len(eds))]

    def doc_counts(c, m):
        out = np.zeros((nd, 3))
        for j in range(3):
            out[:, j] = np.bincount(doc_idx[m], weights=c[m, j], minlength=nd)
        return out

    def summarize(c, subset, with_macro=False):
        m = masks[subset]
        dc = doc_counts(c, m)
        tot = dc.sum(0)
        p, r, f = prf(tot)
        T = W @ dc
        fb = f1(T[:, 0], T[:, 1], T[:, 2])
        pb = T[:, 0] / np.maximum(T[:, 0] + T[:, 1], 1)
        rb = T[:, 0] / np.maximum(T[:, 0] + T[:, 2], 1)
        o = {'tp': int(tot[0]), 'fp': int(tot[1]), 'fn': int(tot[2]), 'p': round(p, 4), 'r': round(r, 4), 'f1': round(f, 4),
             'f1_ci95': [round(float(x), 4) for x in np.percentile(fb, [2.5, 97.5])],
             'p_ci95': [round(float(x), 4) for x in np.percentile(pb, [2.5, 97.5])], 'r_ci95': [round(float(x), 4) for x in np.percentile(rb, [2.5, 97.5])]}
        draws = {'f1': fb}
        if with_macro:
            ok_eds = [e for e in range(len(eds)) if dc[cols_ed[e]][:, 0].sum() + dc[cols_ed[e]][:, 2].sum() > 0]
            per_ed = [f1(*dc[cols_ed[e]].sum(0)) for e in ok_eds]
            mb = np.zeros(B)
            for e in ok_eds:
                Te = W[:, cols_ed[e]] @ dc[cols_ed[e]]
                mb += f1(Te[:, 0], Te[:, 1], Te[:, 2])
            mb /= max(len(ok_eds), 1)
            o['macro_edition_f1'] = round(float(np.mean(per_ed)), 4)
            o['macro_ci95'] = [round(float(x), 4) for x in np.percentile(mb, [2.5, 97.5])]
            draws['macro'] = mb
        return o, draws

    out: dict = {'meta': {'n_passages_scored': N, 'n_documents': nd, 'editions': eds, 'languages': langs, 'bootstrap': {'draws': B, 'seed': 1, 'unit': 'document'},
                          'systems': {n: {'family': family(n), 'passages_covered': int(have[n].sum()), 'has_v1_preds': n in v1cnt,
                                          'v1_table': {k: sysinfo.get(n, {}).get(k) for k in ('v1_test_LOC_P', 'v1_test_LOC_R', 'v1_test_LOC_F1')}} for n in names},
                          'passages_per_subset': {s: int(m.sum()) for s, m in masks.items()}},
                 'scores': {}}
    gain_draws = {}
    main_subsets = list(masks)
    for name in names:
        sc: dict = {}
        for v in ['v1', 'v2a', 'v2'] + VARS[2:]:
            if v == 'v1':
                if name not in v1cnt:
                    continue
                c = v1cnt[name]
            else:
                c = cnt[name][v]
            sc[v] = {}
            subs = main_subsets if v in ('v1', 'v2a', 'v2') else ['all', 'no_koenigsfelden']
            for s in subs:
                o, d = summarize(c, s, with_macro=s in ('all', 'no_koenigsfelden'))
                sc[v][s] = o
                if s == 'all' and v in ('v2a', 'v2', 'v1'):
                    gain_draws[(name, v)] = d['f1']
        # PER (secondary): same passages, repaired text
        pm = np.array([r['per'] is not None for r in recs])
        masks['_per'] = pm
        o, _ = summarize(cnt_per[name], '_per')
        sc['per_v2'] = {'all': o}
        if name in v1per:
            o, _ = summarize(v1per[name], '_per')
            sc['per_v1'] = {'all': o}
        out['scores'][name] = sc
    # ---- paired differences --------------------------------------------------------------------------------------------
    def ci(x):
        return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]

    paired = {}
    for name in names:
        if (name, 'v2a') not in gain_draws:
            continue
        g = gain_draws[(name, 'v2')] - gain_draws[(name, 'v2a')]
        e = {'gain_v2a_to_v2': round(out['scores'][name]['v2']['all']['f1'] - out['scores'][name]['v2a']['all']['f1'], 4), 'gain_ci95': ci(g)}
        if (name, 'v1') in gain_draws:
            e['gain_v1_to_v2a'] = round(out['scores'][name]['v2a']['all']['f1'] - out['scores'][name]['v1']['all']['f1'], 4)
            e['gain_v1_to_v2'] = round(out['scores'][name]['v2']['all']['f1'] - out['scores'][name]['v1']['all']['f1'], 4)
            e['gain_v1_to_v2_ci95'] = ci(gain_draws[(name, 'v2')] - gain_draws[(name, 'v1')])
        paired[name] = e
    out['paired_gain'] = paired
    fams = {'ours': [n for n in names if family(n) == 'ours'], 'offshelf': [n for n in names if family(n) == 'offshelf']}
    famrep = {}
    for f_, lst in fams.items():
        gs = np.mean([gain_draws[(n, 'v2')] - gain_draws[(n, 'v2a')] for n in lst], axis=0)
        famrep[f_] = {'n_systems': len(lst), 'mean_gain': round(float(np.mean([paired[n]['gain_v2a_to_v2'] for n in lst])), 4), 'ci95': ci(gs),
                      'mean_f1_v2a': round(float(np.mean([out['scores'][n]['v2a']['all']['f1'] for n in lst])), 4),
                      'mean_f1_v2': round(float(np.mean([out['scores'][n]['v2']['all']['f1'] for n in lst])), 4)}
    diff = np.mean([gain_draws[(n, 'v2')] - gain_draws[(n, 'v2a')] for n in fams['ours']], axis=0) - np.mean(
        [gain_draws[(n, 'v2')] - gain_draws[(n, 'v2a')] for n in fams['offshelf']], axis=0)
    famrep['ours_minus_offshelf_gain'] = {'diff': round(famrep['ours']['mean_gain'] - famrep['offshelf']['mean_gain'], 4), 'ci95': ci(diff)}
    out['family_gain'] = famrep
    # paired difference between the top systems on v2
    top = ['r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_xlmr_large_H', 'r3_hmteams_HM']
    pd = {}
    for x in top:
        for y in top:
            if x < y:
                d = gain_draws[(x, 'v2')] - gain_draws[(y, 'v2')]
                pd[f'{x} - {y}'] = {'diff': round(out['scores'][x]['v2']['all']['f1'] - out['scores'][y]['v2']['all']['f1'], 4), 'ci95': ci(d)}
    out['paired_top4_v2'] = pd
    vs = {}
    for x, y in (('r3_mdeberta_HM', 'zs_gliner_x_large'), ('r3_mdeberta_HM', 'zs_flair_ner_en_large'), ('r3_mdeberta_HM', 'zs_flair_ner_de_large'),
                 ('r3_xlmr_large_H', 'zs_gliner_x_large'), ('r3_xlmr_large_H', 'zs_flair_ner_en_large'), ('ft_xlmr_large_ours', 'zs_gliner_x_large')):
        e = {}
        for v in ('v1', 'v2a', 'v2'):
            if (x, v) in gain_draws and (y, v) in gain_draws:
                d = gain_draws[(x, v)] - gain_draws[(y, v)]
                e[v] = {'diff': round(out['scores'][x][v]['all']['f1'] - out['scores'][y][v]['all']['f1'], 4), 'ci95': ci(d)}
        vs[f'{x} - {y}'] = e
    out['paired_ours_vs_offshelf'] = vs
    # ---- where do the false positives go? (v2a -> v2) ---------------------------------------------------------------------
    fate = {}
    for name in names:
        a_ = out['scores'][name]['v2a']['all']
        b_ = out['scores'][name]['v2']['all']
        new_tp = b_['tp'] - a_['tp']
        masked = a_['fp'] - b_['fp'] - new_tp
        fate[name] = {'fp_v2a': a_['fp'], 'new_tp': new_tp, 'fp_masked_ambiguous': masked, 'fp_left': b_['fp'],
                      'share_new_tp': round(new_tp / max(a_['fp'], 1), 4), 'share_masked': round(masked / max(a_['fp'], 1), 4), 'share_left': round(b_['fp'] / max(a_['fp'], 1), 4)}
    out['fp_fate'] = fate
    # ---- who found the added spans -----------------------------------------------------------------------------------------
    found = Counter()
    sizes = Counter()
    for k in touched:
        for x in pool[ids[k]]['add']:
            fs = {fam2(s) for s in x['systems']}
            found['ours' if fs == {'ours'} else 'off' if fs == {'off'} else 'both'] += 1
            sizes[min(len(x['systems']), 10)] += 1
    out['added_spans_found_by'] = {'only_ours': found['ours'], 'only_offshelf_or_llm': found['off'], 'both': found['both'], 'n_systems_hist(10=10+)': dict(sorted(sizes.items()))}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', a.out)


if __name__ == '__main__':
    main()
