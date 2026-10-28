"""Independent scoring of the audit predictions (CPU job, slurm/ner_audit_cpu.sbatch).

Reads the gold of data_v1/test.jsonl, the predictions written by ner_audit_predict.py and the leak tables of ner_audit_leak.py; writes one JSON with
  - strict token-span P/R/F1 (own implementation), overall / per edition / per language, LOC and PER
  - a character-level cross-check (gold from the TEI character spans, predictions from character offsets)
  - relaxed (any shared word) P/R, boundary errors
  - recall on gold place names seen / unseen as LOC in the training data (gazetteer overlap)
  - false positives that are strings tagged LOC elsewhere in the same edition (label noise), precision bounds
  - bootstrap by document (95% interval), paired differences between models, leave-one-edition-out
  - precision / recall as a function of the span confidence
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def spans_of(tags):
    out, start = [], None
    for i, t in enumerate(list(tags) + [0]):
        if t == 1 or t == 0:
            if start is not None:
                out.append((start, i))
                start = None
            if t == 1:
                start = i
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'p': round(p, 4), 'r': round(r, 4), 'f1': round(f, 4)}


def norm(s):
    return s.lower().replace('ſ', 's')


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            yield dict(zip(head, line.rstrip('\n').split('\t')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--models', required=True, help='comma list of model names (files <name>__test.jsonl.gz)')
    ap.add_argument('--leak', required=True)
    ap.add_argument('--stored-dir', help='runs/<model>/pred_test.jsonl.gz of the original pipeline, to compare')
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--split', default='test')
    a = ap.parse_args()

    recs = {}
    with open(Path(a.data) / f'{a.split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            recs[d['id']] = d
    ids = list(recs)
    log(len(ids), 'test passages')
    gold = {}
    for i, r in recs.items():
        gold[i] = {'LOC': set(spans_of(r['ner_tags_loc'])), 'PER': set(spans_of(r['ner_tags_per']))}
    docs = defaultdict(list)
    for i, r in recs.items():
        docs[(r['edition'], r['doc'])].append(i)
    dockeys = list(docs)
    # gazetteer / seen
    seen = {}      # (id, s, e) -> train_ours count
    for row in read_tsv(Path(a.leak) / f'{a.split}_loc_spans_seen.tsv'):
        seen[(row['id'], int(row['s']), int(row['e']))] = (int(row['train_ours']), int(row['train_hist_gold']), int(row['train_modern']))
    ed_gold_surf = defaultdict(set)         # edition -> surfaces tagged LOC in that edition
    for i, r in recs.items():
        if r.get('loc_valid', True):
            for s, e in gold[i]['LOC']:
                ed_gold_surf[r['edition']].add(norm(' '.join(r['tokens'][s:e])))
    train_loc = {}                           # surface -> (ours) from the leak table
    for (i, s, e), v in seen.items():
        pass

    result = {'n_passages': len(ids), 'models': {}}
    per_model_counts = {}
    for m in a.models.split(','):
        t0 = time.time()
        pred = {}
        with gzip.open(Path(a.pred_dir) / f'{m}__{a.split}.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                pred[d['id']] = d
        R = {}
        for task in ('LOC', 'PER'):
            valid_key = 'loc_valid' if task == 'LOC' else 'per_valid'
            per_rec = {}      # id -> (tp, fp, fn)
            groups = defaultdict(lambda: np.zeros(3, dtype=np.int64))
            rel = defaultdict(lambda: np.zeros(4, dtype=np.int64))      # pred_total, pred_overlap, gold_total, gold_overlap
            seen_cnt = defaultdict(lambda: np.zeros(2, dtype=np.int64))  # recall by seen bucket: [hit, total]
            fp_kinds = Counter()
            fn_kinds = Counter()
            fp_by_ed_kind = defaultdict(Counter)
            conf_rows = []
            char_tp = char_fp = char_fn = 0
            for i in ids:
                r = recs[i]
                if not r.get(valid_key, True):
                    continue
                g = gold[i][task]
                p = {(x['s'], x['e']): x for x in pred[i][task]}
                ps = set(p)
                tp = len(g & ps)
                fp = len(ps) - tp
                fn = len(g) - tp
                per_rec[i] = (tp, fp, fn)
                for grp in ('all', 'lang=' + (r.get('lang') or 'unk'), 'edition=' + r['edition']):
                    groups[grp] += (tp, fp, fn)
                    pr_ov = sum(1 for x in ps if any(overlaps(x, y) for y in g))
                    gl_ov = sum(1 for y in g if any(overlaps(x, y) for x in ps))
                    rel[grp] += (len(ps), pr_ov, len(g), gl_ov)
                toks = r['tokens']
                if task == 'LOC':
                    for (s, e) in g:
                        v = seen.get((i, s, e), (0, 0, 0))
                        b = 'seen_ours' if v[0] > 0 else ('seen_external_only' if (v[1] + v[2]) > 0 else 'unseen')
                        for grp in ('all', 'edition=' + r['edition']):
                            seen_cnt[(grp, b)] += (1 if (s, e) in ps else 0, 1)
                    for (s, e), x in p.items():
                        ok = (s, e) in g
                        conf_rows.append((x['conf'], ok, r['edition'], r.get('lang') or 'unk'))
                        if not ok:
                            surf = norm(' '.join(toks[s:e]))
                            ov = [y for y in g if overlaps((s, e), y)]
                            if ov:
                                kind = 'boundary'
                            elif surf in ed_gold_surf[r['edition']]:
                                kind = 'tagged_elsewhere_in_edition'
                            else:
                                kind = 'other'
                            fp_kinds[kind] += 1
                            fp_by_ed_kind[r['edition']][kind] += 1
                    for (s, e) in g:
                        if (s, e) not in ps:
                            ov = [y for y in ps if overlaps((s, e), y)]
                            fn_kinds['boundary' if ov else 'missed'] += 1
                    # character-level cross-check: gold from the TEI character spans (outermost kept), prediction from character offsets
                    gc = [(sp['start'], sp['end']) for sp in r['spans'] if sp['label'] == 'LOC']
                    gc = [x for x in gc if not any(y != x and y[0] <= x[0] and x[1] <= y[1] for y in gc)]
                    gcs = set(gc)
                    pcs = set((x['cs'], x['ce']) for x in pred[i][task] if 'cs' in x)
                    # a TEI span can start or end in the middle of a token: compare after snapping both to token edges
                    starts = r['token_starts']
                    lens = [len(w) for w in toks]
                    snapped = set()
                    for (cs, ce) in gcs:
                        idx = [k for k in range(len(starts)) if starts[k] < ce and starts[k] + lens[k] > cs]
                        if idx:
                            snapped.add((starts[idx[0]], starts[idx[-1]] + lens[idx[-1]]))
                    c_tp = len(snapped & pcs)
                    char_tp += c_tp
                    char_fp += len(pcs) - c_tp
                    char_fn += len(snapped) - c_tp
            res = {'overall': prf(*groups['all']), 'by_edition': {}, 'by_lang': {}}
            for grp, v in sorted(groups.items()):
                if grp.startswith('edition='):
                    res['by_edition'][grp[8:]] = prf(*v)
                elif grp.startswith('lang='):
                    res['by_lang'][grp[5:]] = prf(*v)
            fs = [v['f1'] for v in res['by_edition'].values() if v['tp'] + v['fn'] > 0]
            res['macro_edition_f1'] = round(sum(fs) / len(fs), 4) if fs else None
            rr = {}
            for grp, (pt, po, gt, go) in rel.items():
                if grp == 'all' or grp.startswith('edition=') or grp.startswith('lang='):
                    p_ = po / pt if pt else 0
                    r_ = go / gt if gt else 0
                    rr[grp] = {'relaxed_p': round(p_, 4), 'relaxed_r': round(r_, 4), 'relaxed_f1': round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ else 0}
            res['relaxed'] = rr
            if task == 'LOC':
                res['recall_by_seen'] = {f'{g}|{b}': {'hit': int(v[0]), 'n': int(v[1]), 'recall': round(v[0] / v[1], 4) if v[1] else None} for (g, b), v in sorted(seen_cnt.items())}
                res['fp_kinds'] = dict(fp_kinds)
                res['fn_kinds'] = dict(fn_kinds)
                res['fp_kinds_by_edition'] = {k: dict(v) for k, v in fp_by_ed_kind.items()}
                tp, fp, fn = groups['all']
                adj_fp = fp - fp_kinds['tagged_elsewhere_in_edition']
                res['precision_bounds'] = {'strict': round(tp / (tp + fp), 4), 'upper_if_tagged_elsewhere_are_correct': round(tp / (tp + adj_fp), 4)}
                f_up = lambda p_, r_: 2 * p_ * r_ / (p_ + r_)
                res['f1_upper_bound_label_noise'] = round(f_up(tp / (tp + adj_fp), tp / (tp + fn)), 4)
                # confidence curve
                curve = []
                total_gold = tp + fn
                for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995):
                    sel = [x for x in conf_rows if x[0] >= th]
                    t_ = sum(1 for x in sel if x[1])
                    p_ = t_ / len(sel) if sel else 0
                    r_ = t_ / total_gold if total_gold else 0
                    curve.append({'threshold': th, 'n_pred': len(sel), 'precision': round(p_, 4), 'recall': round(r_, 4), 'f1': round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ else 0})
                res['confidence_curve'] = curve
                res['confidence_curve_by_lang'] = {}
                for lg in sorted({x[3] for x in conf_rows}):
                    rows = [x for x in conf_rows if x[3] == lg]
                    res['confidence_curve_by_lang'][lg] = [{'threshold': th, 'n_pred': sum(1 for x in rows if x[0] >= th),
                                                            'precision': round(sum(1 for x in rows if x[0] >= th and x[1]) / max(1, sum(1 for x in rows if x[0] >= th)), 4)}
                                                           for th in (0.5, 0.9, 0.98)]
                res['char_level_check'] = prf(char_tp, char_fp, char_fn)
            R[task] = res
            per_model_counts[(m, task)] = per_rec
        # agreement with the stored predictions of the original pipeline
        if a.stored_dir and (Path(a.stored_dir) / m / 'pred_test.jsonl.gz').exists():
            same = tot = 0
            with gzip.open(Path(a.stored_dir) / m / 'pred_test.jsonl.gz', 'rt', encoding='utf-8') as f:
                for line in f:
                    d = json.loads(line)
                    if d['id'] in pred and recs[d['id']].get('loc_valid', True):
                        a_ = set(tuple(x) for x in d['LOC'])
                        b_ = set((x['s'], x['e']) for x in pred[d['id']]['LOC'])
                        tot += 1
                        same += a_ == b_
            R['agreement_with_original_pipeline_LOC_records_identical'] = {'identical': same, 'records': tot}
        result['models'][m] = R
        log(m, 'LOC', R['LOC']['overall'], f'{time.time() - t0:.0f}s')

    # --------------------------------------------------------------------------------- bootstrap by document, paired, leave-one-edition-out
    rng = random.Random(7)
    names = a.models.split(',')
    for task in ('LOC',):
        mats = {}
        for m in names:
            pr = per_model_counts[(m, task)]
            mats[m] = np.array([[sum(pr[i][k] for i in docs[d] if i in pr) for k in range(3)] for d in dockeys], dtype=np.int64)
        D = len(dockeys)
        boots = {m: [] for m in names}
        for b in range(a.boot):
            idx = np.array([rng.randrange(D) for _ in range(D)])
            for m in names:
                tp, fp, fn = mats[m][idx].sum(0)
                boots[m].append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        result['bootstrap_docs'] = {m: {'f1': round(float(np.mean(boots[m])), 4), 'ci95': [round(float(np.percentile(boots[m], 2.5)), 4), round(float(np.percentile(boots[m], 97.5)), 4)]} for m in names}
        diffs = {}
        for i1 in range(len(names)):
            for i2 in range(i1 + 1, len(names)):
                d_ = np.array(boots[names[i1]]) - np.array(boots[names[i2]])
                diffs[f'{names[i1]} - {names[i2]}'] = {'mean': round(float(d_.mean()), 4), 'ci95': [round(float(np.percentile(d_, 2.5)), 4), round(float(np.percentile(d_, 97.5)), 4)],
                                                      'p_diff_le_0': round(float((d_ <= 0).mean()), 4)}
        result['paired_bootstrap'] = diffs
        # leave-one-edition-out
        eds = sorted({recs[i]['edition'] for i in ids if recs[i].get('loc_valid', True) and gold[i]['LOC']})
        loeo = {}
        for drop in eds:
            row = {}
            for m in names:
                pr = per_model_counts[(m, task)]
                tp = fp = fn = 0
                for i, (a1, b1, c1) in pr.items():
                    if recs[i]['edition'] != drop:
                        tp += a1
                        fp += b1
                        fn += c1
                row[m] = round(2 * tp / (2 * tp + fp + fn), 4)
            loeo[drop] = row
        result['leave_one_edition_out_f1'] = loeo
        # rank stability
        rank_first = Counter(max(r.items(), key=lambda kv: kv[1])[0] for r in loeo.values())
        result['loeo_best_model_counts'] = dict(rank_first)
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    log('written', a.out)


if __name__ == '__main__':
    main()
