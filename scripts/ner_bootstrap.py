"""Bootstrap confidence intervals (resampling whole documents) for the fine-tuned runs on the held-out test set, and paired differences with a reference run.

   python scripts/ner_bootstrap.py RUNS_DIR TEST_JSONL OUT_JSON REFERENCE_RUN [RUN ...]
Uses runs/<run>/pred_test.jsonl.gz (predicted token spans). Resamples: 2000, seed 1. Reports micro F1 and macro-over-editions F1 with 95% intervals.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas.ner.spans import tags_to_spans

runs_dir, test_path, out_path, ref = sys.argv[1:5]
names = sys.argv[5:] or sorted(p.parent.name for p in Path(runs_dir).glob('*/pred_test.jsonl.gz'))
B = 2000

docs, eds, gold = {}, [], {}
rec_doc = {}
with open(test_path) as f:
    for line in f:
        r = json.loads(line)
        key = (r['edition'], r['doc'])
        if key not in docs:
            docs[key] = len(docs)
            eds.append(r['edition'])
        rec_doc[r['id']] = docs[key]
        for task, tk, vk in (('LOC', 'ner_tags_loc', 'loc_valid'), ('PER', 'ner_tags_per', 'per_valid')):
            if r.get(vk):
                gold[(task, r['id'])] = set(tags_to_spans(r[tk]))
nd = len(docs)
edi = {e: i for i, e in enumerate(sorted(set(eds)))}
ed_of_doc = np.array([edi[e] for e in eds])


def counts(run):
    preds = {}
    with gzip.open(Path(runs_dir) / run / 'pred_test.jsonl.gz', 'rt') as f:
        for l in f:
            d = json.loads(l)
            preds[d['id']] = {t: set(map(tuple, d[t])) for t in ('LOC', 'PER')}
    out = {}
    for task in ('LOC', 'PER'):
        a = np.zeros((nd, 3))
        for (t, rid), g in gold.items():
            if t != task:
                continue
            p = preds.get(rid, {}).get(task, set())
            tp = len(g & p)
            a[rec_doc[rid]] += (tp, len(p) - tp, len(g) - tp)
        out[task] = a
    return out


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


rng = np.random.default_rng(1)
idx = rng.integers(0, nd, size=(B, nd))
res, allc = {}, {}
for run in names:
    allc[run] = counts(run)


def stats(a):
    """micro and macro-edition F1 for each bootstrap sample (and for the full data)."""
    full = a.sum(0)
    micro0 = float(f1(*full))
    per_ed0 = np.array([f1(*a[ed_of_doc == e].sum(0)) for e in range(len(edi)) if a[ed_of_doc == e][:, 0].sum() + a[ed_of_doc == e][:, 2].sum() > 0])
    micro = np.empty(B)
    macro = np.empty(B)
    for b in range(B):
        s = a[idx[b]]
        micro[b] = f1(*s.sum(0))
        e_tot = np.zeros((len(edi), 3))
        np.add.at(e_tot, ed_of_doc[idx[b]], s)
        ok = (e_tot[:, 0] + e_tot[:, 2]) > 0
        macro[b] = f1(e_tot[ok, 0], e_tot[ok, 1], e_tot[ok, 2]).mean()
    return micro0, float(per_ed0.mean()), micro, macro


out = {}
for task in ('LOC', 'PER'):
    ref_micro0, ref_macro0, ref_micro, ref_macro = stats(allc[ref][task])
    for run in names:
        m0, M0, micro, macro = stats(allc[run][task])
        lo, hi = np.percentile(micro, [2.5, 97.5])
        mlo, mhi = np.percentile(macro, [2.5, 97.5])
        d = micro - ref_micro
        dm = macro - ref_macro
        out.setdefault(run, {})[task] = {'micro_f1': round(m0, 4), 'micro_ci95': [round(lo, 4), round(hi, 4)], 'macro_edition_f1': round(M0, 4),
                                         'macro_ci95': [round(mlo, 4), round(mhi, 4)], 'vs_ref': ref,
                                         'diff_micro': round(m0 - ref_micro0, 4), 'diff_micro_ci95': [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                         'diff_macro': round(M0 - ref_macro0, 4), 'diff_macro_ci95': [round(float(np.percentile(dm, 2.5)), 4), round(float(np.percentile(dm, 97.5)), 4)]}
Path(out_path).write_text(json.dumps({'reference': ref, 'resamples': B, 'unit': 'document', 'n_documents': nd, 'runs': out}, indent=1))
for run in names:
    print(run, {t: (out[run][t]['micro_f1'], out[run][t]['micro_ci95'], out[run][t]['macro_edition_f1']) for t in ('LOC', 'PER')})
