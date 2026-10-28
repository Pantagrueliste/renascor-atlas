"""Score teacher predictions on dev3 (never on gold v2): LOC and PER, strict and overlap, per language, plus combined teachers.

Predictions: gz/plain jsonl, one line per passage, either {"id", "LOC": [{"s","e"}], "PER": [{"s","e"}]} (token indices, end exclusive)
or the output of renner_llm_teacher.py ({"id", "spans": [{"start","end","label"}], "error"}).
    python scripts/ner3_teacher_score.py --gold dev3/gold_dev3.jsonl --pred name=pred1.jsonl.gz --pred name2=... [--combine and:a,b]
        [--combine union:a,b] [--combine vote:a,b,c,d@3] [--accept-alt] [--sure-only] [--bootstrap 1000] [--json out.json]

--accept-alt: a gold span with a long form (``alt_start``: title or generic word in front of the core) is matched by the core or by the
              long form (the dev3 rule). Overlap mode then uses the long form.
--sure-only:  gold spans marked ``doubt`` count neither as error nor as success: they are removed, and a prediction matching one
              (strict) or touching one (overlap) is ignored.
--bootstrap N: 95% percentile intervals of P, R and F1 per language, passages resampled with replacement (seed 20261023).
Without --accept-alt and --sure-only the numbers are those of the first version of this script.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict


def opn(p):
    return gzip.open(p, 'rt', encoding='utf-8') if str(p).endswith('.gz') else open(p, encoding='utf-8')


def load(p):
    return [json.loads(l) for l in opn(p) if l.strip()]


def to_pred(d):
    """Normalise one prediction line to {"id", "LOC": [{s,e}], "PER": [{s,e}]}."""
    if 'spans' in d and 'LOC' not in d and 'PER' not in d:
        out = {'id': d['id'], 'LOC': [], 'PER': []}
        for s in d.get('spans') or []:
            if s.get('label') in ('LOC', 'PER'):
                out[s['label']].append({'s': s['start'], 'e': s['end']})
        return out
    return d


def gold_items(r, lab, accept_alt=False):
    """Gold spans of one label: list of (accepted (s, e) forms, overlap range, sure)."""
    out = []
    for s in r['spans']:
        if s['label'] != lab:
            continue
        forms = {(s['start'], s['end'])}
        lo = s['start']
        if accept_alt and s.get('alt_start') is not None:
            forms.add((s['alt_start'], s['end']))
            lo = min(lo, s['alt_start'])
        out.append((forms, (lo, s['end']), s.get('certainty', 'sure') != 'doubt'))
    return out


def gold_spans(r, lab):
    return {(s['start'], s['end']) for s in r['spans'] if s['label'] == lab}


def pred_spans(d, lab):
    return {(x['s'], x['e']) for x in d.get(lab, [])}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _ov(a, b):
    return a[0] < b[1] and b[0] < a[1]


def count_items(g, p, mode, sure_only=False):
    """(tp, fp, fn) for gold items ``g`` (see gold_items) and predicted spans ``p``.

    strict: a gold item is found when one prediction equals one of its accepted forms (each prediction used once).
    overlap: as in the first version (predictions touching a gold span are right; gold spans touched are found)."""
    p = sorted(p)
    if mode == 'strict':
        used = set()
        tp = fn = 0
        ignored = set()
        for forms, _, sure in g:
            hit = next((x for x in p if x in forms and x not in used), None)
            if hit is not None:
                used.add(hit)
            if sure_only and not sure:
                if hit is not None:
                    ignored.add(hit)
                continue
            if hit is not None:
                tp += 1
            else:
                fn += 1
        fp = sum(1 for x in p if x not in used and x not in ignored)
        return tp, fp, fn
    keep = [it for it in g if it[2] or not sure_only]
    doubt = [it for it in g if sure_only and not it[2]]
    tp_p = fp = 0
    for x in p:
        if any(_ov(x, it[1]) for it in keep):
            tp_p += 1
        elif any(_ov(x, it[1]) for it in doubt):
            continue
        else:
            fp += 1
    tp_g = sum(any(_ov(x, it[1]) for x in p) for it in keep)
    return tp_p, fp, len(keep) - tp_g


def count(g, p, mode):
    """First-version count on plain gold span sets (kept for callers of the old function)."""
    return count_items([({x}, x, True) for x in g], p, mode)


def combine(spec, preds):
    kind, rest = spec.split(':')
    names, _, k = rest.partition('@')
    names = names.split(',')
    k = int(k) if k else len(names)
    out = {}
    ids = set().union(*(preds[n].keys() for n in names))
    for i in ids:
        d = {'id': i}
        for lab in ('LOC', 'PER'):
            cnt = defaultdict(int)
            for n in names:
                for sp in pred_spans(preds[n].get(i, {}), lab):
                    cnt[sp] += 1
            need = {'and': len(names), 'union': 1, 'vote': k}[kind]
            d[lab] = [{'s': s, 'e': e} for (s, e), c in cnt.items() if c >= need]
        out[i] = d
    return out


def per_passage_counts(gold, pred, accept_alt=False, sure_only=False):
    """{passage id: {(lab, mode): (tp, fp, fn)}}"""
    out = {}
    for r in gold:
        c = {}
        for lab in ('LOC', 'PER'):
            g = gold_items(r, lab, accept_alt)
            p = pred_spans(pred.get(r['id'], {}), lab)
            for mode in ('strict', 'overlap'):
                c[(lab, mode)] = count_items(g, p, mode, sure_only)
        out[r['id']] = c
    return out


def score(gold, pred, accept_alt=False, sure_only=False, counts=None):
    counts = counts or per_passage_counts(gold, pred, accept_alt, sure_only)
    res = {}
    for lang in sorted({r['lang'] for r in gold}) + ['all']:
        for lab in ('LOC', 'PER'):
            for mode in ('strict', 'overlap'):
                t = [0, 0, 0]
                for r in gold:
                    if lang != 'all' and r['lang'] != lang:
                        continue
                    t = [a + b for a, b in zip(t, counts[r['id']][(lab, mode)])]
                res[(lang, lab, mode)] = prf(*t)
    return res


def bootstrap(gold, counts, n=1000, seed=20261023):
    """95% percentile intervals of (P, R, F1) per (lang, lab, mode); passages resampled within each language."""
    rnd = random.Random(seed)
    by_lang = defaultdict(list)
    for r in gold:
        by_lang[r['lang']].append(r['id'])
    by_lang['all'] = [r['id'] for r in gold]
    out = {}
    for lang, ids in by_lang.items():
        samples = defaultdict(list)
        for _ in range(n):
            pick = [ids[rnd.randrange(len(ids))] for _ in ids]
            for lab in ('LOC', 'PER'):
                for mode in ('strict', 'overlap'):
                    t = [0, 0, 0]
                    for i in pick:
                        t = [a + b for a, b in zip(t, counts[i][(lab, mode)])]
                    samples[(lab, mode)].append(prf(*t))
        for key, vals in samples.items():
            ci = []
            for k in range(3):
                xs = sorted(v[k] for v in vals)
                ci.append((xs[int(0.025 * (n - 1))], xs[int(0.975 * (n - 1))]))
            out[(lang,) + key] = ci
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', action='append', default=[])
    ap.add_argument('--combine', action='append', default=[])
    ap.add_argument('--accept-alt', action='store_true')
    ap.add_argument('--sure-only', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--json')
    a = ap.parse_args()
    assert 'final_gold_v2' not in a.gold, 'gold v2 is the final test set'
    gold = load(a.gold)
    preds = {}
    for s in a.pred:
        n, p = s.split('=', 1)
        preds[n] = {d['id']: to_pred(d) for d in load(p)}
    for s in a.combine:
        preds[s] = combine(s, preds)
    rows = ['| teacher | lang | LOC strict P/R/F1 | LOC overlap F1 | PER strict P/R/F1 | PER overlap F1 |', '|---|---|---|---|---|---|']
    js = {}
    for n, p in preds.items():
        counts = per_passage_counts(gold, p, a.accept_alt, a.sure_only)
        sc = score(gold, p, counts=counts)
        ci = bootstrap(gold, counts, a.bootstrap) if a.bootstrap else {}
        js[n] = {'|'.join(k): {'p': v[0], 'r': v[1], 'f1': v[2], **({'ci': ci[k]} if k in ci else {})} for k, v in sc.items()}
        for lang in sorted({r['lang'] for r in gold}) + ['all']:
            f = lambda lab, m: '/'.join(f'{x:.3f}' for x in sc[(lang, lab, m)])
            rows.append(f'| {n} | {lang} | {f("LOC","strict")} | {sc[(lang,"LOC","overlap")][2]:.3f} | {f("PER","strict")} | {sc[(lang,"PER","overlap")][2]:.3f} |')
    txt = '\n'.join(rows)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')
    if a.json:
        json.dump({'accept_alt': a.accept_alt, 'sure_only': a.sure_only, 'bootstrap': a.bootstrap, 'scores': js}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
