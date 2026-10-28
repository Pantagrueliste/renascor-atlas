"""Independent leakage audit of the NER data (run on Hypatia, CPU job: slurm/ner_audit_cpu.sbatch).

Questions answered (all on the data as the training code reads it):
  1. exact duplicates: normalised passage text of test (and dev) found in the training data (ours / external hist_gold / external modern_silver)
  2. near duplicates: share of the 8-word shingles of every test passage that occur in each training group; the training edition that supplies them
  3. same documents: dev vs train (documents, neighbouring passages), test vs train (document paths)
  4. gazetteer overlap: share of the test LOC gold spans whose surface string is tagged LOC in training; per edition; writes a per-span table
  5. duplicates inside the training data across editions, and inside test.

Writes JSON + TSV in --out. Nothing here changes the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.spans import tags_to_spans  # noqa: E402

N = 8
WORD = re.compile(r'\w+', re.U)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def norm_tokens(tokens):
    """lower-case, long s folded, punctuation dropped"""
    out = []
    for t in tokens:
        t = t.lower().replace('ſ', 's')
        if WORD.search(t):
            out.append(t)
    return out


def shingles(nt, n=N):
    if len(nt) < n:
        return []
    return [hash(tuple(nt[i:i + n])) for i in range(len(nt) - n + 1)]


def text_key(nt):
    return hashlib.md5(' '.join(nt).encode()).hexdigest()


def read(path, fields):
    with open(path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            yield {k: d.get(k) for k in fields}


def edition_of(r):
    return r.get('edition') or f"{r.get('source')}/{r.get('subset')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--agg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-modern', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    F = ['id', 'edition', 'source', 'subset', 'doc', 'passage', 'tokens', 'ner_tags_loc', 'ner_tags_per', 'loc_valid', 'per_valid', 'lang', 'u_loc']

    # ----------------------------------------------------------------------------------------------------- training groups
    groups = {}          # name -> (sorted unique hashes, edition id of first occurrence, edition names)
    exact = {}           # name -> set of text keys
    gaz = {}             # name -> Counter of lower-cased LOC surface strings; and Counter of all occurrences of the string as a word sequence (later)
    train_docs = defaultdict(set)     # ours train: doc -> set of passage numbers
    train_eds = {}
    sets_def = [('ours_train', f'{a.data}/train.jsonl'), ('hist_gold_train', f'{a.agg}/hist_gold/train.jsonl')]
    if not a.skip_modern:
        sets_def.append(('modern_silver_train', f'{a.agg}/modern_silver/train.jsonl'))
    loc_strings = {}
    for name, path in sets_def:
        t0 = time.time()
        hs, eds = [], []
        ek = set()
        locs = Counter()
        ed_names = {}
        for r in read(path, F):
            nt = norm_tokens(r['tokens'])
            sh = shingles(nt)
            ed = edition_of(r)
            ei = ed_names.setdefault(ed, len(ed_names))
            hs.append(np.array(sh, dtype=np.int64))
            eds.append(np.full(len(sh), ei, dtype=np.int32))
            ek.add(text_key(nt))
            if r.get('loc_valid', True):
                toks = r['tokens']
                for s, e in tags_to_spans(r['ner_tags_loc']):
                    locs[' '.join(toks[s:e]).lower().replace('ſ', 's')] += 1
            if name == 'ours_train':
                train_docs[(ed, r['doc'])].add(r['passage'])
        h = np.concatenate(hs) if hs else np.zeros(0, np.int64)
        e = np.concatenate(eds) if eds else np.zeros(0, np.int32)
        del hs, eds
        order = np.argsort(h, kind='stable')
        h, e = h[order], e[order]
        keep = np.ones(len(h), dtype=bool)
        keep[1:] = h[1:] != h[:-1]
        groups[name] = (h[keep], e[keep], [k for k, _ in sorted(ed_names.items(), key=lambda kv: kv[1])])
        exact[name] = ek
        loc_strings[name] = locs
        log(f'{name}: {len(ek)} passages, {int(keep.sum())} distinct {N}-shingles, {len(locs)} LOC strings, {time.time() - t0:.0f}s')

    # ----------------------------------------------------------------------------------------------------- evaluation sets
    report = {'n_gram': N, 'training_groups': {k: {'distinct_shingles': int(len(v[0])), 'passages': len(exact[k])} for k, v in groups.items()}}
    all_loc_train = Counter()
    for c in loc_strings.values():
        all_loc_train.update(c)

    for split in ('test', 'dev'):
        recs = list(read(f'{a.data}/{split}.jsonl', F))
        rows = []
        per_ed = defaultdict(lambda: defaultdict(float))
        span_rows = []
        ex_dev_keys = set()
        seen_in_split = Counter()
        for r in recs:
            nt = norm_tokens(r['tokens'])
            sh = np.array(shingles(nt), dtype=np.int64)
            ed = r['edition']
            d = per_ed[ed]
            d['passages'] += 1
            d['words'] += len(r['tokens'])
            k = text_key(nt)
            res = {}
            for gname, (gh, ge, gnames) in groups.items():
                if len(sh):
                    pos = np.searchsorted(gh, sh)
                    pos[pos >= len(gh)] = 0
                    hit = gh[pos] == sh
                    frac = float(hit.mean())
                    if hit.any():
                        top = Counter(ge[pos[hit]].tolist()).most_common(1)[0]
                        top_ed = gnames[top[0]]
                    else:
                        top_ed = ''
                else:
                    frac, top_ed = 0.0, ''
                res[gname] = (frac, top_ed)
                d[f'{gname}_exact'] += k in exact[gname]
                d[f'{gname}_frac_sum'] += frac
                d[f'{gname}_ge50'] += frac >= 0.5
                d[f'{gname}_ge20'] += frac >= 0.2
                d[f'{gname}_ge05'] += frac >= 0.05
                if frac >= 0.2:
                    d[f'{gname}_src:{top_ed}'] += 1
            rows.append({'id': r['id'], 'edition': ed, 'doc': r['doc'], 'n_shingles': int(len(sh)),
                         **{f'{g}_frac': round(v[0], 3) for g, v in res.items()}, **{f'{g}_top_edition': v[1] for g, v in res.items()}})
            # gazetteer
            if r.get('loc_valid', True):
                toks = r['tokens']
                for s, e in tags_to_spans(r['ner_tags_loc']):
                    surf = ' '.join(toks[s:e]).lower().replace('ſ', 's')
                    seen_in_split[surf] += 1
                    span_rows.append({'id': r['id'], 'edition': ed, 's': s, 'e': e, 'surface': surf, 'train_ours': loc_strings['ours_train'].get(surf, 0),
                                      'train_hist_gold': loc_strings['hist_gold_train'].get(surf, 0), 'train_modern': loc_strings.get('modern_silver_train', Counter()).get(surf, 0)})
        # summaries
        eds = {}
        for ed, d in sorted(per_ed.items()):
            n = d['passages']
            row = {'passages': int(n), 'words': int(d['words'])}
            for g in groups:
                row[g] = {'exact_dup_passages': int(d[f'{g}_exact']), 'mean_shingle_overlap': round(d[f'{g}_frac_sum'] / n, 4),
                          'passages_ge50': int(d[f'{g}_ge50']), 'passages_ge20': int(d[f'{g}_ge20']), 'passages_ge05': int(d[f'{g}_ge05']),
                          'source_editions_of_ge20': {k.split(':', 1)[1]: int(v) for k, v in d.items() if k.startswith(f'{g}_src:')}}
            eds[ed] = row
        gz = defaultdict(lambda: Counter())
        for s in span_rows:
            ed = s['edition']
            gz[ed]['spans'] += 1
            gz[ed]['seen_ours'] += s['train_ours'] > 0
            gz[ed]['seen_any'] += (s['train_ours'] + s['train_hist_gold'] + s['train_modern']) > 0
            gz[ed]['seen_ours_ge3'] += s['train_ours'] >= 3
        tot = Counter()
        for c in gz.values():
            tot.update(c)
        report[split] = {'editions': eds, 'gazetteer': {ed: dict(c) for ed, c in gz.items()}, 'gazetteer_total': dict(tot),
                         'passages': len(recs)}
        with open(out / f'{split}_passage_overlap.tsv', 'w') as f:
            keys = list(rows[0].keys())
            f.write('\t'.join(keys) + '\n')
            for r_ in rows:
                f.write('\t'.join(str(r_[k]) for k in keys) + '\n')
        with open(out / f'{split}_loc_spans_seen.tsv', 'w') as f:
            keys = ['id', 'edition', 's', 'e', 'surface', 'train_ours', 'train_hist_gold', 'train_modern']
            f.write('\t'.join(keys) + '\n')
            for r_ in span_rows:
                f.write('\t'.join(str(r_[k]) for k in keys) + '\n')
        log(f'{split}: done {len(recs)} passages')
        if split == 'dev':
            # same documents in train and dev
            by_ed = defaultdict(lambda: Counter())
            for r in recs:
                key = (r['edition'], r['doc'])
                by_ed[r['edition']]['passages'] += 1
                if key in train_docs:
                    by_ed[r['edition']]['doc_in_train'] += 1
                    if any(abs(r['passage'] - p) <= 1 for p in train_docs[key]):
                        by_ed[r['edition']]['neighbour_in_train'] += 1
            report['dev_docs'] = {k: dict(v) for k, v in by_ed.items()}
            report['dev_docs_total'] = dict(sum((Counter(v) for v in by_ed.values()), Counter()))
        else:
            # test documents vs training documents (path basename / doc)
            tdocs = {(r['edition'], r['doc']) for r in recs}
            train_doc_names = {os.path.basename(d) for (_, d) in train_docs}
            report['test_docs'] = {'documents': len(tdocs), 'doc_path_in_train_same_edition': sum(1 for k in tdocs if k in train_docs),
                                   'doc_basename_in_other_train_edition': sorted({f'{e}:{os.path.basename(d)}' for e, d in tdocs if os.path.basename(d) in train_doc_names})[:50]}

    # ----------------------------------------------------------------------------------------------------- duplicates inside ours (train) across editions
    # one pass over the training passages: normalised text key -> editions
    t0 = time.time()
    key_eds = defaultdict(set)
    for r in read(f'{a.data}/train.jsonl', ['id', 'edition', 'tokens']):
        nt = norm_tokens(r['tokens'])
        if len(nt) >= 20:
            key_eds[text_key(nt)].add(r['edition'])
    multi = {k: v for k, v in key_eds.items() if len(v) > 1}
    pairs = Counter()
    for v in multi.values():
        v = sorted(v)
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs[f'{v[i]} | {v[j]}'] += 1
    report['train_cross_edition_exact_duplicates'] = {'passages_in_several_editions': len(multi), 'pairs': dict(pairs.most_common(30))}
    log('cross-edition dup', len(multi), f'{time.time() - t0:.0f}s')
    (out / 'leak_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log('done')


if __name__ == '__main__':
    os.environ.setdefault('PYTHONHASHSEED', '0')
    main()
