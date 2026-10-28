"""Annotation completeness of an edition: how many of the names the editors tag are left untagged elsewhere in the same text?

    python -m ratlas.ner.complete SLUG... --data ~/renascor-atlas-data/ner/data_v1 [--out DIR]

Method (per edition and per label, LOC and PER separately)
  * candidates: the strings the edition tags (LOC spans for LOC, PER spans for PER) that
        - start with a capital letter and have >= 3 characters,
        - are not also tagged with the other label somewhere in the edition (``Lincoln``, ``Paris`` tagged as person and as place),
        - are not common words: a single-word string whose lower-case form is frequent in the edition itself
          (>= 3 times and >= 25% of the capitalised occurrences) or that is a closed-class word of the stop-word lists;
  * tagged = the labelled spans whose text is a candidate (counted once each);
  * untagged = occurrences of a candidate in the primary text (token match, exact case, longest candidate first) that touch no span at all
    (neither a label, nor a nested tag, nor an ignored pronoun / common noun, nor an organisation name);
  * untagged share = untagged / (tagged + untagged).
  Two pools: the documents that have at least one tagged span (used for the verdict) and every sampled document (reported: it shows the
  editions that tagged only part of their documents).
  Verdict: ``exhaustive`` (<= 10% untagged), ``partial`` (<= 50%), ``sparse`` (> 50%), ``too_few`` (fewer than 30 tagged spans of the label),
  ``none`` (no tagged span of the label).
Each passage gets ``u_loc`` / ``u_per`` (untagged candidates it holds) and up to 5 examples, so that a partial edition can be filtered
passage by passage (``splits.py``).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .langs import _SWSETS
from .tei import tokenize

EXHAUSTIVE = 0.10
PARTIAL = 0.50
MIN_SPANS = 30
STOP = frozenset(w for s in _SWSETS.values() for w in s) | frozenset(
    '''in an auf von zu der die das den dem des ein eine et cum ad de ex per pro sub the and of to a i o il el la le les lo los las
    herr frau sankt saint san santo santa sant sanct divus diuus dominus domina dom don donna mr mrs dr'''.split())


def read_passages(path: Path) -> list[dict]:
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def _surface(p: dict, s: dict) -> str:
    return p['text'][s['start']:s['end']]


def _first_upper(t: str) -> bool:
    return bool(t) and t[0].isupper()


def verdict(share: Optional[float], n_tagged: int) -> str:
    if n_tagged == 0:
        return 'none'
    if n_tagged < MIN_SPANS or share is None:
        return 'too_few'
    return 'exhaustive' if share <= EXHAUSTIVE else ('partial' if share <= PARTIAL else 'sparse')


def analyse(passages: list[dict]) -> tuple[dict, list[dict]]:
    """-> (report of the edition, passages with u_loc / u_per / untagged examples)."""
    toks: list[list[tuple[int, int]]] = []
    words: list[list[str]] = []
    exact: Counter = Counter()
    lower: Counter = Counter()
    surf = {'LOC': Counter(), 'PER': Counter()}
    has_spans: dict = defaultdict(bool)
    for p in passages:
        tk = tokenize(p['text'])
        ws = [p['text'][a:b] for a, b in tk]
        toks.append(tk)
        words.append(ws)
        for w in ws:
            exact[w] += 1
            if w.islower():
                lower[w] += 1
        for s in p['spans']:
            seg = _surface(p, s)
            tt = tuple(seg[a:b] for a, b in tokenize(seg))
            surf[s['label']][tt] += 1
            has_spans[p['path']] = True
    other = {'LOC': 'PER', 'PER': 'LOC'}
    cands: dict = {}
    for lab in ('LOC', 'PER'):
        oth = {tuple(x.lower() for x in k) for k in surf[other[lab]]}
        cs = {}
        for tt, n in surf[lab].items():
            text = ' '.join(tt)
            if len(''.join(tt)) < 3 or not _first_upper(tt[0]) or not any(ch.isalpha() for ch in text):
                continue
            if tuple(x.lower() for x in tt) in oth:
                continue
            if len(tt) == 1:
                low = tt[0].lower()
                if low in STOP:
                    continue
                lc = lower.get(low, 0)
                if lc >= 3 and lc >= 0.25 * exact.get(tt[0], 0):
                    continue
            cs[tt] = n
        cands[lab] = cs
    first: dict = {}
    for lab in ('LOC', 'PER'):
        for tt in cands[lab]:
            first.setdefault(tt[0], []).append((tt, lab))
    for k in first:
        first[k].sort(key=lambda x: -len(x[0]))
    # tagged occurrences of candidates, untagged occurrences
    tag_n = {'LOC': {'docs': 0, 'all': 0}, 'PER': {'docs': 0, 'all': 0}}
    unt_n = {'LOC': {'docs': 0, 'all': 0}, 'PER': {'docs': 0, 'all': 0}}
    unt_types = {'LOC': Counter(), 'PER': Counter()}
    pass_unt = {'LOC': {'docs': [0, 0], 'all': [0, 0]}, 'PER': {'docs': [0, 0], 'all': [0, 0]}}   # [passages with untagged, passages]
    out = []
    for p, tk, ws in zip(passages, toks, words):
        cov = bytearray(len(p['text']))
        for a, b in [(s['start'], s['end']) for k in ('spans', 'nested', 'ignored') for s in p.get(k, [])] + [tuple(o) for o in p.get('orgs', [])]:
            cov[a:b] = b'\x01' * (b - a)
        u = {'LOC': 0, 'PER': 0}
        ex = []
        i = 0
        n = len(ws)
        while i < n:
            hit = first.get(ws[i])
            if hit:
                for tt, lab in hit:
                    L = len(tt)
                    if tuple(ws[i:i + L]) == tt:
                        a, b = tk[i][0], tk[i + L - 1][1]
                        if not any(cov[a:b]):
                            u[lab] += 1
                            unt_types[lab][' '.join(tt)] += 1
                            if len(ex) < 5:
                                ex.append([lab, ' '.join(tt)])
                        i += L - 1
                        break
            i += 1
        tagged_here = {'LOC': 0, 'PER': 0}
        for s in p['spans']:
            seg = _surface(p, s)
            tt = tuple(seg[a:b] for a, b in tokenize(seg))
            if tt in cands[s['label']]:
                tagged_here[s['label']] += 1
        indocs = has_spans[p['path']]
        for lab in ('LOC', 'PER'):
            tag_n[lab]['all'] += tagged_here[lab]
            unt_n[lab]['all'] += u[lab]
            pass_unt[lab]['all'][0] += 1 if u[lab] else 0
            pass_unt[lab]['all'][1] += 1
            if indocs:
                tag_n[lab]['docs'] += tagged_here[lab]
                unt_n[lab]['docs'] += u[lab]
                pass_unt[lab]['docs'][0] += 1 if u[lab] else 0
                pass_unt[lab]['docs'][1] += 1
        q = dict(p)
        q['u_loc'], q['u_per'], q['untagged_examples'] = u['LOC'], u['PER'], ex
        out.append(q)
    rep: dict = {'passages': len(passages), 'docs': len({p['path'] for p in passages}),
                 'docs_with_spans': sum(1 for v in has_spans.values() if v),
                 'words': sum(p['words'] for p in passages)}
    for lab in ('LOC', 'PER'):
        key = lab.lower()
        ntag = sum(surf[lab].values())
        for pool in ('docs', 'all'):
            t, un = tag_n[lab][pool], unt_n[lab][pool]
            rep[f'{key}_tagged_cand_{pool}'] = t
            rep[f'{key}_untagged_{pool}'] = un
            rep[f'{key}_untagged_share_{pool}'] = round(un / (t + un), 4) if (t + un) else None
            pu = pass_unt[lab][pool]
            rep[f'{key}_passages_with_untagged_{pool}'] = round(pu[0] / pu[1], 4) if pu[1] else None
        rep[f'{key}_spans'] = ntag
        rep[f'{key}_types'] = len(surf[lab])
        rep[f'{key}_candidate_types'] = len(cands[lab])
        rep[f'{key}_candidate_share_of_spans'] = round(sum(cands[lab].values()) / ntag, 4) if ntag else None
        rep[f'{key}_verdict'] = verdict(rep[f'{key}_untagged_share_docs'], ntag)
        rep[f'{key}_top_untagged'] = unt_types[lab].most_common(12)
    return rep, out


def _one(arg):
    slug, data = arg
    src = data / 'raw' / 'passages' / f'{slug}.jsonl.gz'
    if not src.exists():
        return slug, None
    ps = read_passages(src)
    rep, scored = analyse(ps)
    rep['slug'] = slug
    (data / 'scored').mkdir(parents=True, exist_ok=True)
    (data / 'completeness').mkdir(parents=True, exist_ok=True)
    tmp = data / 'scored' / f'{slug}.jsonl.gz.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
        for q in scored:
            fh.write(json.dumps(q, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(data / 'scored' / f'{slug}.jsonl.gz')
    (data / 'completeness' / f'{slug}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    return slug, rep


def main(argv=None) -> int:
    import multiprocessing as mp
    ap = argparse.ArgumentParser(prog='ratlas.ner.complete')
    ap.add_argument('slugs', nargs='*', help='default: every edition that has passages')
    ap.add_argument('--data', type=Path, required=True, help='.../data_v1 (raw/passages -> scored/, completeness/)')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args(argv)
    slugs = a.slugs or sorted(f.name[:-len('.jsonl.gz')] for f in (a.data / 'raw' / 'passages').glob('*.jsonl.gz'))
    # largest editions first
    slugs = sorted(slugs, key=lambda s: -(a.data / 'raw' / 'passages' / f'{s}.jsonl.gz').stat().st_size if (a.data / 'raw' / 'passages' / f'{s}.jsonl.gz').exists() else 0)
    args = [(s, a.data) for s in slugs]
    it = mp.get_context('fork').Pool(a.workers).imap_unordered(_one, args) if a.workers > 1 else map(_one, args)
    for slug, rep in it:
        if rep is None:
            print(f'[{slug}] no passages', file=sys.stderr)
            continue
        print(f'[{slug}] LOC {rep["loc_verdict"]} {rep["loc_untagged_share_docs"]} | PER {rep["per_verdict"]} {rep["per_untagged_share_docs"]}',
              file=sys.stderr, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
