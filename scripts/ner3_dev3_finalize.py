"""dev3: from the candidate passages and the leak check, keep 120 passages per language.

1. passages flagged by the leak check (8-gram share >= --flag, or document name found in v1) are dropped;
2. the rest is trimmed to --per-lang by removing, one at a time, the passage that sits in the most over-represented cells: documents with several passages, authors above
   --author-share, pools above their target, period bins above their share, genres above --genre-share. Ties are broken by a seeded shuffle. No text is read for this choice,
   only metadata, and no model output is used.

    python scripts/ner3_dev3_finalize.py --cand cand_dev3.jsonl --leak leak_cand_dev3.json --out passages.jsonl --report finalize_report.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

TARGET_POOL = {  # final 120 per language
    'la': {'la_nos': 45, 'la_eebo': 30, 'la_ws': 20, 'la_sal': 25},
    'es': {'es_est': 44, 'es_sal': 36, 'es_wsn': 22, 'es_sga': 18},
    'it': {'it_ws': 28, 'it_ll': 36, 'it_bibit': 56},
}
BIN_SHARE = [0.18, 0.27, 0.32, 0.23]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--leak', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report')
    ap.add_argument('--per-lang', type=int, default=120)
    ap.add_argument('--flag', type=float, default=0.02)
    ap.add_argument('--author-share', type=float, default=0.15)
    ap.add_argument('--genre-share', type=float, default=0.35)
    ap.add_argument('--max-per-doc', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--drop-pools', nargs='*', default=[], help='pools left out after reading samples (e.g. OCR too damaged)')
    ap.add_argument('--drop-regex', default='', help='drop passages whose text matches (OCR page furniture)')
    ap.add_argument('--la-lexicon', help='json word counts of clean Latin (folded); Latin passages above --oov-nos / --oov-other unknown-word share are dropped')
    ap.add_argument('--oov-nos', type=float, default=0.10)
    ap.add_argument('--oov-other', type=float, default=0.14)
    ap.add_argument('--langs', nargs='*', default=['la', 'es', 'it'])
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cand = [json.loads(l) for l in open(a.cand, encoding='utf-8') if l.strip()]
    leak = {o['id']: o for o in json.load(open(a.leak))['per_passage']}
    import re
    junk = re.compile(a.drop_regex) if a.drop_regex else None
    lexj = json.load(open(a.la_lexicon)) if a.la_lexicon else None
    if lexj is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from ratlas.ner.ner2_passages import ALPHA, fold

    def oov(seg):
        ws = [fold(w) for w in re.findall(rf'[{ALPHA}]+', seg)]
        ws = [w for w in ws if len(w) >= 4]
        return sum(lexj.get(w, 0) < 2 for w in ws) / max(1, len(ws))

    out, dropped_leak, dropped_quality = [], [], []
    for r in cand:
        if lexj is not None and r['lang'] == 'la' and (oov(r['text']) > (a.oov_nos if r['pool'] == 'la_nos' else a.oov_other) or r['text'].count('¶') > 1):
            dropped_quality.append(r['id'])
            continue
        if r['pool'] in a.drop_pools or (junk and junk.search(r['text'])):
            dropped_quality.append(r['id'])
            continue
        o = leak.get(r['id'])
        if o is None:
            raise SystemExit(f'no leak result for {r["id"]}')
        r['leak_share'] = o['share']
        r['leak_n_shingles'] = o['n_found']
        r['leak_doc_in_v1'] = o['doc_name_in_v1']
        if o['share'] >= a.flag or o['doc_name_in_v1']:
            dropped_leak.append(r['id'])
        else:
            out.append(r)
    final, removed = [], []
    for lang in a.langs:
        rs = [r for r in out if r['lang'] == lang]
        rng.shuffle(rs)
        N = a.per_lang
        cdoc = Counter(r['doc'] for r in rs)
        keep = Counter()
        rs2 = []
        for r in rs:                                      # hard document cap (rs is shuffled: the extra passages of a document are random ones)
            keep[r['doc']] += 1
            if keep[r['doc']] > a.max_per_doc:
                removed.append(r['id'])
            else:
                rs2.append(r)
        rs = rs2
        cap = int(a.author_share * N)
        while True:                                       # hard author cap first
            ca = Counter(r['author'] for r in rs)
            over = [x for x, c in ca.items() if c > cap]
            if not over:
                break
            cd = Counter(r['doc'] for r in rs)
            cands = [i for i, r in enumerate(rs) if r['author'] in over]
            i = max(cands, key=lambda i: cd[rs[i]['doc']])
            removed.append(rs.pop(i)['id'])
        while len(rs) > N:
            cd = Counter(r['doc'] for r in rs)
            ca = Counter(r['author'] for r in rs)
            cp = Counter(r['pool'] for r in rs)
            cb = Counter(r['period_bin'] for r in rs)
            cg = Counter(r['genre'] for r in rs)
            n = len(rs)
            best, bi = -1, -1
            for i, r in enumerate(rs):
                pen = 0.4 * max(0, cd[r['doc']] - 1)
                pen += 2.0 * max(0, ca[r['author']] - a.author_share * N) / max(1, a.author_share * N) * 6
                tp = TARGET_POOL[lang].get(r['pool'], N * 0.1)
                pen += 8.0 * max(0, cp[r['pool']] - tp) / tp
                tb = BIN_SHARE[r['period_bin']] * N
                pen += 3.0 * max(0, cb[r['period_bin']] - 1.1 * tb) / tb
                tg = a.genre_share * N
                pen += 3.0 * max(0, cg[r['genre']] - tg) / tg
                if pen > best:
                    best, bi = pen, i
            removed.append(rs.pop(bi)['id'])
        final += rs
    final.sort(key=lambda r: (r['lang'], r['year'], r['id']))
    for r in final:
        r['seed'] = False
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rep = {'candidates': len(cand), 'dropped_by_leak_check': dropped_leak, 'dropped_for_quality': dropped_quality, 'removed_by_trimming': len(removed), 'kept': len(final)}
    for lang in a.langs:
        rs = [r for r in final if r['lang'] == lang]
        ws = [len(r['text'].split()) for r in rs]
        rep[lang] = {'n': len(rs), 'docs': len({r['doc'] for r in rs}), 'authors': len({r['author'] for r in rs}), 'max_per_doc': max(Counter(r['doc'] for r in rs).values()),
                     'top_author': Counter(r['author'] for r in rs).most_common(3), 'pools': dict(Counter(r['pool'] for r in rs)), 'collections': dict(Counter(r['collection'] for r in rs)),
                     'period_bins': dict(sorted(Counter(r['period_bin'] for r in rs).items())), 'genres': dict(Counter(r['genre'] for r in rs).most_common()),
                     'words_min_median_max': [min(ws), sorted(ws)[len(ws) // 2], max(ws)], 'tokens': sum(len(r['tokens']) for r in rs),
                     'year_min_max': [min(r['year'] for r in rs), max(r['year'] for r in rs)], 'max_leak_share': max(r['leak_share'] for r in rs)}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
