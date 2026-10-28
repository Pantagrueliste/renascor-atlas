"""train / dev / test, the 15% cap, HF-compatible JSONL and the datasheet.

    python -m ratlas.ner.splits --data ~/renascor-atlas-data/ner/data_v1 [--meta ~/renascor-atlas-data/ner/meta] [--seed ner-v1]

Reads   <data>/scored/<slug>.jsonl.gz (ner.complete) and <data>/completeness/<slug>.json
Writes  <data>/{train,dev,test}.jsonl            tokens + BIO tags (+ text and character spans), all kept passages
        <data>/{loc,per,locper}/{train,dev,test}.jsonl   the views: only the passages whose label(s) are validly annotated
        <data>/labels.json, <data>/dataset_info.json
        <data>/tables/*.tsv, <data>/DATASHEET.md

Rules (all deterministic, hash based)
  * a passage is dropped when it holds a span that is longer than the markup allows (``dropped_span``), when its document has no tagged span at
    all (``untagged_doc``: the editors did not tag that document), when its text is an exact duplicate of another one (``duplicate``),
    or when the verdict of its edition for the label is ``sparse`` / ``too_few`` / ``none`` (then the label is not valid for the passage);
  * ``loc_valid`` = the LOC verdict of the edition is exhaustive, or partial and the passage holds no untagged LOC candidate; same for ``per_valid``;
    a passage is kept when at least one of the two is valid;
  * test = whole editions held out per language (``choose_test``); the editions that share texts (``editions.OVERLAP_GROUPS``) stay together;
  * dev = held-out documents of the training editions (a unit = a document, or a block of 25 passages in an edition with fewer than 12 documents);
  * dev and test keep only the CLEAN passages (no untagged candidate of the valid labels);
  * train: no single edition above 15% of the training tokens (``cap_tokens``: water-filling, the passages of a capped edition are sampled by hash);
  * a train passage that shares >= 20% of its sampled 8-gram shingles with a dev or test passage is removed (leakage check).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from . import editions as E
from .complete import read_passages
from .tei import tokenize

LABELS = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']
LOC_LABELS = ['O', 'B-LOC', 'I-LOC']
PER_LABELS = ['O', 'B-PER', 'I-PER']
CAP = 0.15
DEV_SHARE = 0.05
BLOCK = 25
MIN_TEST_LOC = 300
SHINGLE = 8
LEAK_SHARE = 0.20


def h32(s: str) -> float:
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) / 2 ** 32


def norm_key(text: str) -> str:
    return hashlib.sha1(re.sub(r'[\W_]+', '', text.lower()).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# tokens and tags
# ---------------------------------------------------------------------------------------------------------------------
def tokens_and_tags(text: str, spans: list[dict]) -> tuple[list[str], list[int], list[int], list[int], list[int]]:
    """Tokens (a token that straddles a span boundary is split) and the BIO tags (joint, LOC only, PER only) plus the token offsets."""
    import bisect
    cuts = sorted({x for s in spans for x in (s['start'], s['end'])})
    toks: list[tuple[int, int]] = []
    for a, b in tokenize(text):
        i = bisect.bisect_right(cuts, a)
        while i < len(cuts) and cuts[i] < b:
            toks.append((a, cuts[i]))
            a = cuts[i]
            i += 1
        toks.append((a, b))
    sp = sorted(spans, key=lambda s: s['start'])
    starts = [s['start'] for s in sp]
    tags, loc, per = [], [], []
    prev = None
    for a, b in toks:
        k = bisect.bisect_right(starts, a) - 1
        if k < 0 or not (sp[k]['start'] <= a and b <= sp[k]['end']):
            k = None
        t = 0
        if k is not None:
            first = k != prev
            t = (1 if first else 2) if sp[k]['label'] == 'LOC' else (3 if first else 4)
        prev = k
        tags.append(t)
        loc.append(t if t in (0, 1, 2) else 0)
        per.append(t - 2 if t in (3, 4) else 0)
    return [text[a:b] for a, b in toks], tags, loc, per, [a for a, _ in toks]


def shingles(text: str) -> set:
    ws = re.findall(r'\w+', text.lower())
    out = set()
    for i in range(len(ws) - SHINGLE + 1):
        h = zlib.crc32(' '.join(ws[i:i + SHINGLE]).encode('utf-8'))   # stable across processes (hash() is salted)
        if h % 8 == 0:
            out.add(h)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# validity and filters
# ---------------------------------------------------------------------------------------------------------------------
def validity(p: dict, comp: dict) -> tuple[bool, bool]:
    def ok(verdict: str, u: int) -> bool:
        return verdict == 'exhaustive' or (verdict == 'partial' and u == 0)
    return ok(comp.get('loc_verdict', 'none'), p.get('u_loc', 0)), ok(comp.get('per_verdict', 'none'), p.get('u_per', 0))


def cap_tokens(sizes: dict, cap: float = CAP) -> dict:
    """Per-edition token allowance so that no edition is above ``cap`` of the total (water-filling)."""
    if not sizes:
        return {}
    if len(sizes) * cap < 1:         # fewer than 1/cap editions: the cap cannot hold, nothing is cut (the datasheet says so)
        return dict(sizes)
    c = float(max(sizes.values()))
    for _ in range(200):
        tot = sum(min(v, c) for v in sizes.values())
        nc = cap * tot
        if abs(nc - c) < 0.5:
            c = nc
            break
        c = nc
    return {k: int(min(v, c)) for k, v in sizes.items()}


def dominant_lang(ps: list[dict]) -> tuple[Optional[str], float]:
    c = Counter()
    for p in ps:
        c[p.get('lang') or 'None'] += p['words']
    if not c:
        return None, 0.0
    l, n = c.most_common(1)[0]
    return (None if l == 'None' else l), n / sum(c.values())


def choose_test(info: dict) -> tuple[dict, list[dict]]:
    """Whole editions held out for the test, per language.

    info[slug] = {'lang': dominant language, 'lang_share', 'loc_spans' (valid, clean), 'group': overlap group}. A language needs at least two
    units (groups) with >= MIN_TEST_LOC valid LOC spans and a dominant-language share >= 0.7; the held-out unit is the one whose LOC span count is
    closest to 10% of the language's total (never the largest unit; an edition marked ``not_test`` in ``editions.EXTRA`` is never held out). With five or more units two are held out.
    -> ({slug: language}, notes)
    """
    by_lang: dict = defaultdict(lambda: defaultdict(lambda: {'slugs': [], 'loc': 0}))
    for slug, d in info.items():
        if d['lang'] and d['lang_share'] >= 0.7 and d['loc_spans'] >= MIN_TEST_LOC and not E.EXTRA.get(slug, {}).get('not_test'):
            u = by_lang[d['lang']][d['group']]
            u['slugs'].append(slug)
            u['loc'] += d['loc_spans']
    test: dict = {}
    notes: list[dict] = []
    for lang, units in sorted(by_lang.items()):
        total = sum(u['loc'] for u in units.values())
        names = sorted(units)
        if len(names) < 2:
            notes.append({'lang': lang, 'note': f'only one edition unit with >= {MIN_TEST_LOC} valid LOC spans ({names[0]}): no edition held out'})
            continue
        n_hold = 2 if len(names) >= 5 else 1
        biggest = max(names, key=lambda g: (units[g]['loc'], g))
        pool = [g for g in names if g != biggest]
        chosen = []
        for _ in range(n_hold):
            g = min((x for x in pool if x not in chosen), key=lambda x: (abs(units[x]['loc'] - 0.10 * total), x), default=None)
            if g is None:
                break
            chosen.append(g)
        for g in chosen:
            for s in units[g]['slugs']:
                test[s] = lang
        notes.append({'lang': lang, 'note': f'{len(names)} edition units; held out: {", ".join(chosen)} '
                                          f'({sum(units[g]["loc"] for g in chosen)} of {total} valid LOC spans)'})
    return test, notes


# ---------------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------------
def build(data: Path, meta_dir: Optional[Path], seed: str = 'ner-v1', cap: float = CAP) -> dict:
    meta = E.load_meta(meta_dir / 'edition_meta.tsv' if meta_dir else None)
    comps = {}
    for f in sorted((data / 'completeness').glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        comps[d['slug']] = d
    summaries = {}
    for f in sorted((data / 'raw' / 'summary').glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        summaries[d['slug']] = d
    excl: dict = defaultdict(Counter)                   # slug -> reason -> passages
    excl_w: dict = defaultdict(Counter)                 # slug -> reason -> words
    kept: dict = defaultdict(list)                      # slug -> passages (with validity)
    seen_hash: dict = {}
    for slug in sorted(comps):
        ps = read_passages(data / 'scored' / f'{slug}.jsonl.gz')
        comp = comps[slug]
        docs_with_spans = {p['path'] for p in ps if p['spans']}
        for p in ps:
            def drop(reason):
                excl[slug][reason] += 1
                excl_w[slug][reason] += p['words']
            if p['dropped_span']:
                drop('dropped_span')
                continue
            if p['path'] not in docs_with_spans:
                drop('untagged_doc')
                continue
            lv, pv = validity(p, comp)
            if not (lv or pv):
                drop('label_not_valid')           # sparse / too few / none, or partial with an untagged candidate
                continue
            k = norm_key(p['text'])
            if k in seen_hash:
                drop('duplicate')
                continue
            seen_hash[k] = slug
            p['loc_valid'], p['per_valid'] = lv, pv
            kept[slug].append(p)
    # ---- editions: language, group, sizes
    info = {}
    for slug, ps in kept.items():
        lang, share = dominant_lang(ps)
        clean_loc = sum(len([s for s in p['spans'] if s['label'] == 'LOC']) for p in ps if p['loc_valid'] and p['u_loc'] == 0)
        info[slug] = {'lang': lang, 'lang_share': share, 'loc_spans': clean_loc, 'group': E.overlap_group(slug)}
    test_map, test_notes = choose_test(info)
    # the whole overlap group of a test edition is test
    for slug in list(test_map):
        for g in E.OVERLAP_GROUPS:
            if slug in g:
                for o in g:
                    if o in kept:
                        test_map[o] = test_map[slug]
    split: dict = {'train': [], 'dev': [], 'test': []}
    pool: dict = defaultdict(list)
    for slug, ps in kept.items():
        if slug in test_map:
            for p in ps:
                lv = p['loc_valid'] and p['u_loc'] == 0
                pv = p['per_valid'] and p['u_per'] == 0
                if not (lv or pv):
                    excl[slug]['test_not_clean'] += 1
                    excl_w[slug]['test_not_clean'] += p['words']
                    continue
                p['loc_valid'], p['per_valid'] = lv, pv
                split['test'].append(p)
        else:
            pool[slug] = ps
    # ---- dev: held-out documents (or blocks) of the training editions
    train_pool: dict = defaultdict(list)
    dev_by_slug: dict = defaultdict(list)
    for slug, ps in pool.items():
        ndocs = len({p['path'] for p in ps})
        for p in ps:
            unit = p['path'] if ndocs >= 12 else f'{p["path"]}#{p["i"] // BLOCK}'
            if h32(f'{seed}|dev|{unit}') < DEV_SHARE:
                dev_by_slug[slug].append(p)
            else:
                train_pool[slug].append(p)
    # dev: clean passages only, and the same 15% cap on its tokens
    dev_pass: list = []
    for slug, ps in dev_by_slug.items():
        cl = []
        for p in ps:
            lv = p['loc_valid'] and p['u_loc'] == 0
            pv = p['per_valid'] and p['u_per'] == 0
            if not (lv or pv):
                excl[slug]['dev_not_clean'] += 1
                excl_w[slug]['dev_not_clean'] += p['words']
                continue
            p['loc_valid'], p['per_valid'] = lv, pv
            cl.append(p)
        dev_by_slug[slug] = cl
    dsizes = {s: sum(p['words'] for p in ps) for s, ps in dev_by_slug.items() if ps}
    dallow = cap_tokens(dsizes, cap)
    for slug, ps in dev_by_slug.items():
        budget = dallow.get(slug, 0)
        acc = 0
        for p in sorted(ps, key=lambda p: h32(f'{seed}|devcap|{p["id"]}')):
            if acc + p['words'] > budget:
                excl[slug]['dev_cap'] += 1
                excl_w[slug]['dev_cap'] += p['words']
                continue
            acc += p['words']
            dev_pass.append(p)
    split['dev'] = dev_pass
    # ---- train cap
    sizes = {s: sum(p['words'] for p in ps) for s, ps in train_pool.items() if ps}
    allow = cap_tokens(sizes, cap)
    for slug, ps in train_pool.items():
        budget = allow.get(slug, 0)
        acc = 0
        for p in sorted(ps, key=lambda p: h32(f'{seed}|train|{p["id"]}')):
            if acc + p['words'] > budget:
                excl[slug]['train_cap'] += 1
                excl_w[slug]['train_cap'] += p['words']
                continue
            acc += p['words']
            split['train'].append(p)
    # ---- leakage: shingles of dev and test against train
    held = set()
    for sp in ('dev', 'test'):
        for p in split[sp]:
            held |= shingles(p['text'])
    keep_train = []
    for p in split['train']:
        sh = shingles(p['text'])
        if sh and len(sh & held) >= LEAK_SHARE * len(sh):
            excl[p['slug']]['leak_with_dev_test'] += 1
            excl_w[p['slug']]['leak_with_dev_test'] += p['words']
            continue
        keep_train.append(p)
    split['train'] = keep_train
    for sp in split:
        split[sp].sort(key=lambda p: p['id'])
    return {'split': split, 'excl': excl, 'excl_w': excl_w, 'info': info, 'test_map': test_map, 'test_notes': test_notes,
            'comps': comps, 'summaries': summaries, 'meta': meta, 'allow': allow, 'sizes': sizes, 'kept': kept}


# ---------------------------------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------------------------------
def record(p: dict, split: str, meta: dict) -> dict:
    toks, tags, loc, per, offs = tokens_and_tags(p['text'], p['spans'])
    m = meta.get(p['slug'], {})
    return {'id': p['id'], 'split': split, 'edition': p['slug'], 'doc': p['path'], 'passage': p['i'], 'lang': p.get('lang'),
            'lang_src': p.get('lang_src'), 'year': p.get('year'), 'licence': p['licence'], 'tier': p['tier'],
            'restricted': p['tier'] == 'T3' or m.get('no_derivatives') == 'yes',
            'text': p['text'], 'spans': [{'start': s['start'], 'end': s['end'], 'label': s['label'], 'tag': s['tag']} for s in p['spans']],
            'tokens': toks, 'token_starts': offs, 'ner_tags': tags, 'ner_tags_loc': loc, 'ner_tags_per': per,
            'loc_valid': p['loc_valid'], 'per_valid': p['per_valid'], 'u_loc': p.get('u_loc', 0), 'u_per': p.get('u_per', 0),
            'flags': p.get('flags', []), 'words': p['words']}


def write_all(res: dict, data: Path) -> dict:
    meta = res['meta']
    stats: dict = defaultdict(Counter)       # (split, slug, lang) -> counters
    for view in ('', 'loc', 'per', 'locper'):
        (data / view if view else data).mkdir(parents=True, exist_ok=True)
    handles = {}
    for sp in ('train', 'dev', 'test'):
        handles[('', sp)] = open(data / f'{sp}.jsonl', 'w', encoding='utf-8')
        for view in ('loc', 'per', 'locper'):
            handles[(view, sp)] = open(data / view / f'{sp}.jsonl', 'w', encoding='utf-8')
    try:
        for sp in ('train', 'dev', 'test'):
            for p in res['split'][sp]:
                r = record(p, sp, meta)
                line = json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n'
                handles[('', sp)].write(line)
                if r['loc_valid']:
                    v = {k: r[k] for k in r if k not in ('ner_tags', 'ner_tags_per')}
                    handles[('loc', sp)].write(json.dumps(v, ensure_ascii=False, separators=(',', ':')) + '\n')
                if r['per_valid']:
                    v = {k: r[k] for k in r if k not in ('ner_tags', 'ner_tags_loc')}
                    handles[('per', sp)].write(json.dumps(v, ensure_ascii=False, separators=(',', ':')) + '\n')
                if r['loc_valid'] and r['per_valid']:
                    v = {k: r[k] for k in r if k not in ('ner_tags_loc', 'ner_tags_per')}
                    handles[('locper', sp)].write(json.dumps(v, ensure_ascii=False, separators=(',', ':')) + '\n')
                key = (sp, r['edition'], r['lang'] or 'None')
                c = stats[key]
                c['passages'] += 1
                c['tokens'] += len(r['tokens'])
                c['words'] += r['words']
                c['LOC'] += sum(1 for s in p['spans'] if s['label'] == 'LOC')
                c['PER'] += sum(1 for s in p['spans'] if s['label'] == 'PER')
                c['loc_valid_passages'] += int(r['loc_valid'])
                c['per_valid_passages'] += int(r['per_valid'])
                c['LOC_in_loc_valid'] += sum(1 for s in p['spans'] if s['label'] == 'LOC') if r['loc_valid'] else 0
                c['PER_in_per_valid'] += sum(1 for s in p['spans'] if s['label'] == 'PER') if r['per_valid'] else 0
                c['no_span_passages'] += int(not p['spans'])
    finally:
        for h in handles.values():
            h.close()
    (data / 'labels.json').write_text(json.dumps({'ner_tags': LABELS, 'ner_tags_loc': LOC_LABELS, 'ner_tags_per': PER_LABELS}, indent=1))
    (data / 'dataset_info.json').write_text(json.dumps({
        'description': 'Place-name (LOC) and person-name (PER) spans tagged by the editors of TEI editions; private, not for publication.',
        'features': {'id': 'string', 'edition': 'string', 'doc': 'string', 'lang': 'string', 'year': 'int32', 'licence': 'string', 'tier': 'string',
                     'text': 'string', 'spans': 'list of {start, end, label, tag} (character offsets in text)', 'tokens': 'list of string',
                     'token_starts': 'list of int32', 'ner_tags': 'list of ClassLabel ' + json.dumps(LABELS),
                     'ner_tags_loc': 'list of ClassLabel ' + json.dumps(LOC_LABELS), 'ner_tags_per': 'list of ClassLabel ' + json.dumps(PER_LABELS),
                     'loc_valid': 'bool', 'per_valid': 'bool'},
        'splits': {sp: len(res['split'][sp]) for sp in ('train', 'dev', 'test')},
        'usage': "datasets.load_dataset('json', data_files={'train': 'loc/train.jsonl', 'validation': 'loc/dev.jsonl', 'test': 'loc/test.jsonl'})"},
        indent=1))
    return stats


def tsv(path: Path, header: list, rows: Iterable) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def md_table(header: list, rows: list) -> str:
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def write_tables(res: dict, stats: dict, data: Path) -> dict:
    t = data / 'tables'
    t.mkdir(parents=True, exist_ok=True)
    comps, summ, meta, info = res['comps'], res['summaries'], res['meta'], res['info']
    slugs = sorted(set(comps) | set(summ))
    # completeness table
    crow = []
    for s in slugs:
        c = comps.get(s)
        if not c:
            crow.append([s] + [''] * 15)
            continue
        crow.append([s, meta.get(s, {}).get('licence', ''), c['passages'], c['docs'], c['docs_with_spans'], c['loc_spans'], c['loc_candidate_types'],
                     c['loc_untagged_share_docs'], c['loc_untagged_share_all'], c['loc_verdict'], c['per_spans'], c['per_candidate_types'],
                     c['per_untagged_share_docs'], c['per_untagged_share_all'], c['per_verdict'], c['words']])
    hdr_c = ['edition', 'licence', 'passages', 'docs', 'docs_with_spans', 'loc_spans', 'loc_candidate_types', 'loc_untagged_share_taggeddocs',
             'loc_untagged_share_alldocs', 'loc_verdict', 'per_spans', 'per_candidate_types', 'per_untagged_share_taggeddocs',
             'per_untagged_share_alldocs', 'per_verdict', 'words']
    tsv(t / 'completeness.tsv', hdr_c, crow)
    # counts per split / edition / language
    rows = []
    for (sp, slug, lang), c in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rows.append([sp, slug, lang, c['passages'], c['tokens'], c['words'], c['LOC'], c['PER'], c['loc_valid_passages'], c['per_valid_passages'],
                     c['LOC_in_loc_valid'], c['PER_in_per_valid'], c['no_span_passages']])
    hdr_s = ['split', 'edition', 'lang', 'passages', 'tokens', 'words', 'LOC_spans', 'PER_spans', 'loc_valid_passages', 'per_valid_passages',
             'LOC_spans_in_loc_valid', 'PER_spans_in_per_valid', 'passages_without_span']
    tsv(t / 'counts_by_split_edition_lang.tsv', hdr_s, rows)
    lang_rows = defaultdict(Counter)
    for (sp, slug, lang), c in stats.items():
        for k in ('passages', 'tokens', 'LOC', 'PER'):
            lang_rows[(sp, lang)][k] += c[k]
        lang_rows[(sp, lang)]['editions'] += 1
    tot = {sp: sum(c['tokens'] for (s2, l), c in lang_rows.items() if s2 == sp) for sp in ('train', 'dev', 'test')}
    lrows = [[sp, lang, c['editions'], c['passages'], c['tokens'], round(100 * c['tokens'] / tot[sp], 1) if tot[sp] else 0, c['LOC'], c['PER']]
             for (sp, lang), c in sorted(lang_rows.items(), key=lambda kv: (kv[0][0], -kv[1]['tokens']))]
    hdr_l = ['split', 'lang', 'editions', 'passages', 'tokens', 'share_of_split_tokens_pct', 'LOC_spans', 'PER_spans']
    tsv(t / 'language_mix.tsv', hdr_l, lrows)
    # exclusions
    erows = []
    reasons = sorted({r for s in res['excl'] for r in res['excl'][s]})
    for s in slugs:
        erows.append([s] + [res['excl'].get(s, {}).get(r, 0) for r in reasons] + [res['excl_w'].get(s, {}).get(r, 0) for r in reasons])
    tsv(t / 'exclusions.tsv', ['edition'] + [f'passages_{r}' for r in reasons] + [f'words_{r}' for r in reasons], erows)
    # edition summary
    per_slug = defaultdict(Counter)
    for (sp, slug, lang), c in stats.items():
        per_slug[slug][f'{sp}_tokens'] += c['tokens']
        per_slug[slug][f'{sp}_passages'] += c['passages']
        per_slug[slug][f'{sp}_LOC'] += c['LOC']
        per_slug[slug][f'{sp}_PER'] += c['PER']
    srows = []
    for s in slugs:
        m = meta.get(s, {})
        sm = summ.get(s, {})
        i = info.get(s, {})
        role = 'test' if s in res['test_map'] else ('train/dev' if per_slug[s]['train_tokens'] or per_slug[s]['dev_tokens'] else 'unused')
        srows.append([s, m.get('licence', ''), m.get('tier', ''), m.get('no_derivatives', ''), m.get('languages', ''), m.get('period', ''),
                      sm.get('files_total', ''), sm.get('words_total', ''), sm.get('files_tagged_total', ''), sm.get('passages', ''),
                      i.get('lang', ''), round(i.get('lang_share', 0), 2) if i else '', role,
                      per_slug[s]['train_tokens'], per_slug[s]['dev_tokens'], per_slug[s]['test_tokens'],
                      per_slug[s]['train_LOC'], per_slug[s]['dev_LOC'], per_slug[s]['test_LOC'],
                      per_slug[s]['train_PER'], per_slug[s]['dev_PER'], per_slug[s]['test_PER']])
    hdr_e = ['edition', 'licence', 'tier', 'no_derivatives', 'languages_catalogue', 'period', 'files_total', 'words_total', 'files_with_tags',
             'passages_extracted', 'dominant_lang', 'dominant_lang_share', 'role', 'train_tokens', 'dev_tokens', 'test_tokens', 'train_LOC',
             'dev_LOC', 'test_LOC', 'train_PER', 'dev_PER', 'test_PER']
    tsv(t / 'editions.tsv', hdr_e, srows)
    tsv(t / 'test_editions.tsv', ['edition', 'language'], sorted(res['test_map'].items()))
    # Italian (the language the model is wanted for) by split and edition, passage language 'it'
    hdr_i = ['split', 'edition', 'passages', 'tokens', 'LOC_spans', 'PER_spans', 'loc_valid_passages']
    irows = [[sp, slug, c['passages'], c['tokens'], c['LOC'], c['PER'], c['loc_valid_passages']]
             for (sp, slug, lang), c in sorted(stats.items()) if lang == 'it']
    tsv(t / 'italian.tsv', hdr_i, irows)
    # concentration of the spans (the token cap does not cap the spans: a register of names holds many more names per token)
    conc = []
    for lab in ('LOC', 'PER'):
        tot_l = sum(per_slug[s][f'train_{lab}'] for s in per_slug) or 1
        top = sorted(((per_slug[s][f'train_{lab}'], s) for s in per_slug), reverse=True)[:3]
        conc.append([lab, tot_l] + [f'{s} {round(100 * n / tot_l, 1)}%' for n, s in top])
    tsv(t / 'span_concentration_train.tsv', ['label', 'train_spans', 'top1', 'top2', 'top3'], conc)
    return {'completeness': (hdr_c, crow), 'langs': (hdr_l, lrows), 'editions': (hdr_e, srows), 'reasons': reasons, 'excl_rows': erows,
            'tot': tot, 'per_slug': per_slug, 'italian': (hdr_i, irows), 'conc': conc}


def datasheet(res: dict, tabs: dict, stats: dict, data: Path, seed: str) -> str:
    comps, meta, summ = res['comps'], res['meta'], res['summaries']
    hdr_c, crow = tabs['completeness']
    hdr_e, srows = tabs['editions']
    hdr_l, lrows = tabs['langs']
    n = {sp: len(res['split'][sp]) for sp in ('train', 'dev', 'test')}
    tot = tabs['tot']
    lab = {sp: Counter() for sp in n}
    for (sp, slug, lang), c in stats.items():
        lab[sp]['LOC'] += c['LOC']
        lab[sp]['PER'] += c['PER']
    verdicts = Counter((c.get('loc_verdict') for c in comps.values()))
    pverd = Counter((c.get('per_verdict') for c in comps.values()))
    big = max(((s, tabs['per_slug'][s]['train_tokens']) for s in tabs['per_slug']), key=lambda x: x[1], default=('', 0))
    lines = [
        '# DATASHEET: place and person NER data, v1 (private)',
        '',
        '**Status.** Private working data for a first, private fine-tuning experiment. Nothing here is uploaded or published (no GitHub, Hugging Face,',
        'Zenodo or site). Passages of editions with a restricted licence (tier T3, or no-derivatives licences) are flagged `restricted: true` in every',
        'record; any later release must filter them out and check the licence of each edition (table below).',
        '',
        '## 1. What this is',
        '',
        'Passages of at most ~200 words taken from the primary text of TEI editions, with character spans for the names the **editors** tagged:',
        '`LOC` (places) and `PER` (persons). The aim is a model that finds place names (secondarily person names) in editions that are **not** TEI-tagged.',
        '',
        f'- Passages: train {n["train"]}, dev {n["dev"]}, test {n["test"]}. Tokens: train {tot["train"]}, dev {tot["dev"]}, test {tot["test"]}.',
        f'- Spans: LOC train {lab["train"]["LOC"]} / dev {lab["dev"]["LOC"]} / test {lab["test"]["LOC"]}; PER train {lab["train"]["PER"]} / dev {lab["dev"]["PER"]} / test {lab["test"]["PER"]}.',
        f'- Seed of every hash decision: `{seed}`. Everything is deterministic (SHA-1 of ids and paths).',
        '',
        '## 2. Sources',
        '',
        'The editions are the collection modules of the Renascor Atlas (`ratlas.collections`); one edition = one module (slug). London Lives is excluded',
        'entirely (counting-only licence), the Medici Archive Project is not used, CELT (private) is only used inside this private data. Documents dated after 1800 are left out.',
        'Per edition, at most 8 M words of documents that carry tags and 2 M words of untagged documents were sampled (SHA-1 order of the path) for the extraction.',
        '',
        '### Editions, licences, roles',
        '',
        md_table(['edition', 'licence', 'tier', 'ND', 'role', 'dominant lang', 'train tok', 'dev tok', 'test tok', 'LOC (train/dev/test)', 'PER (train/dev/test)'],
                 [[r[0], r[1], r[2], r[3], r[12], f'{r[10]} ({r[11]})', r[13], r[14], r[15], f'{r[16]}/{r[17]}/{r[18]}', f'{r[19]}/{r[20]}/{r[21]}'] for r in srows]),
        '',
        '## 3. What is read and how (extraction)',
        '',
        '- Primary text only: every outermost `<body>`. Not read: `teiHeader`, `front`, `back`, registers (`listPlace`, `listPerson`, `listOrg`, `standOff`), `msIdentifier`,',
        '  regests (`abstract`), the apparatus (`rdg`, `witDetail`), running heads and catchwords (`fw`), figures, index markers, and the divisions the editors mark as',
        '  commentary / translation / regest / register (by `@type`, `@subtype`, `@decls`, or the `div_zone` of the edition).',
        '- Notes: only original ones (marginalia, glosses, interlinear). Untyped notes, footnotes and editorial notes are left out.',
        '- **One reading: diplomatic** (closest to the printed or written text, because the editions the model will be used on are mostly diplomatic transcriptions):',
        '  `choice`: `orig`/`abbr`/`sic` win over `reg`/`expan`/`corr` (if the diplomatic child holds no text, e.g. a glyph, the editorial reading is used and the passage gets the flag',
        '  `choice_fallback`); `subst`: `add` kept, `del` dropped; `app`: `lem` kept, `rdg` dropped; `supplied` and `ex` (editor additions) dropped; `lb break="no"` joins the word halves without space.',
        '  Passage flags record what happened (`choice`, `subst`, `app`, `join`, `ex_dropped`, `supplied_dropped`, `del_dropped`, `choice_fallback`, `mixed_lang`, `lang_conflict`).',
        '- Labels: `LOC` = placeName, settlement, country, region, district, geogName, bloc, and name/rs typed as place (plus the edition\'s own `type_map`, e.g. Casebooks `rs[@type=address]`);',
        '  `PER` = persName and name/rs typed as person. `orgName` is not a label (its text is `O`; a place or person tagged inside keeps its label). Outermost tag wins for nested tags,',
        '  except a LOC (PER) around several LOC (PER) children separated only by punctuation, which is split. Tagged pronouns and common nouns (`generic_terms`) and placeholders are `O`.',
        '- Passages are built from blocks (paragraphs, verse lines, ...) up to 200 words, never cutting a tagged span; long blocks are cut at sentence ends. Whitespace is normalised.',
        '- Tokens: words and single punctuation marks (`\\w+|[^\\w\\s]`); a token that straddles a span boundary is split. BIO tags: `O B-LOC I-LOC B-PER I-PER`.',
        '- Language of a passage: its own `xml:lang`, else the language of the document, else the language of the atlas mentions of the document, else a stop-word guess, else the edition\'s only language (`lang_src`).',
        '',
        '## 4. Annotation completeness',
        '',
        'Per edition and per label, the strings that the edition tags (capitalised, >= 3 characters, not also tagged with the other label, not common words) are searched in its own',
        'primary text; the occurrences that touch no span are "untagged". Untagged share = untagged / (tagged + untagged). Verdict: exhaustive <= 10%, partial <= 50%, sparse > 50%,',
        'too_few (< 30 spans), none. Verdicts are computed on the documents that have at least one tagged span (the training pool); the share over all sampled documents is also given.',
        f'LOC verdicts: {dict(verdicts)}. PER verdicts: {dict(pverd)}.',
        '',
        md_table(['edition', 'LOC spans', 'LOC untagged (tagged docs)', 'LOC untagged (all docs)', 'LOC verdict', 'PER spans', 'PER untagged (tagged docs)', 'PER untagged (all docs)', 'PER verdict'],
                 [[r[0], r[5], r[7], r[8], r[9], r[10], r[12], r[13], r[14]] for r in crow]),
        '',
        'Use of the verdicts: only exhaustive editions are used as they are; for a partial edition only the passages with no untagged candidate (`u_loc == 0`, `u_per == 0`) are used; sparse,',
        'too_few and none editions are not used for that label. `loc_valid` / `per_valid` tell, per passage, for which label the passage is a valid example. A passage is kept if at least one is valid.',
        'Train a LOC model on `loc/`, a PER model on `per/`, a joint model on `locper/`. An edition whose persons are not tagged in the text (Casebooks) therefore never teaches `PER`.',
        '',
        '## 5. Splits',
        '',
        '- **test**: whole editions held out per language, so the score measures transfer to new editions: ' + ('; '.join(f'{k} ({v})' for k, v in sorted(res['test_map'].items())) or 'none') + '.',
        '  Selection rule: a language needs at least two edition units (overlap groups count as one) with >= 300 valid clean LOC spans and a dominant-language share >= 0.7; the unit closest to 10% of the',
        '  language\'s LOC spans is held out (never the largest; two units when there are five or more). Notes by language: ' + ' | '.join(f'{x["lang"]}: {x["note"]}' for x in res['test_notes']),
        '- **dev**: held-out documents of the training editions (5% of the documents, hash order; blocks of 25 passages in editions with < 12 documents); clean passages only; same 15% cap.',
        f'- **train**: the rest, with no edition above 15% of the training tokens (largest edition after the cap: {big[0]}, {big[1]} tokens = {round(100 * big[1] / max(tot["train"], 1), 1)}%).',
        '  Casebooks is capped like the others.',
        '- Leakage: exact duplicate passages are removed (first edition in slug order keeps it); a train passage sharing >= 20% of its sampled 8-gram shingles with a dev or test passage is removed;',
        '  editions that share texts (`OVERLAP_GROUPS`: FreEM/Epistemon, DraCor-it/Biblioteca Italiana, DTA/Sandrart, CAMENA, Gongora, Selbstzeugnisse HAB, ELAN) are never split across train and test.',
        '',
        '### Language mix',
        '',
        md_table(hdr_l, lrows),
        '',
        '### Italian (passages whose language is `it`), by split and edition',
        '',
        md_table(*tabs['italian']),
        '',
        '### Concentration of the training spans',
        '',
        'The 15% cap is on tokens; a register of names holds many more spans per token, so the span counts are less even than the token counts:',
        '',
        md_table(['label', 'train spans', 'top 1', 'top 2', 'top 3'], tabs['conc']),
        '',
        '## 6. Exclusions',
        '',
        'Per edition, the passages and words left out and why: `tables/exclusions.tsv`. Reasons: `untagged_doc` (document without any tagged span), `label_not_valid` (edition sparse/none for both labels,',
        'or a partial edition\'s passage with an untagged candidate), `dropped_span` (a span of more than 15 words: markup error), `duplicate`, `test_not_clean` / `dev_not_clean`, `dev_cap`, `train_cap`',
        '(the 15% cap), `leak_with_dev_test`.',
        '',
        md_table(['reason', 'passages', 'words'], [[r, sum(res['excl'][s].get(r, 0) for s in res['excl']), sum(res['excl_w'][s].get(r, 0) for s in res['excl_w'])] for r in tabs['reasons']]),
        '',
        '## 7. Known limits',
        '',
        '- The labels are the editors\' tagging, not a gold standard: no edition is perfect, and the completeness measure only sees strings that the edition itself tags elsewhere.',
        '- Tagged `rs`/`name` typed as person or place can be descriptions rather than names; generic ones are removed, the rest is kept.',
        '- Diplomatic readings keep scribal abbreviations and symbols (`p m\'`, `ꝭ`, long s); some editions encode them as glyphs without text and fall back to the editorial reading.',
        '- Italian is thin: see the Italian table. Only one Italian-language edition (cavriana) has >= 300 valid LOC spans, so no whole Italian edition is held out by the rule; the Italian test passages are the Italian passages of held-out multilingual editions. For a strict Italian transfer test, hold out `cavriana` or `paesi_novamente` yourself and retrain.',
        '- `hearth_tax` is a register of names (hundreds of thousands of PER spans in few tokens): it is capped by tokens like the others, still dominates the PER spans, and is never held out as a test edition (`not_test`). Passages without any language evidence take the main language of their document (`lang_src` = `doc_majority`), for `hearth_tax` English (`edition_default`).',
        '- Periods and genres differ strongly (letters, laws, travel, court records); the dev set does not measure that, the test editions do.',
        '',
        '## 8. Files',
        '',
        '`{train,dev,test}.jsonl` (all kept passages), `{loc,per,locper}/{train,dev,test}.jsonl` (views), `labels.json`, `dataset_info.json`, `tables/`, `raw/` (extracted passages), `scored/` (with completeness counts).',
        'Fields: `id, split, edition, doc, passage, lang, lang_src, year, licence, tier, restricted, text, spans[{start,end,label,tag}], tokens, token_starts, ner_tags, ner_tags_loc, ner_tags_per,',
        'loc_valid, per_valid, u_loc, u_per, flags, words`.',
        '',
    ]
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas.ner.splits')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--meta', type=Path, default=None)
    ap.add_argument('--seed', default='ner-v1')
    a = ap.parse_args(argv)
    res = build(a.data, a.meta, a.seed)
    stats = write_all(res, a.data)
    tabs = write_tables(res, stats, a.data)
    (a.data / 'DATASHEET.md').write_text(datasheet(res, tabs, stats, a.data, a.seed), encoding='utf-8')
    print({sp: len(res['split'][sp]) for sp in res['split']}, 'test editions:', sorted(res['test_map']), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
