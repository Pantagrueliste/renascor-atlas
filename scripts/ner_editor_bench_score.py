"""Score NER predictions on the editor-tag bench (RQ/derived/atlas/ner/v2/editor_bench): the editors' own place and person tags, removed
from the text the model reads, are the reference. Never run on gold v2.

    python scripts/ner_editor_bench_score.py --bench editor_bench/bench.jsonl --editions editor_bench/editions.json \
        --pred r3_mdeberta_HM=pred/r3_mdeberta_HM.jsonl.gz --pred gemma=raw/gemma.jsonl [--ref r3_mdeberta_HM] \
        [--subset ids.txt] [--boot 2000] --out scores.json [--md scores.md] [--samples-dir samples/]

Predictions: gz or plain jsonl, one line per passage, either {"id", "LOC": [{"s","e"}], "PER": [{"s","e"}]} (token indices, end
exclusive: scripts/ner_audit_predict.py) or the teacher output {"id", "spans": [{"start","end","label"}], "error"} (renner_*_teacher.py).
A passage without a prediction line is not scored for that system (coverage is reported); with --subset, every system is scored on the
same ids (a missing line then counts as an empty prediction).

Views
  LOC_editor  the editors' LOC tags (``ner_tags_loc_editor``).
  LOC_v2      the corrected LOC gold of test_v2 (``loc_gold_v2``), the 7 editions of the v1 test only; ``loc_mask`` tokens are masked.
  PER         the editors' PER tags (``ner_tags_per_editor``).
  In every view the tokens of the ``ignored`` spans of that label (generic words dropped by the reader) are masked: gold and predicted
  spans touching a masked token are removed from both sides (the test_v2 rule).
Which edition is scored, and how, comes from ``editions.json`` (``measures``): None = not scored; "R" = recall only; "R+P" = recall,
precision and F1 (the label is exhaustive in the edition, or the v2 view). "(biaisé)" editions are recall-only and outside main_score.
Recall uses every passage of a scored edition. Precision / F1 use only the passages of an R+P edition that v1 marked valid for the label
(``loc_valid_v1`` / ``per_valid_v1`` not False, as in test_v2); R is reported again on that subset so that P, R and F1 agree.

Modes
  strict   exact token boundaries.
  overlap  a gold span is found if a prediction overlaps it; a prediction is right if it overlaps a gold span.
  rule     (PER only) the edition's ``per_match``: exact / exact_with_titles -> strict; head_overlap -> the prediction overlaps the gold span
           and covers its first token (Koenigsfelden: the editor's span holds name, office and domicile); overlap -> overlap.
Groups: main (editions with main_score) and all, each overall and per language (passage ``lang``); and per edition.
Owner-rule view (--owner-rules; off by default, the default scores are unchanged): before scoring, the editors' gold spans of every view are
mapped to the owner's decisions of 2026-10-25 (guide_annotation.md v0.2) by scripts/ner_owner_rules.py: final period trimmed; LOC on peoples,
demonyms, place adjectives and languages dropped (Latin -ensis kept); generic-building LOC dropped; PER reduced to the name (titles, address,
kinship removed; koenigsfelden: name only, office and domicile cut); title-only PER dropped; dignity + place PER dropped and the place added
to the LOC_editor gold (Vespasiano, Cavriana, Kaiserin, Ferdinand: « Duca di Nevers », « roi de France »); sovereign + numeral + place split;
God and personification PER dropped. The rule mode keeps each edition's per_match on the mapped spans (head_overlap: the prediction covers
the first token of the name-only span). Predictions are never changed. Counts per edition and mapping are in meta.owner_rules;
--owner-log writes every changed gold span. What cannot be mapped is listed in ner_owner_rules.UNMAPPED (also in meta).
Bootstrap: documents ((edition, doc)) resampled with replacement over the whole scored set (seed 1, the same draws for every system, view,
mode and group, as scripts/ner2_score_testv2.py); 95% percentile intervals; with --ref, paired differences with the reference system.

New finds: predicted spans of a label that touch no editor span of that label (``spans`` and ``nested``), in the passages of editions that
tag the label (``loc_tagged`` / ``per_tagged``). Category: other_label (touches an editor span of the other label), org (touches an editor
orgName), v2_confirmed / v2_rejected / v2_masked (LOC, v2 view available: overlaps a v2 gold span / not / touches loc_mask), new.
--samples-dir: for each system, a random sample of at most --sample-n (100) unjudged new finds (categories new, other_label, org) per
language and label, with context, for later judgement (<dir>/<system>__new_finds.jsonl; seed "editor-bench-new-finds").
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ratlas.ner.spans import tags_to_spans  # noqa: E402
import ner_owner_rules as OR  # noqa: E402

VIEWS = ('LOC_editor', 'LOC_v2', 'PER')
MODES = {'LOC_editor': ('strict', 'overlap'), 'LOC_v2': ('strict', 'overlap'), 'PER': ('strict', 'overlap', 'rule')}
RULE_MODE = {'exact': 'strict', 'exact_with_titles': 'strict', 'head_overlap': 'head', 'overlap': 'overlap'}
SAMPLE_SEED = 'editor-bench-new-finds'


# ---- input ------------------------------------------------------------------------------------------------------------------------------
def opn(p):
    return gzip.open(p, 'rt', encoding='utf-8') if str(p).endswith('.gz') else open(p, encoding='utf-8')


def to_pred(d: dict) -> dict:
    """One prediction line -> {"LOC": {(s, e)}, "PER": {(s, e)}} (token spans, end exclusive)."""
    out = {'LOC': set(), 'PER': set()}
    if 'spans' in d and 'LOC' not in d and 'PER' not in d:
        for s in d.get('spans') or []:
            if s.get('label') in out:
                out[s['label']].add((int(s['start']), int(s['end'])))
        return out
    for lab in out:
        for x in d.get(lab) or []:
            s, e = (x['s'], x['e']) if isinstance(x, dict) else (x[0], x[1])
            out[lab].add((int(s), int(e)))
    return out


def read_preds(path) -> tuple[dict, int]:
    preds, n_err = {}, 0
    with opn(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            n_err += bool(d.get('error'))
            preds[d['id']] = to_pred(d)
    return preds, n_err


def char_to_tokens(starts, toks, cs, ce) -> list[int]:
    return [k for k, (s, t) in enumerate(zip(starts, toks)) if s < ce and cs < s + len(t)]


def char_spans_to_token_spans(r, items, label=None) -> set:
    out = set()
    for x in items:
        if isinstance(x, (list, tuple)):          # orgs: [start, end] character pairs
            x = {'start': x[0], 'end': x[1]}
        if label and x.get('label') != label:
            continue
        ks = char_to_tokens(r['token_starts'], r['tokens'], x['start'], x['end'])
        if ks:
            out.add((ks[0], ks[-1] + 1))
    return out


def measure(m) -> tuple[bool, bool]:
    """editions.json measure string -> (recall scored, precision scored)."""
    if not m:
        return False, False
    head = m.split('(')[0]
    return 'R' in head, 'P' in head


def load_bench(path, editions, owner: bool = False, owner_stats: dict | None = None) -> list[dict]:
    """Per passage: gold spans, masks and flags of each view. owner=True: gold mapped to the owner rules (ner_owner_rules.owner_map);
    owner_stats (dict) receives {'counts': {edition: Counter}, 'changes': [...]}."""
    recs = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            ed = editions[r['edition']]
            meas = ed.get('measures') or {}
            mask = {lab: set() for lab in ('LOC', 'PER')}
            for x in r.get('ignored') or []:
                if x.get('label') in mask:
                    mask[x['label']].update(char_to_tokens(r['token_starts'], r['tokens'], x['start'], x['end']))
            views = {}
            rec_loc, prec_loc = measure(meas.get('LOC_editor'))
            if rec_loc:
                views['LOC_editor'] = {'gold': set(tags_to_spans(r['ner_tags_loc_editor'])), 'mask': mask['LOC'],
                                       'precision': prec_loc and r.get('loc_valid_v1') is not False, 'rule': 'strict'}
            rec_v2, prec_v2 = measure(meas.get('LOC_v2'))
            if rec_v2 and r.get('loc_gold_v2') is not None and r.get('loc_valid_v1') is not False:
                views['LOC_v2'] = {'gold': {tuple(x) for x in r['loc_gold_v2']}, 'mask': mask['LOC'] | set(r.get('loc_mask') or ()),
                                   'precision': prec_v2, 'rule': 'strict'}
            rec_per, prec_per = measure(meas.get('PER'))
            if rec_per:
                views['PER'] = {'gold': set(tags_to_spans(r['ner_tags_per_editor'])), 'mask': mask['PER'],
                                'precision': prec_per and r.get('per_valid_v1') is not False, 'rule': RULE_MODE[ed.get('per_match', 'overlap')]}
            known = {lab: char_spans_to_token_spans(r, (r.get('spans') or []) + (r.get('nested') or []), lab) for lab in ('LOC', 'PER')}
            if owner:
                loc_o, per_o, v2_o, log, changes = OR.owner_map(
                    r, r['edition'], views['LOC_editor']['gold'] if 'LOC_editor' in views else None,
                    views['PER']['gold'] if 'PER' in views else None, views['LOC_v2']['gold'] if 'LOC_v2' in views else None,
                    add_loc='LOC_editor' in views)
                if loc_o is not None:
                    views['LOC_editor']['gold'] = loc_o
                    known['LOC'] |= loc_o
                if per_o is not None:
                    views['PER']['gold'] = per_o
                if v2_o is not None:
                    views['LOC_v2']['gold'] = v2_o
                if owner_stats is not None:
                    owner_stats.setdefault('counts', {}).setdefault(r['edition'], Counter()).update(log)
                    for lab, before, after, maps, adds in changes:
                        owner_stats.setdefault('changes', []).append({
                            'id': r['id'], 'edition': r['edition'], 'label': lab, 'before': list(before),
                            'before_text': ' '.join(r['tokens'][before[0]:before[1]]), 'after': list(after) if after else None,
                            'after_text': ' '.join(r['tokens'][after[0]:after[1]]) if after else None, 'mappings': maps,
                            'loc_from_per': [' '.join(r['tokens'][a:b]) for a, b in adds]})
            recs.append({'id': r['id'], 'edition': r['edition'], 'doc': r['doc'], 'lang': r.get('lang') or 'und', 'main': bool(ed.get('main_score')),
                         'tagged': {'LOC': bool(ed.get('loc_tagged')), 'PER': bool(ed.get('per_tagged'))}, 'views': views, 'mask': mask,
                         'known': known, 'orgs': char_spans_to_token_spans(r, r.get('orgs') or []),
                         'v2': ({tuple(x) for x in r['loc_gold_v2']}, set(r.get('loc_mask') or ())) if r.get('loc_gold_v2') is not None else None,
                         'text': r['text'], 'token_starts': r['token_starts'], 'tokens': r['tokens']})
    return recs


# ---- matching ---------------------------------------------------------------------------------------------------------------------------
def drop(spans, mask: set) -> set:
    if not mask:
        return set(spans)
    return {sp for sp in spans if not any(t in mask for t in range(sp[0], sp[1]))}


def _ov(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _head(p, g) -> bool:
    return p[0] <= g[0] < p[1]


def match(gold: set, pred: set, mode: str) -> tuple[int, int, int, int]:
    """(gold found, gold total, predictions right, predictions total)."""
    if mode == 'strict':
        tp = len(gold & pred)
        return tp, len(gold), tp, len(pred)
    ok = _ov if mode == 'overlap' else _head
    found = sum(1 for g in gold if any(ok(p, g) for p in pred))
    right = sum(1 for p in pred if any(ok(p, g) for g in gold))
    return found, len(gold), right, len(pred)


def passage_counts(view: dict, pred: set, mode: str) -> tuple[int, int, int, int]:
    m = view['rule'] if mode == 'rule' else mode
    return match(drop(view['gold'], view['mask']), drop(pred, view['mask']), m)


# ---- new finds --------------------------------------------------------------------------------------------------------------------------
def new_find_category(r: dict, lab: str, sp) -> str | None:
    """None if the prediction touches an editor span of its label (or a masked token); else its category."""
    if any(t in r['mask'][lab] for t in range(sp[0], sp[1])):
        return None
    if any(_ov(sp, k) for k in r['known'][lab]):
        return None
    other = 'PER' if lab == 'LOC' else 'LOC'
    if any(_ov(sp, k) for k in r['known'][other]):
        return 'other_label'
    if any(_ov(sp, k) for k in r['orgs']):
        return 'org'
    if lab == 'LOC' and r['v2'] is not None:
        g, m = r['v2']
        if any(t in m for t in range(sp[0], sp[1])):
            return 'v2_masked'
        return 'v2_confirmed' if any(_ov(sp, k) for k in g) else 'v2_rejected'
    return 'new'


def span_chars(r, sp):
    s = r['token_starts'][sp[0]]
    e = r['token_starts'][sp[1] - 1] + len(r['tokens'][sp[1] - 1])
    return s, e


def sample_key(*parts) -> str:
    return hashlib.sha1('|'.join(map(str, (SAMPLE_SEED,) + parts)).encode()).hexdigest()


# ---- statistics -------------------------------------------------------------------------------------------------------------------------
def ratio(a, b):
    return a / np.maximum(b, 1)


def summarize(dc: np.ndarray, W: np.ndarray, with_p: bool) -> tuple[dict, dict]:
    """dc: (n_docs, 4) counts (found, gold, right, pred) of one group. -> (summary, bootstrap draws)."""
    f, g, rt, p = dc.sum(0)
    T = W @ dc
    rb = ratio(T[:, 0], T[:, 1])
    o = {'gold': int(g), 'found': int(f), 'r': round(float(f / g), 4) if g else None,
         'r_ci95': [round(float(x), 4) for x in np.percentile(rb, [2.5, 97.5])] if g else None, 'pred': int(p)}
    draws = {'r': rb}
    if with_p:
        pb = ratio(T[:, 2], T[:, 3])
        fb = 2 * pb * rb / np.maximum(pb + rb, 1e-12)
        P = rt / p if p else 0.0
        R = f / g if g else 0.0
        o.update({'right': int(rt), 'p': round(float(P), 4) if p else None, 'p_ci95': [round(float(x), 4) for x in np.percentile(pb, [2.5, 97.5])],
                  'f1': round(float(2 * P * R / (P + R)), 4) if P + R else 0.0, 'f1_ci95': [round(float(x), 4) for x in np.percentile(fb, [2.5, 97.5])]})
        draws.update({'p': pb, 'f1': fb})
    return o, draws


def ci(x) -> list:
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bench', required=True)
    ap.add_argument('--editions', required=True)
    ap.add_argument('--pred', action='append', required=True, help='NAME=PATH (repeatable)')
    ap.add_argument('--ref', help='reference system for paired differences')
    ap.add_argument('--subset', help='file of passage ids (one per line): every system is scored on these ids only')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--md', help='markdown summary tables')
    ap.add_argument('--samples-dir', help='write the new-finds samples here')
    ap.add_argument('--sample-n', type=int, default=100)
    ap.add_argument('--paired-all-groups', action='store_true', help='paired differences for every group (also all, all:lang, edition)')
    ap.add_argument('--owner-rules', action='store_true', help='owner-rule view: map the editor gold to the owner decisions (default: off)')
    ap.add_argument('--owner-log', help='with --owner-rules: write every changed gold span here (jsonl)')
    a = ap.parse_args(argv)
    if 'gold2' in a.bench or 'gold_v2' in a.bench:
        ap.error('gold v2 is the final test set')
    if a.owner_log and not a.owner_rules:
        ap.error('--owner-log needs --owner-rules')

    editions = json.load(open(a.editions, encoding='utf-8'))
    owner_stats: dict = {}
    recs = load_bench(a.bench, editions, owner=a.owner_rules, owner_stats=owner_stats)
    if a.subset:
        keep = {l.strip() for l in open(a.subset, encoding='utf-8') if l.strip()}
        recs = [r for r in recs if r['id'] in keep]
    systems = []
    for spec in a.pred:
        name, _, path = spec.partition('=')
        if not path:
            ap.error(f'--pred {spec}: NAME=PATH expected')
        systems.append((name, path))
    names = [n for n, _ in systems]
    if a.ref and a.ref not in names:
        ap.error(f'--ref {a.ref} is not a --pred name')

    preds, errors = {}, {}
    for name, path in systems:
        preds[name], errors[name] = read_preds(path)
    # passages scored: with --subset all of them (for every system); else those covered by every system
    if a.subset:
        scored = recs
    else:
        scored = [r for r in recs if all(r['id'] in preds[n] for n in names)]
    N = len(scored)
    docs = {}
    for r in scored:
        docs.setdefault((r['edition'], r['doc']), len(docs))
    nd = len(docs)
    doc_idx = np.array([docs[(r['edition'], r['doc'])] for r in scored], dtype=np.int64)
    rng = np.random.default_rng(1)
    W = np.zeros((a.boot, nd), dtype=np.float32)
    for b in range(a.boot):
        W[b] = np.bincount(rng.integers(0, nd, size=nd), minlength=nd)

    lang = np.array([r['lang'] for r in scored])
    edition = np.array([r['edition'] for r in scored])
    main_m = np.array([r['main'] for r in scored], dtype=bool)
    groups = {'main': main_m, 'all': np.ones(N, dtype=bool)}
    for l in sorted(set(lang)):
        groups[f'main:lang={l}'] = main_m & (lang == l)
        groups[f'all:lang={l}'] = lang == l
    for e in sorted(set(edition)):
        groups[f'edition={e}'] = edition == e
    has = {v: np.array([v in r['views'] for r in scored], dtype=bool) for v in VIEWS}
    has_p = {v: np.array([v in r['views'] and r['views'][v]['precision'] for r in scored], dtype=bool) for v in VIEWS}

    def doc_counts(c: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = np.zeros((nd, 4), dtype=np.float32)
        for j in range(4):
            out[:, j] = np.bincount(doc_idx[m], weights=c[m, j], minlength=nd)
        return out

    out = {'meta': {'bench': str(a.bench), 'n_passages_scored': N, 'n_documents': nd, 'subset': a.subset,
                    'bootstrap': {'draws': a.boot, 'seed': 1, 'unit': 'document (edition, doc)'},
                    'systems': {n: {'path': p, 'lines': len(preds[n]), 'lines_with_error': errors[n],
                                    'bench_passages_covered': sum(r['id'] in preds[n] for r in recs)} for n, p in systems},
                    'passages_per_group': {g: int(m.sum()) for g, m in groups.items()},
                    'rules': 'recall on every passage of a scored edition; P/F1 only where measures has P and the passage is v1-valid for the label'},
           'scores': {}, 'new_finds': {}}
    if a.owner_rules:   # only with the option: the default output is byte-identical to before
        out['meta']['owner_rules'] = {'on': True, 'decisions': 'renner/conventions_proprietaire_2026-10-25.md', 'mapper': 'scripts/ner_owner_rules.py',
                                      'counts_per_edition': {e: dict(sorted(c.items())) for e, c in sorted(owner_stats.get('counts', {}).items())},
                                      'gold_spans_changed': len(owner_stats.get('changes', [])), 'unmapped': OR.UNMAPPED}
    draws_all: dict = {}
    samples: dict = {}
    for name in names:
        P = preds[name]
        sc: dict = {}
        for view in VIEWS:
            lab = view[:3]
            for mode in MODES[view]:
                c = np.zeros((N, 4), dtype=np.int64)
                for k, r in enumerate(scored):
                    v = r['views'].get(view)
                    if v is None:
                        continue
                    c[k] = passage_counts(v, P.get(r['id'], {}).get(lab, set()), mode)
                res = {}
                for g, gm in groups.items():
                    mr = gm & has[view]
                    if not mr.any():
                        continue
                    o_r, d_r = summarize(doc_counts(c, mr), W, with_p=False)
                    e = {'passages': int(mr.sum()), 'recall': o_r}
                    draws_all[(name, view, mode, g, 'r')] = d_r['r']
                    mp = gm & has_p[view]
                    if mp.any():
                        o_p, d_p = summarize(doc_counts(c, mp), W, with_p=True)
                        e['exhaustive'] = dict(o_p, passages=int(mp.sum()))
                        for key in ('p', 'f1'):
                            draws_all[(name, view, mode, g, key)] = d_p[key]
                    res[g] = e
                sc.setdefault(view, {})[mode] = res
        out['scores'][name] = sc

        # new finds
        nf = {'by_lang': defaultdict(Counter), 'by_edition': defaultdict(Counter)}
        tot = defaultdict(Counter)
        cand = defaultdict(list)
        for r in scored:
            p = P.get(r['id'], {})
            for lab in ('LOC', 'PER'):
                if not r['tagged'][lab]:
                    continue
                for sp in sorted(p.get(lab, ())):
                    if not (0 <= sp[0] < sp[1] <= len(r['tokens'])):
                        continue
                    tot[lab]['predicted'] += 1
                    cat = new_find_category(r, lab, sp)
                    if cat is None:
                        continue
                    for key, grp in ((r['lang'], 'by_lang'), (r['edition'], 'by_edition')):
                        nf[grp][f'{lab}|{key}'][cat] += 1
                        nf[grp][f'{lab}|{key}']['total'] += 1
                    tot[lab][cat] += 1
                    tot[lab]['no_editor_tag'] += 1
                    if cat in ('new', 'other_label', 'org'):
                        cs, ce = span_chars(r, sp)
                        cand[(r['lang'], lab)].append((sample_key(name, r['id'], lab, sp[0], sp[1]), {
                            'system': name, 'id': r['id'], 'edition': r['edition'], 'lang': r['lang'], 'label': lab, 's': sp[0], 'e': sp[1],
                            'cs': cs, 'ce': ce, 'text': r['text'][cs:ce], 'category': cat,
                            'left': r['text'][max(0, cs - 100):cs], 'right': r['text'][ce:ce + 100], 'judgement': None}))
        out['new_finds'][name] = {'total': {lab: dict(c) for lab, c in tot.items()},
                                  'by_lang': {k: dict(v) for k, v in sorted(nf['by_lang'].items())},
                                  'by_edition': {k: dict(v) for k, v in sorted(nf['by_edition'].items())}}
        samples[name] = [x for key in sorted(cand) for _, x in sorted(cand[key], key=lambda t: t[0])[:a.sample_n]]

    if a.ref:
        paired = {}
        for name in names:
            if name == a.ref:
                continue
            for (n_, view, mode, g, key), d in draws_all.items():
                if n_ != name or (a.ref, view, mode, g, key) not in draws_all:
                    continue
                if not (a.paired_all_groups or g == 'main' or g.startswith('main:lang=')):
                    continue
                diff = d - draws_all[(a.ref, view, mode, g, key)]
                sec = 'recall' if key == 'r' else 'exhaustive'
                k2 = 'r' if key == 'r' else key
                x = out['scores'][name][view][mode][g][sec][k2]
                y = out['scores'][a.ref][view][mode][g][sec][k2]
                if x is None or y is None:
                    continue
                paired.setdefault(name, {}).setdefault(view, {}).setdefault(mode, {})[f'{g}:{key}'] = {'diff': round(x - y, 4), 'ci95': ci(diff)}
        out['paired_vs_ref'] = {'ref': a.ref, 'diffs': paired}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    if a.samples_dir:
        sd = Path(a.samples_dir)
        sd.mkdir(parents=True, exist_ok=True)
        for name, rows in samples.items():
            with open(sd / f'{name}__new_finds.jsonl', 'w', encoding='utf-8') as fh:
                for x in rows:
                    fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    if a.owner_log:
        Path(a.owner_log).parent.mkdir(parents=True, exist_ok=True)
        with open(a.owner_log, 'w', encoding='utf-8') as fh:
            for x in owner_stats.get('changes', []):
                fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    if a.md:
        md = render_md(out, names)
        if a.owner_rules:
            md = ('Vue « règles du propriétaire » (décisions du 2026-10-25) : or des éditeurs transformé par scripts/ner_owner_rules.py. '
                  f"{out['meta']['owner_rules']['gold_spans_changed']} empans d'or changés.\n\n") + md
        Path(a.md).write_text(md, encoding='utf-8')
    print(f'{N} passages, {nd} documents, {len(names)} systems -> {a.out}', file=sys.stderr)
    return out


def _fmt(v, c=None):
    if v is None:
        return '–'
    s = f'{100 * v:.1f}'
    if c:
        s += f' [{100 * c[0]:.1f}–{100 * c[1]:.1f}]'
    return s


def render_md(out: dict, names: list) -> str:
    """Tables: per view and mode, rows = groups 'main' and 'main:lang=..', one recall column and one P/F1 column per system."""
    lines = []
    for view in VIEWS:
        for mode in MODES[view]:
            groups = sorted({g for n in names for g in out['scores'][n].get(view, {}).get(mode, {}) if g == 'main' or g.startswith('main:lang=')},
                            key=lambda g: (g != 'main', g))
            if not groups:
                continue
            lines += [f'### {view}, {mode}', '', '| groupe | gold | ' + ' | '.join(f'{n} R' for n in names) + ' | ' +
                      ' | '.join(f'{n} P / F1' for n in names) + ' |', '|---|---:|' + '---:|' * (2 * len(names))]
            for g in groups:
                ref = next((out['scores'][n][view][mode][g] for n in names if g in out['scores'][n][view][mode]), None)
                cells = [g.replace('main:lang=', ''), str(ref['recall']['gold'] if ref else '')]
                for n in names:
                    e = out['scores'][n][view][mode].get(g)
                    cells.append(_fmt(e['recall']['r'], e['recall']['r_ci95']) if e else '–')
                for n in names:
                    e = out['scores'][n][view][mode].get(g)
                    x = e.get('exhaustive') if e else None
                    cells.append(f"{_fmt(x['p'])} / {_fmt(x['f1'], x['f1_ci95'])} (n={x['gold']})" if x else '–')
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
