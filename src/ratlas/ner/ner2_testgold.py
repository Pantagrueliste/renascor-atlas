"""NER v2, test_v2 gold: the editors' LOC spans of the repaired test passages (``test_v2_loc.jsonl``) corrected with the judgements of the candidate groups.

Input
  * ``test_v2_loc.jsonl``   repaired passages, editor LOC spans as BIO tags (``ner_tags_loc``);
  * ``pred/<system>.jsonl.gz``  LOC (and PER) predictions of the 42 systems of the candidate pool (word spans ``s``/``e`` of the passage's own tokens);
  * ``judged_<edition>.jsonl``  one judgement per group (edition, normalised surface) written by six judges with six slightly different layouts
    (``canon_row`` maps all of them to one form).

Rules (the task text of the owner)
  * ``add``        a candidate occurrence judged a place that the editors left untagged: its span (the span proposed by most systems) joins the gold;
                   a ``fix`` / ``correct_span`` (wrong boundaries) adds the corrected span instead and the proposed span stays a wrong prediction;
  * ``ambiguous``  the tokens of the occurrence (the whole cluster of overlapping predicted spans) are masked: gold spans and predicted spans that touch a
                   masked token are dropped from the scoring. An editor span judged ``ambiguous`` is masked in the same way;
  * ``remove``     an editor span judged not to be a place leaves the gold (it was a person, a fragment, a heaven); ``keep`` leaves it in;
  * ``error``      the occurrence is a false positive: nothing changes.
Occurrences of a group that the judge did not look at get the group decision. When a group is ``mixed`` (the judged occurrences disagree) and the judge
gave no decision for the others, they are masked (``ambiguous``): precision and recall are then neither helped nor hurt by what nobody read.

The occurrences are rebuilt from the predictions exactly as ``ner2_cand.build`` did (same clustering, same surface choice, same group key), and the
count of every group is checked against the ``n`` of its judgement.

    python -m ratlas.ner.ner2_testgold build --data test_v2_loc.jsonl --full test_v2.jsonl --pred-dir pred --judged-dir . --out test_v2_gold.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .ner2_cand import clusters, family, norm_surface, read_preds, span_text
from .spans import spans_to_tags, tags_to_spans

DECISIONS = {'add', 'error', 'ambiguous', 'keep', 'remove'}
JUDGE = 'claude_model_assisted_to_be_verified_by_human'


# ---------------------------------------------------------------------------------------------------------------------
# judgement rows -> one form
# ---------------------------------------------------------------------------------------------------------------------
def _dec(x: Optional[str]) -> Optional[str]:
    return x if x in DECISIONS else None


def canon_row(row: dict) -> dict:
    """-> {'kind', 'edition', 'surface', 'n', 'default', 'occ': [{'id', 's', 'e', 'span', 'decision', 'fix'}], 'fixes': [..], 'mixed': bool}.

    ``default`` is the decision of the occurrences that carry no decision of their own. ``occ`` are the judged occurrences (``s``/``e`` may be None:
    then they are matched by passage id and surface text). ``fixes`` are corrected spans valid for every occurrence of the group."""
    kind = row['kind']
    raw = row.get('decision') or row.get('group_decision') or row.get('verdict')
    mixed = bool(row.get('mixed')) or raw == 'mixed'
    if raw == 'mixed':
        default = _dec(row.get('decision_unseen_occurrences') or row.get('unseen_decision') or row.get('decision_unjudged_occurrences')) or 'ambiguous'
    else:
        default = _dec(raw)
        if default is None:
            default = 'ambiguous'
    occ = []
    items = row.get('per_context') or row.get('occurrences') or row.get('contexts') or []
    for c in items:
        d = _dec(c.get('decision') or c.get('verdict'))
        if d is None:
            continue
        o = {'id': c['id'], 's': c.get('s'), 'e': c.get('e'), 'span': c.get('span'), 'decision': d}
        if c.get('correct_span'):
            o['fix'] = [c['correct_span']]
        occ.append(o)
    fixes = []
    for k in ('fix', 'fix_span'):
        if row.get(k):
            fixes += list(row[k])
    if row.get('correct_span') and not any('fix' in o for o in occ):
        fixes.append(row['correct_span'])
    by_ctx = {}
    for key, v in (row.get('fix_span_by_context') or {}).items():
        rid, _, s = key.partition('@')
        by_ctx[(rid, int(s) if s.isdigit() else None)] = list(v)
    return {'kind': kind, 'edition': row['edition'], 'surface': row['surface'], 'n': row.get('n'), 'default': default, 'occ': occ, 'fixes': fixes,
            'fix_by_ctx': by_ctx, 'mixed': mixed}


def load_judgements(judged_dir: Path, editions: Optional[list] = None) -> dict:
    """-> {(edition, kind, surface): canon row}. Duplicated keys are an error."""
    out: dict = {}
    for f in sorted(Path(judged_dir).glob('judged_*.jsonl')):
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                c = canon_row(json.loads(line))
                if editions and c['edition'] not in editions:
                    continue
                key = (c['edition'], c['kind'], c['surface'])
                if key in out:
                    raise ValueError(f'duplicate judgement {key}')
                out[key] = c
    return out


# ---------------------------------------------------------------------------------------------------------------------
# occurrences from the predictions (same procedure as ner2_cand.build)
# ---------------------------------------------------------------------------------------------------------------------
def _compact(s: str) -> str:
    return re.sub(r'\s+', '', norm_surface(s))


def occurrences(recs: dict, preds: dict) -> tuple[dict, dict]:
    """-> (cands, gold_nf). ``cands[(edition, surface)]`` = list of occurrences ``{'id','s','e','cs','ce','systems','text'}`` (cs/ce: extent of the cluster);
    ``gold_nf[(edition, surface)]`` = editor spans that no system touches ``{'id','s','e'}``."""
    names = sorted(preds)
    cands: dict = defaultdict(list)
    gold_nf: dict = defaultdict(list)
    for rid, rec in recs.items():
        ed = rec['edition']
        gold = tags_to_spans(rec['ner_tags_loc'])
        items = []
        for n in names:
            p = preds[n].get(rid)
            if p is None:
                continue
            for sp in p.get('LOC', ()):
                items.append((sp['s'], sp['e'], n, sp.get('conf', 1.0)))
        for gs, ge in gold:
            if not any(it[0] < ge and gs < it[1] for it in items):
                gold_nf[(ed, norm_surface(span_text(rec, gs, ge)))].append({'id': rid, 's': gs, 'e': ge})
        for cs, ce, its in clusters(items):
            if any(gs < ce and cs < ge for gs, ge in gold):
                continue
            cnt = Counter((s, e) for s, e, _, _ in its)
            conf = defaultdict(float)
            for s, e, _, c in its:
                conf[(s, e)] += c
            s, e = min(cnt, key=lambda x: (-cnt[x], -conf[x], x[1] - x[0], x[0]))
            key = norm_surface(span_text(rec, s, e))
            cands[(ed, key)].append({'id': rid, 's': s, 'e': e, 'cs': cs, 'ce': ce, 'systems': sorted({n for _, _, n, _ in its}),
                                     'text': span_text(rec, s, e)})
    return dict(cands), dict(gold_nf)


def _find_fix(rec: dict, text: str, cs: int, ce: int, window: int = 8) -> Optional[tuple[int, int]]:
    """Token span whose text equals ``text`` (blanks and case ignored) in the passage, closest to the cluster [cs, ce)."""
    want = _compact(text)
    if not want:
        return None
    toks = rec['tokens']
    n = len(toks)
    best = None
    for a in range(max(0, cs - window), min(n, ce + window)):
        if not any(ch.isalnum() for ch in toks[a]):
            continue
        acc = ''
        for b in range(a, min(n, a + 12)):
            acc += toks[b]
            c = _compact(acc)
            if c == want:
                dist = max(0, a - ce, cs - (b + 1))
                if best is None or dist < best[0]:
                    best = (dist, a, b + 1)
                break
            if len(c) > len(want) or not want.startswith(c):
                break
    return (best[1], best[2]) if best else None


def iter_decisions(cands: dict, judged: dict):
    """Final decision of every candidate occurrence: yields ``(edition, surface, occurrence, decision, fixes, explicit)``; ``explicit`` is True when the
    judge recorded a decision for this very occurrence, False when the group decision (or the 'mixed' default) was applied to it. Groups without
    judgement (empty surface form) yield decision None."""
    for key, occs in cands.items():
        ed, surf = key
        j = judged.get((ed, 'pred_not_gold', surf))
        if j is None:
            for o in occs:
                yield ed, surf, o, None, [], False
            continue
        by_exact = {(d['id'], d['s'], d['e']): d for d in j['occ'] if d['s'] is not None}
        by_text: dict = defaultdict(list)
        for d in j['occ']:
            if d['s'] is None:
                by_text[(d['id'], _compact(d['span'] or ''))].append(d)
        used_text: Counter = Counter()
        for o in sorted(occs, key=lambda x: (x['id'], x['s'])):
            d = by_exact.get((o['id'], o['s'], o['e']))
            if d is None:
                tk = (o['id'], _compact(o['text']))
                lst = by_text.get(tk)
                if lst and used_text[tk] < len(lst):
                    d = lst[used_text[tk]]
                    used_text[tk] += 1
            fixes = list(d.get('fix', j['fixes'])) if d is not None else list(j['fixes'])
            fixes += j['fix_by_ctx'].get((o['id'], o['s']), []) + j['fix_by_ctx'].get((o['id'], None), [])
            yield ed, surf, o, (d['decision'] if d is not None else j['default']), fixes, d is not None


def decide(recs: dict, cands: dict, gold_nf: dict, judged: dict) -> tuple[dict, dict]:
    """-> (per-passage changes, report). changes[id] = {'add': [..], 'mask': [..], 'remove': [..]} with spans as dicts (s, e, + provenance)."""
    ch: dict = defaultdict(lambda: {'add': [], 'mask': [], 'remove': []})
    rep: dict = defaultdict(Counter)
    for key, occs in cands.items():
        j = judged.get((key[0], 'pred_not_gold', key[1]))
        if j is None:
            rep[key[0]]['groups_unjudged'] += 1
            rep[key[0]]['occurrences_unjudged'] += len(occs)
        else:
            rep[key[0]]['groups'] += 1
            if j['n'] is not None and j['n'] != len(occs):
                rep[key[0]]['groups_n_mismatch'] += 1
    for ed, surf, o, dec, fixes, explicit in iter_decisions(cands, judged):
        if dec is None:
            continue
        rep[ed]['occ_judged' if explicit else 'occ_default'] += 1
        rep[ed][f'occ_{dec}'] += 1
        rec = recs[o['id']]
        src = {'edition': ed, 'surface': surf, 'systems': o['systems']}
        if fixes and dec == 'error':
            got = 0
            for t in fixes:
                sp = _find_fix(rec, t, o['cs'], o['ce'])
                if sp:
                    ch[o['id']]['add'].append({'s': sp[0], 'e': sp[1], 'fix_of': [o['s'], o['e']], **src})
                    got += 1
            rep[ed]['fix_located' if got else 'fix_not_located'] += 1
        elif dec == 'add':
            ch[o['id']]['add'].append({'s': o['s'], 'e': o['e'], **src})
        elif dec == 'ambiguous':
            ch[o['id']]['mask'].append({'s': o['cs'], 'e': o['ce'], 'what': 'candidate', 'cand': [o['s'], o['e']], **src})
    for key, occs in gold_nf.items():
        ed, surf = key
        j = judged.get((ed, 'gold_not_found', surf))
        if j is None:
            rep[ed]['gold_nf_unjudged'] += len(occs)
            continue
        dec = j['default']
        for o in occs:
            rep[ed][f'gold_nf_{dec}'] += 1
            if dec == 'remove':
                ch[o['id']]['remove'].append({'s': o['s'], 'e': o['e'], 'edition': ed, 'surface': surf})
            elif dec == 'ambiguous':
                ch[o['id']]['mask'].append({'s': o['s'], 'e': o['e'], 'what': 'editor_span', 'edition': ed, 'surface': surf})
    return dict(ch), {k: dict(v) for k, v in rep.items()}


# ---------------------------------------------------------------------------------------------------------------------
# apply to a record
# ---------------------------------------------------------------------------------------------------------------------
def apply_changes(editor: list, change: Optional[dict], n_tokens: int) -> dict:
    """editor: LOC spans of the editors. -> {'gold': spans, 'mask': sorted token indices, 'added': [...], 'removed': [...]}.
    An added span that touches an editor span or another added span is dropped (kept in ``skipped``)."""
    change = change or {'add': [], 'mask': [], 'remove': []}
    rem = {(x['s'], x['e']) for x in change['remove']}
    gold = [sp for sp in editor if sp not in rem]
    added, skipped = [], []
    taken = list(gold)
    for a in sorted(change['add'], key=lambda x: (x['s'], x['e'])):
        sp = (a['s'], a['e'])
        if any(sp[0] < e and s < sp[1] for s, e in taken):
            skipped.append(a)
            continue
        taken.append(sp)
        added.append(a)
    mask = set()
    for m in change['mask']:
        mask.update(range(m['s'], min(m['e'], n_tokens)))
    gold_all = sorted(set(gold) | {(a['s'], a['e']) for a in added})
    return {'gold': gold_all, 'mask': sorted(mask), 'added': added, 'removed': sorted(rem), 'skipped': skipped, 'mask_items': change['mask']}


def drop_masked(spans, mask: set) -> list:
    return [(s, e) for s, e in spans if not any(t in mask for t in range(s, e))]


def build(data_loc: Path, full: Path, pred_dir: Path, judged_dir: Path, out: Path, pool_out: Optional[Path] = None) -> dict:
    recs = {}
    with open(data_loc, encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            recs[d['id']] = d
    files = {p.name[:-len('.jsonl.gz')]: p for p in sorted(Path(pred_dir).glob('*.jsonl.gz'))}
    preds = {k: read_preds(v) for k, v in files.items()}
    cands, gold_nf = occurrences(recs, preds)
    judged = load_judgements(judged_dir)
    changes, rep = decide(recs, cands, gold_nf, judged)
    # groups judged but not rebuilt
    seen = {(ed, s) for (ed, s) in cands}
    missing = [k for k in judged if k[1] == 'pred_not_gold' and (k[0], k[2]) not in seen]
    stats: dict = defaultdict(Counter)
    n_out = 0
    tmp = out.with_suffix(out.suffix + '.tmp')
    with open(full, encoding='utf-8') as fin, open(tmp, 'w', encoding='utf-8') as fout:
        for line in fin:
            r = json.loads(line)
            ed = r['edition']
            st = stats[ed]
            st['passages'] += 1
            editor = tags_to_spans(r['ner_tags_loc'])
            scored = bool(r['loc_valid']) and r['id'] in recs
            st['passages_loc_scored'] += scored
            r['ner_tags_loc_editor'] = r['ner_tags_loc']
            if scored:
                res = apply_changes(editor, changes.get(r['id']), len(r['tokens']))
            else:
                res = {'gold': editor, 'mask': [], 'added': [], 'removed': [], 'skipped': [], 'mask_items': []}
            r['ner_tags_loc'] = spans_to_tags(len(r['tokens']), res['gold'])
            r['loc_added'] = [{'s': a['s'], 'e': a['e'], 'text': span_text(r, a['s'], a['e']), 'group': a['surface'], 'n_systems': len(a['systems']),
                               **({'fix_of': a['fix_of']} if 'fix_of' in a else {})} for a in res['added']]
            r['loc_removed'] = [{'s': s, 'e': e, 'text': span_text(r, s, e)} for s, e in res['removed']]
            r['loc_mask'] = res['mask']
            r['loc_ambiguous'] = [{'s': m['s'], 'e': m['e'], 'what': m['what'], **({'span': m['cand']} if 'cand' in m else {}), 'group': m['surface']}
                                  for m in res['mask_items']]
            r['loc_gold_v2'] = [[s, e] for s, e in res['gold']]
            r['loc_gold_v2_source'] = JUDGE if (res['added'] or res['removed'] or res['mask']) else 'editors'
            r.pop('ner_tags', None)
            st['loc_editor'] += len(editor)
            st['loc_added'] += len(res['added'])
            st['loc_removed'] += len(res['removed'])
            st['loc_gold_v2'] += len(res['gold'])
            st['mask_tokens'] += len(res['mask'])
            st['added_skipped'] += len(res['skipped'])
            if scored:
                st['loc_editor_scored'] += len(editor)
                st['loc_gold_v2_scored'] += len(res['gold'])
            fout.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
            n_out += 1
    tmp.replace(out)
    if pool_out:
        with gzip.open(pool_out, 'wt', encoding='utf-8') as fh:
            for rid, c in changes.items():
                fh.write(json.dumps({'id': rid, **c}, ensure_ascii=False, separators=(',', ':')) + '\n')
    return {'written': n_out, 'groups_judged_not_rebuilt': len(missing), 'missing_examples': [list(k) for k in missing[:10]],
            'per_edition': {ed: dict(c) for ed, c in stats.items()}, 'decisions': rep,
            'systems': sorted(files), 'families': dict(Counter(family(n) for n in files))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas.ner.ner2_testgold')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--data', type=Path, required=True, help='test_v2_loc.jsonl')
    b.add_argument('--full', type=Path, required=True, help='test_v2.jsonl (all passages)')
    b.add_argument('--pred-dir', type=Path, required=True)
    b.add_argument('--judged-dir', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--pool-out', type=Path, default=None)
    b.add_argument('--report', type=Path, default=None)
    a = ap.parse_args(argv)
    res = build(a.data, a.full, a.pred_dir, a.judged_dir, a.out, a.pool_out)
    if a.report:
        a.report.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'decisions'}, indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
