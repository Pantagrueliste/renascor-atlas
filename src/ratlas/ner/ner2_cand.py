"""NER v2: candidate places for the clean test set. Union of the LOC predictions of all systems, minus what the editors tagged, grouped by (edition, surface form).

Input: ``test_v2_loc.jsonl`` (repaired passages with the editors' LOC spans as BIO tags) and one prediction file per system (``scripts/ner_audit_predict.py``
and ``scripts/ner2_zs.py`` format: ``{"id", "LOC": [{"s", "e", "conf", ...}], "PER": [...]}`` with ``s``/``e`` word indices of the record's own tokens).

Occurrences. In a passage the predicted LOC spans of all systems are merged into clusters of overlapping spans (one cluster = one place mention proposed by one
or more systems). A cluster that touches an editor LOC span is not a candidate (the editor tagged that mention; a different boundary is a convention matter,
counted apart). The other clusters are candidate occurrences. The surface of an occurrence is the span proposed by most systems (ties: sum of confidences,
then the shorter one).

Groups (``kind = "pred_not_gold"``). Occurrences with the same edition and the same normalised surface: number of occurrences and passages, the systems that
found it (with their occurrence counts), the agreement of the four best v1 encoders, the v1 rule (confidence >= 0.98 and >= 2 of the 4), whether a top
encoder reads the mention as a person, and up to five contexts (+-12 tokens) with passage ids.

Editor spans that no system finds (``kind = "gold_not_found"``): editor LOC spans that no predicted span of any system touches, grouped the same way; this is
where editors may have over-tagged (generic words, persons, institutions) or where the extraction is wrong. A system that did not run on a passage (the LLM
runs on a sample) does not count as having missed it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .spans import tags_to_spans

TOP4 = ('r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_xlmr_large_H', 'r3_hmteams_HM')
RULE_CONF = 0.98
CTX = 12
MAX_CTX = 5
_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.U)


def norm_surface(s: str) -> str:
    """Key of a surface form: NFKC, lower case, long s as s, line-break hyphens and surrounding punctuation removed, white space collapsed."""
    s = unicodedata.normalize('NFKC', s).lower().replace('ſ', 's')
    s = re.sub(r'(?<=\w)[-­‐‑‒]\s+(?=\w)', '', s)
    s = _PUNCT.sub('', s)
    return re.sub(r'\s+', ' ', s).strip()


def family(name: str) -> str:
    if 'llm' in name:
        return 'llm'
    if name.startswith(('ft_', 'r3_')):
        return 'ours'
    return 'offshelf'


def read_preds(path: Path) -> dict:
    out = {}
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            out[d['id']] = d
    return out


def clusters(items: list) -> list:
    """items: (s, e, system, conf). -> list of (start, end, [items]) merged on overlap (half-open word spans)."""
    out = []
    for it in sorted(items, key=lambda x: (x[0], x[1])):
        if out and it[0] < out[-1][1]:
            out[-1][1] = max(out[-1][1], it[1])
            out[-1][2].append(it)
        else:
            out.append([it[0], it[1], [it]])
    return out


def span_text(rec: dict, s: int, e: int) -> str:
    ts, toks = rec['token_starts'], rec['tokens']
    return rec['text'][ts[s]:ts[e - 1] + len(toks[e - 1])]


def context(rec: dict, s: int, e: int, k: int = CTX) -> dict:
    ts, toks, text = rec['token_starts'], rec['tokens'], rec['text']
    a, b = max(0, s - k), min(len(toks), e + k)
    left = text[ts[a]:ts[s]].replace('\n', ' ').strip() if s > a else ''
    right = text[ts[e - 1] + len(toks[e - 1]):ts[b - 1] + len(toks[b - 1])].replace('\n', ' ').strip() if e < b else ''
    return {'id': rec['id'], 'v1_id': rec.get('v1_id', rec['id']), 'doc': rec.get('doc'), 'lang': rec.get('lang'), 's': s, 'e': e,
            'left': left, 'span': span_text(rec, s, e), 'right': right}


def pick_contexts(occ: list, rec_of, k: int = MAX_CTX) -> list:
    """Up to k contexts: best supported first, one per document before a second one from the same document."""
    ranked = sorted(occ, key=lambda o: (-o['n_sys'], o['id'], o['s']))
    chosen, seen_docs = [], set()
    for o in ranked:
        d = rec_of[o['id']].get('doc')
        if d not in seen_docs:
            seen_docs.add(d)
            chosen.append(o)
        if len(chosen) >= k:
            break
    for o in ranked:
        if len(chosen) >= k:
            break
        if o not in chosen:
            chosen.append(o)
    out = []
    for o in chosen:
        c = context(rec_of[o['id']], o['s'], o['e'])
        c['n_sys'] = o['n_sys']
        out.append(c)
    return out


def build(data_path: Path, pred_files: dict, editions: Optional[list] = None) -> tuple[dict, dict]:
    """-> ({edition: [group, ...]}, summary)."""
    recs = {}
    with open(data_path, encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            if editions and d['edition'] not in editions:
                continue
            recs[d['id']] = d
    preds = {name: read_preds(p) for name, p in pred_files.items()}
    names = sorted(preds)
    occ_by_group: dict = defaultdict(list)
    gold_by_group: dict = defaultdict(list)
    gold_total: Counter = Counter()
    summ: dict = defaultdict(Counter)
    for rid, rec in recs.items():
        ed = rec['edition']
        sm = summ[ed]
        sm['passages'] += 1
        gold = tags_to_spans(rec['ner_tags_loc'])
        sm['gold_spans'] += len(gold)
        have = [n for n in names if rid in preds[n]]
        sm['system_passages'] += len(have)
        items, per_items = [], []
        for n in have:
            for sp in preds[n][rid].get('LOC', ()):
                items.append((sp['s'], sp['e'], n, sp.get('conf', 1.0)))
            if n in TOP4:
                per_items += [(sp['s'], sp['e']) for sp in preds[n][rid].get('PER', ())]
        cl = clusters(items)
        # editor spans: found by at least one system? (any overlap)
        for gs, ge in gold:
            key = (ed, norm_surface(span_text(rec, gs, ge)))
            gold_total[key] += 1
            if not any(it[0] < ge and gs < it[1] for it in items):
                gold_by_group[key].append({'id': rid, 's': gs, 'e': ge, 'n_sys': len(have)})
                sm['gold_not_found'] += 1
            else:
                sm['gold_touched'] += 1
        for cs, ce, its in cl:
            if any(gs < ce and cs < ge for gs, ge in gold):
                sm['clusters_touching_gold'] += 1
                continue
            sm['occurrences'] += 1
            cnt = Counter((s, e) for s, e, _, _ in its)
            conf_sum = defaultdict(float)
            for s, e, _, c in its:
                conf_sum[(s, e)] += c
            s, e = min(cnt, key=lambda x: (-cnt[x], -conf_sum[x], x[1] - x[0], x[0]))
            syst = {}
            for s_, e_, n, c in its:
                syst[n] = max(syst.get(n, 0.0), c)
            top = [n for n in TOP4 if n in syst]
            occ_by_group[(ed, norm_surface(span_text(rec, s, e)))].append({
                'id': rid, 's': s, 'e': e, 'n_sys': len(syst), 'systems': syst, 'forms': Counter(span_text(rec, s_, e_) for s_, e_, _, _ in its),
                'n_top4': len(top), 'conf_top4': max([syst[n] for n in top], default=0.0),
                'per_top4': any(ps < ce and cs < pe for ps, pe in per_items) if top else False, 'lang': rec.get('lang'), 'doc': rec.get('doc')})
    out: dict = defaultdict(list)
    for (ed, key), occ in occ_by_group.items():
        if not key:
            continue
        sysc = Counter()
        forms: Counter = Counter()
        for o in occ:
            sysc.update(o['systems'].keys())
            forms.update(o['forms'])
        fam = Counter(family(n) for n in sysc)
        out[ed].append({
            'kind': 'pred_not_gold', 'edition': ed, 'surface': key, 'forms': [f for f, _ in forms.most_common(4)],
            'n': len(occ), 'n_passages': len({o['id'] for o in occ}), 'n_docs': len({o['doc'] for o in occ}),
            'n_gold_same_surface': gold_total[(ed, key)],     # the editors tagged this string this many times elsewhere in the edition
            'n_systems': len(sysc), 'n_ours': fam['ours'], 'n_offshelf': fam['offshelf'], 'n_llm': fam['llm'],
            'systems': dict(sysc.most_common()), 'n_top4_max': max(o['n_top4'] for o in occ),
            'conf_top4_max': round(max(o['conf_top4'] for o in occ), 4),
            'n_rule_v1': sum(1 for o in occ if o['n_top4'] >= 2 and o['conf_top4'] >= RULE_CONF),
            'n_per_top4': sum(1 for o in occ if o['per_top4']),
            'langs': dict(Counter(o['lang'] for o in occ).most_common(3)),
            'contexts': pick_contexts(occ, recs)})
    for (ed, key), occ in gold_by_group.items():
        if not key:
            continue
        tags = Counter()
        for o in occ:
            r = recs[o['id']]
            for sp in r['spans']:
                if sp['label'] == 'LOC' and r['text'][sp['start']:sp['end']] == span_text(r, o['s'], o['e']):
                    tags[sp['tag']] += 1
                    break
        out[ed].append({
            'kind': 'gold_not_found', 'edition': ed, 'surface': key,
            'forms': [f for f, _ in Counter(span_text(recs[o['id']], o['s'], o['e']) for o in occ).most_common(4)],
            'n': len(occ), 'n_passages': len({o['id'] for o in occ}), 'n_docs': len({recs[o['id']].get('doc') for o in occ}),
            'n_gold_same_surface': gold_total[(ed, key)], 'n_found_same_surface': gold_total[(ed, key)] - len(occ),
            'tags': dict(tags), 'n_systems_ran': max(o['n_sys'] for o in occ), 'contexts': pick_contexts(occ, recs)})
    for ed in out:
        out[ed].sort(key=lambda g: (g['kind'] != 'pred_not_gold', -g.get('n_systems', 0), -g['n'], g['surface']))
    summary = {ed: dict(c) for ed, c in summ.items()}
    for ed, groups in out.items():
        pg = [g for g in groups if g['kind'] == 'pred_not_gold']
        gg = [g for g in groups if g['kind'] == 'gold_not_found']
        summary[ed].update({
            'groups_pred_not_gold': len(pg), 'groups_gold_not_found': len(gg),
            'groups_ge2_systems': sum(1 for g in pg if g['n_systems'] >= 2), 'groups_ge5_systems': sum(1 for g in pg if g['n_systems'] >= 5),
            'groups_rule_v1': sum(1 for g in pg if g['n_rule_v1'] > 0), 'occurrences_rule_v1': sum(g['n_rule_v1'] for g in pg),
            'groups_with_per_top4': sum(1 for g in pg if g['n_per_top4'] > 0)})
    summary['_systems'] = {n: {'family': family(n), 'passages': len(preds[n])} for n in names}
    return dict(out), summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas.ner.ner2_cand')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--exclude', default='', help='comma-separated system names to leave out')
    a = ap.parse_args(argv)
    skip = {x for x in a.exclude.split(',') if x}
    files = {p.name[:-len('.jsonl.gz')]: p for p in sorted(a.pred_dir.glob('*.jsonl.gz'))}
    files = {k: v for k, v in files.items() if k not in skip}
    print(f'{len(files)} systems: {", ".join(files)}', file=sys.stderr)
    out, summ = build(a.data, files)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for ed, groups in out.items():
        with open(a.out_dir / f'cand_{ed}.jsonl', 'w', encoding='utf-8') as fh:
            for g in groups:
                fh.write(json.dumps(g, ensure_ascii=False, separators=(',', ':')) + '\n')
    (a.out_dir / 'cand_summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({ed: len(g) for ed, g in out.items()}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
