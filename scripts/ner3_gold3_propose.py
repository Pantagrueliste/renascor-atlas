"""Gold v3 proposals for the owner's review: LOC and PER spans from Gemma-4-31B (+pp, as in the silver) and the four R3 encoders.

    python scripts/ner3_gold3_propose.py --passages gold3/passages.jsonl --llm gold3/proposals/llm_gemma4_31b_pp.jsonl \
        --enc r3_mdeberta_HM=pred_enc/r3_mdeberta_HM.jsonl.gz ... --out gold3/proposed_gold_v3.jsonl --stats gold3/proposed_gold_v3_stats.json

--llm is the output of ner3_silver_combine.py (one teacher, +pp: final '.' dropped, repeated names propagated, overlaps resolved).
--enc NAME=FILE are outputs of ner_audit_predict.py (token spans of the passage tokens, LOC and PER heads).

Statuses, per candidate span (start, end, label; end exclusive, token indices):
  agreed    the LLM and at least 2 encoders give exactly this span and label;
  proposed  the LLM alone (0 or 1 encoder), or at least 2 encoders without the LLM;
  to_check  one encoder alone, or a conflict: overlapping candidates with other boundaries or another label. In a conflict every
            variant is to_check, except an agreed variant whose rivals each come from one system only (it stays agreed, the
            rivals are to_check, reason "variante minoritaire");
  owner_v2  gold v2 passages only: the owner's LOC decisions ("place", and the one "doubt"), fixed. On these passages the models'
            LOC outputs are ignored (never merged, never scored) and only PER is proposed; a PER candidate that overlaps an owner
            place is to_check.
Output: one passage per line (all fields of passages.jsonl) plus spans[], llm_unmatched (strings of the LLM answer not found in the
text), n_spans_by_status, review_rank. Order: by language (la, es, it, fr, de, en), then most to_check spans first, then id.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict

LANGS = ['la', 'es', 'it', 'fr', 'de', 'en']
LLM = 'llm_gemma4_31b'


def jl(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf8') as f:
        return [json.loads(l) for l in f if l.strip()]


def base_status(llm, n_enc):
    if llm and n_enc >= 2:
        return 'agreed', f'LLM + {n_enc} encodeurs'
    if llm:
        return 'proposed', 'LLM seul' if n_enc == 0 else 'LLM + 1 encodeur'
    if n_enc >= 2:
        return 'proposed', f'{n_enc} encodeurs sans le LLM'
    return 'to_check', 'un seul encodeur'


def clusters(items):
    """items: list of (s, e, key) -> list of lists of keys whose spans overlap (transitively)."""
    out, cur, end = [], [], -1
    for s, e, k in sorted(items):
        if cur and s >= end:
            out.append(cur)
            cur, end = [], -1
        cur.append(k)
        end = max(end, e)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passages', required=True)
    ap.add_argument('--llm', required=True)
    ap.add_argument('--llm-raw', default='', help='raw teacher files, comma-separated (for the unmatched strings)')
    ap.add_argument('--enc', action='append', required=True, help='NAME=FILE')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats')
    a = ap.parse_args()
    encs = []
    for x in a.enc:
        name, _, path = x.partition('=')
        encs.append((name, {r['id']: r for r in jl(path)}))
    llm = {r['id']: r for r in jl(a.llm)}
    unmatched = {}
    for path in filter(None, a.llm_raw.split(',')):
        for r in jl(path):
            if r.get('error') is None:
                unmatched[r['id']] = r.get('unmatched') or []
    passages = jl(a.passages)
    missing = {n: sum(p['id'] not in d for p in passages) for n, d in encs}
    missing[LLM] = sum(p['id'] not in llm for p in passages)
    llm_not_ok = sum(1 for p in passages if p['id'] in llm and not llm[p['id']].get('silver_ok'))

    out = []
    for p in passages:
        pid, toks, ts = p['id'], p['tokens'], p['token_starts']
        ends = [s + len(t) for s, t in zip(ts, toks)]
        g2 = bool(p.get('from_gold2'))
        labels = ('PER',) if g2 else ('LOC', 'PER')
        src = defaultdict(dict)            # (s, e, label) -> {system: conf or True}
        lr = llm.get(pid)
        if lr:
            for x in lr['spans']:
                if x['label'] in labels:
                    src[(x['start'], x['end'], x['label'])][LLM] = None
        for name, d in encs:
            r = d.get(pid)
            if not r:
                continue
            for lab in labels:
                for x in r.get(lab, []):
                    src[(x['s'], x['e'], lab)][name] = x.get('conf')
        owner = []
        if g2:
            for s in p['gold2_loc']:
                if s['status'] in ('place', 'doubt'):
                    owner.append(s)
        items = [(k[0], k[1], ('c', k)) for k in src] + [(s['start'], s['end'], ('o', i)) for i, s in enumerate(owner)]
        spans = []
        for cl in clusters(items):
            cands = [k for t, k in cl if t == 'c']
            has_owner = any(t == 'o' for t, _ in cl)
            info = {}
            for k in cands:
                sy = src[k]
                n_enc = sum(1 for n, _ in encs if n in sy)
                st, why = base_status(LLM in sy, n_enc)
                info[k] = [st, why, (LLM in sy) + n_enc]
            if has_owner:
                for k in cands:
                    info[k][0], info[k][1] = 'to_check', f"chevauche un lieu validé (gold v2) ; {info[k][1]}"
            elif len(cands) > 1:
                order = sorted(cands, key=lambda k: (-info[k][2], LLM not in src[k], -(k[1] - k[0]), k))
                top = order[0]
                kind = 'conflit de frontière' if len({k[2] for k in cands}) == 1 else (
                    "conflit d'étiquette" if len({(k[0], k[1]) for k in cands}) == 1 else "conflit de frontière et d'étiquette")
                if info[top][0] == 'agreed' and all(info[k][2] <= 1 for k in order[1:]):
                    for k in order[1:]:
                        info[k][0], info[k][1] = 'to_check', f'variante minoritaire ({info[k][1]}) face à un span accordé'
                else:
                    for k in cands:
                        info[k][0], info[k][1] = 'to_check', f'{kind} ; {info[k][1]}'
            cid = f'c{len(spans)}' if len(cl) > 1 else None
            for k in cands:
                s, e, lab = k
                sy = src[k]
                spans.append({'start': s, 'end': e, 'label': lab, 'text': ' '.join(toks[s:e]), 'surface': p['text'][ts[s]:ends[e - 1]],
                              'char_start': ts[s], 'char_end': ends[e - 1], 'status': info[k][0], 'reason': info[k][1],
                              'sources': {LLM: LLM in sy, **{n: n in sy for n, _ in encs}},
                              'confs': {n: sy[n] for n, _ in encs if n in sy and sy[n] is not None},
                              'n_systems': info[k][2], 'conflict_group': cid})
            for t, i in cl:
                if t == 'o':
                    s = owner[i]
                    spans.append({'start': s['start'], 'end': s['end'], 'label': 'LOC', 'text': s['text'], 'surface': s['surface'],
                                  'char_start': s['char_start'], 'char_end': s['char_end'], 'status': 'owner_v2',
                                  'reason': 'décision du propriétaire (gold v2), fixe', 'owner_status': s['status'],
                                  'owner_reason': s.get('reason'), 'decided_by': s.get('decided_by'), 'gold2_span_id': s['span_id'],
                                  'sources': s.get('sources'), 'n_systems': None, 'conflict_group': cid})
        spans.sort(key=lambda x: (x['start'], x['end'], x['label'], x['status']))
        for i, x in enumerate(spans):
            x['span_id'] = f'{pid}#{i}'
        rec = dict(p)
        rec['span_convention'] = 'start inclusive, end exclusive, token indices into tokens; char offsets from token_starts'
        rec['spans'] = [{'span_id': x.pop('span_id'), **x} for x in spans]
        rec['labels_proposed'] = list(labels)
        rec['llm_unmatched'] = unmatched.get(pid, [])
        rec['llm_ok'] = bool(lr and lr.get('silver_ok'))
        rec['n_spans_by_status'] = dict(Counter(x['status'] for x in spans))
        out.append(rec)

    out.sort(key=lambda r: (LANGS.index(r['lang']), -r['n_spans_by_status'].get('to_check', 0), r['id']))
    rank = Counter()
    for r in out:
        rank[r['lang']] += 1
        r['review_rank'] = rank[r['lang']]
    with open(a.out, 'w', encoding='utf8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    st = {'n_passages': len(out), 'missing': missing, 'llm_not_ok': llm_not_ok, 'by_lang': {}}
    tot = Counter()
    for lg in LANGS:
        rs = [r for r in out if r['lang'] == lg]
        c = Counter((x['status'], x['label']) for r in rs for x in r['spans'])
        tot.update(c)
        st['by_lang'][lg] = {'passages': len(rs), 'from_gold2': sum(bool(r.get('from_gold2')) for r in rs),
                             'counts': {f'{s}|{l}': n for (s, l), n in sorted(c.items())},
                             'passages_with_to_check': sum(r['n_spans_by_status'].get('to_check', 0) > 0 for r in rs),
                             'passages_without_span': sum(not r['spans'] for r in rs),
                             'llm_unmatched_strings': sum(len(r['llm_unmatched']) for r in rs)}
    st['total'] = {f'{s}|{l}': n for (s, l), n in sorted(tot.items())}
    print(json.dumps(st, ensure_ascii=False, indent=1))
    if a.stats:
        json.dump(st, open(a.stats, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
