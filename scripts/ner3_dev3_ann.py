"""dev3 annotation tools.

view   : print passages for annotation (text only, blind: no model output, no other pass)
parse  : turn a hand-written annotation file into spans (token and character offsets) and validate it against the passages
compare: agreement between two passes, list of disagreements for adjudication
merge  : build gold_dev3.jsonl from pass A, pass B and the adjudication decisions

Annotation file (UTF-8, one block per passage):

    @dev3/es/xxx                         passage id
    L: Roma | <río> Guadalquivir | Francia#2 | Tiro?
    P: <Don> Quijote | Sancho Panza

L = place (LOC), P = person (PER); '-' = none. Items are separated by ' | ' and name a surface string (tokens as in the passage).
  - the item tags EVERY occurrence of that token sequence in the passage, unless '#n' or '#n,m' selects occurrences (1-based, in text order)
  - '<...>' before the name: optional leading tokens (a generic word such as 'río', an honorific such as 'Don'); the core span is the name alone,
    the extended span (alt) includes them; either segmentation is accepted by the score (as for Mildenhall / Mildenhall River in gold v2)
  - trailing '?' : the annotator doubts (certainty 'doubt'); otherwise 'sure'
Lines starting with '#!' are comments.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.ner2_passages import fold, tokenize  # noqa: E402


def norm(t: str) -> str:
    return t.replace('ſ', 's')


def load_passages(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def read_ann(path):
    """-> {id: {'LOC': [item,...], 'PER': [item,...]}}; raises on a passage that has no L: and P: line."""
    out = {}
    cur = None
    for ln, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#!'):
            continue
        if line.startswith('@'):
            cur = unicodedata.normalize('NFC', line[1:].strip())
            if cur in out:
                raise ValueError(f'{path}:{ln} duplicate id {cur}')
            out[cur] = {'LOC': None, 'PER': None}
            continue
        m = re.match(r'^([LP]):\s*(.*)$', line)
        if not m or cur is None:
            raise ValueError(f'{path}:{ln} cannot read: {line[:60]}')
        lab = 'LOC' if m.group(1) == 'L' else 'PER'
        body = m.group(2).strip()
        items = [] if body in ('-', '') else [x.strip() for x in body.split(' | ') if x.strip()]
        out[cur][lab] = items
    for k, v in out.items():
        if v['LOC'] is None or v['PER'] is None:
            raise ValueError(f'{path}: passage {k} lacks an L: or P: line')
    return out


ITEM = re.compile(r'^(?:<(?P<opt>[^>]+)>\s*)?(?P<core>.+?)(?:#(?P<occ>[\d,]+))?(?P<doubt>\?)?$')


def fold_keep_case(t: str) -> str:
    t = unicodedata.normalize('NFD', t.replace('ſ', 's'))
    return ''.join(c for c in t if unicodedata.category(c) != 'Mn')


def groups_of(tokens, starts):
    """Merge tokens made only of combining marks (decomposed macrons, tildes ...) into the token they follow. -> [(first, last_exclusive, folded)]"""
    out = []
    for i, t in enumerate(tokens):
        if out and t and all(unicodedata.category(c) == 'Mn' for c in t) and starts[i] == starts[out[-1][1] - 1] + len(tokens[out[-1][1] - 1]):
            f, _, g = out[-1]
            out[-1] = (f, i + 1, g)
        else:
            out.append((i, i + 1, fold_keep_case(t)))
    return out


def find_seq(toks, seq):
    n = len(seq)
    return [i for i in range(len(toks) - n + 1) if toks[i:i + n] == seq]


def spans_of(passage, items, label):
    grp = groups_of(passage['tokens'], passage['token_starts'])
    gtok = [g[2] for g in grp]
    out, errs = [], []
    for it in items:
        m = ITEM.match(it)
        if not m:
            errs.append(f'unreadable item {it!r}')
            continue
        core = m.group('core').strip()
        seq = [fold_keep_case(t) for t in tokenize(core)[0]]
        seq = [t for t in seq if t]
        hits = find_seq(gtok, seq)
        if not hits:   # case-insensitive fallback
            lt = [g.lower() for g in gtok]
            ls = [g.lower() for g in seq]
            hits = find_seq(lt, ls)
            if hits:
                errs.append(f'warn: {it!r} matched only ignoring case')
        if not hits:
            errs.append(f'not found: {it!r}')
            continue
        if m.group('occ'):
            want = [int(x) for x in m.group('occ').split(',')]
            bad = [w for w in want if w < 1 or w > len(hits)]
            if bad:
                errs.append(f'occurrence {bad} of {it!r}: only {len(hits)} found')
            hits = [hits[w - 1] for w in want if 1 <= w <= len(hits)]
        opt = [t for t in (fold_keep_case(x) for x in tokenize(m.group('opt'))[0]) if t] if m.group('opt') else []
        for h in hits:
            sp = {'label': label, 'start': grp[h][0], 'end': grp[h + len(seq) - 1][1], 'certainty': 'doubt' if m.group('doubt') else 'sure'}
            if opt and h - len(opt) >= 0 and gtok[h - len(opt):h] == opt:
                sp['alt_start'] = grp[h - len(opt)][0]
            elif opt:
                errs.append(f'warn: optional prefix {opt} not before occurrence at token {grp[h][0]} of {it!r}')
            out.append(sp)
    return out, errs


def finish(passage, sp):
    toks, st = passage['tokens'], passage['token_starts']
    sp['text'] = ' '.join(toks[sp['start']:sp['end']])
    sp['surface'] = passage['text'][st[sp['start']]: st[sp['end'] - 1] + len(toks[sp['end'] - 1])]
    sp['char_start'] = st[sp['start']]
    sp['char_end'] = st[sp['end'] - 1] + len(toks[sp['end'] - 1])
    if 'alt_start' in sp:
        sp['alt_char_start'] = st[sp['alt_start']]
    return sp


def parse(passages, ann):
    byid = {unicodedata.normalize('NFC', p['id']): p for p in passages}
    res, report = {}, []
    for pid, a in ann.items():
        if pid not in byid:
            report.append((pid, 'unknown passage id'))
            continue
        p = byid[pid]
        spans = []
        for lab in ('LOC', 'PER'):
            sps, errs = spans_of(p, a[lab], lab)
            spans += sps
            report += [(pid, e) for e in errs]
        # exact duplicates (two items that tag the same occurrence)
        seen = {}
        uniq = []
        for s in sorted(spans, key=lambda s: (s['start'], s['end'], s['label'])):
            k = (s['label'], s['start'], s['end'])
            if k in seen:
                report.append((pid, f'warn: duplicate span {s["label"]} at {s["start"]}-{s["end"]}'))
                continue
            seen[k] = 1
            uniq.append(s)
        # overlaps
        for i, s in enumerate(uniq):
            for t in uniq[i + 1:]:
                if t['start'] < s['end'] and s['start'] < t['end']:
                    report.append((pid, f'overlap {s["label"]} {s["start"]}-{s["end"]} / {t["label"]} {t["start"]}-{t["end"]} ({" ".join(p["tokens"][s["start"]:s["end"]])} / {" ".join(p["tokens"][t["start"]:t["end"]])})'))
        res[p['id']] = [finish(p, s) for s in uniq]
    missing = [p['id'] for p in passages if unicodedata.normalize('NFC', p['id']) not in ann]
    return res, report, missing


def same(a, b):
    """Two spans are the same entity mention when label and boundaries match; boundaries match if the cores are equal, or one core equals the other's extended span."""
    if a['label'] != b['label']:
        return False
    ra = {(a['start'], a['end'])} | ({(a['alt_start'], a['end'])} if 'alt_start' in a else set())
    rb = {(b['start'], b['end'])} | ({(b['alt_start'], b['end'])} if 'alt_start' in b else set())
    return bool(ra & rb)


def compare(spa, spb):
    """-> matched pairs, only_a, only_b (greedy one-to-one on `same`)."""
    used = set()
    pairs, only_a = [], []
    for a in spa:
        j = next((j for j, b in enumerate(spb) if j not in used and same(a, b)), None)
        if j is None:
            only_a.append(a)
        else:
            used.add(j)
            pairs.append((a, spb[j]))
    only_b = [b for j, b in enumerate(spb) if j not in used]
    return pairs, only_a, only_b


def prf(tp, fa, fb):
    p = tp / (tp + fa) if tp + fa else 1.0
    r = tp / (tp + fb) if tp + fb else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- compare / adjudicate / merge

def conflicts_of(pa, pb, spa, spb):
    """Pair the spans of two passes. Returns (agreed [(a,b)], conflicts [{'A': [...], 'B': [...]}]) ; a conflict groups overlapping unmatched spans (boundary or label
    differences) or is a single span seen by one pass only."""
    pairs, oa, ob = compare(spa, spb)
    items = [('A', s) for s in oa] + [('B', s) for s in ob]
    items.sort(key=lambda t: (t[1]['start'], t[1]['end']))
    groups = []
    for side, sp in items:
        lo, hi = sp.get('alt_start', sp['start']), sp['end']
        if groups and lo < groups[-1]['hi'] and sp['start'] < groups[-1]['hi'] + 0 and lo < groups[-1]['hi']:
            g = groups[-1]
            g[side].append(sp)
            g['hi'] = max(g['hi'], hi)
        else:
            groups.append({'A': [], 'B': [], 'hi': hi})
            groups[-1][side].append(sp)
    return pairs, groups


def cmd_compare(a):
    ps = {p['id']: p for p in load_passages(a.passages)}
    A = json.load(open(a.a))
    B = json.load(open(a.b))
    ids = [i for i in ps if i in A and i in B]
    stat = defaultdict(Counter)
    tokstat = defaultdict(Counter)
    lines = []
    summary = {}
    for pid in ids:
        p = ps[pid]
        pairs, groups = conflicts_of(p, p, A[pid], B[pid])
        lang = p['lang']
        for lab in ('LOC', 'PER'):
            na = sum(1 for s in A[pid] if s['label'] == lab)
            nb = sum(1 for s in B[pid] if s['label'] == lab)
            tp = sum(1 for x, y in pairs if x['label'] == lab)
            for key in (lang, 'all'):
                stat[(key, lab)]['A'] += na
                stat[(key, lab)]['B'] += nb
                stat[(key, lab)]['match'] += tp
            # token level (core tokens), for kappa
            n = len(p['tokens'])
            ta = [0] * n
            tb = [0] * n
            for s in A[pid]:
                if s['label'] == lab:
                    for i in range(s['start'], s['end']):
                        ta[i] = 1
            for s in B[pid]:
                if s['label'] == lab:
                    for i in range(s['start'], s['end']):
                        tb[i] = 1
            for key in (lang, 'all'):
                c = tokstat[(key, lab)]
                for x, y in zip(ta, tb):
                    c[(x, y)] += 1
        cert = Counter()
        for x, y in pairs:
            cert[(x['certainty'], y['certainty'])] += 1
        for key in (lang, 'all'):
            stat[(key, 'cert')].update({f'{k[0]}/{k[1]}': v for k, v in cert.items()})
        if groups:
            lines.append(f'@{pid}   [{p["year"]}] {len(groups)} conflict(s)')
            lines.append(p['text'])
            for gi, g in enumerate(groups, 1):
                parts = []
                for side in ('A', 'B'):
                    for k, s in enumerate(g[side], 1):
                        alt = f' (+{" ".join(p["tokens"][s["alt_start"]:s["start"]])})' if 'alt_start' in s else ''
                        ctx = ' '.join(p['tokens'][max(0, s['start'] - 4):s['start']]) + ' [[' + ' '.join(p['tokens'][s['start']:s['end']]) + ']] ' + ' '.join(p['tokens'][s['end']:s['end'] + 3])
                        parts.append(f'    {side}{k} {s["label"]} {s["certainty"]}: {" ".join(p["tokens"][s["start"]:s["end"]])}{alt}   ...{ctx}')
                if not g['A']:
                    parts.insert(0, '    A: -')
                if not g['B']:
                    parts.append('    B: -')
                lines.append(f'  c{gi}:')
                lines += parts
            lines.append('')
    def kappa(c):
        n = sum(c.values())
        po = (c[(0, 0)] + c[(1, 1)]) / n if n else 1
        pa = (c[(1, 0)] + c[(1, 1)]) / n if n else 0
        pb = (c[(0, 1)] + c[(1, 1)]) / n if n else 0
        pe = pa * pb + (1 - pa) * (1 - pb)
        return (po - pe) / (1 - pe) if pe < 1 else 1.0
    res = {}
    for (key, lab), c in sorted(stat.items(), key=lambda kv: (kv[0][0] != 'all', kv[0])):
        if lab == 'cert':
            res[f'{key}/certainty (A/B) of matched spans'] = dict(c)
            continue
        p_, r_, f_ = prf(c['match'], c['A'] - c['match'], c['B'] - c['match'])
        tk = tokstat[(key, lab)]
        res[f'{key}/{lab}'] = {'spans_A': c['A'], 'spans_B': c['B'], 'matched': c['match'], 'precision_B_vs_A': round(p_, 4), 'recall_B_vs_A': round(r_, 4), 'F1': round(f_, 4),
                                'token_kappa': round(kappa(tk), 4)}
    Path(a.out_json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Path(a.out_todo).write_text('\n'.join(lines))
    print(json.dumps(res, indent=1, ensure_ascii=False))


def read_decisions(path):
    """@id then lines 'cN: A1' | 'cN: B2?' | 'cN: A1 B1' (both kept, e.g. two different spans) | 'cN: -' (nothing kept). '?' after a code = the adjudicated span is kept with certainty doubt."""
    out, cur = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#!'):
            continue
        if line.startswith('@'):
            cur = line[1:].split()[0]
            out[cur] = {}
            continue
        m = re.match(r'^c(\d+):\s*(.*)$', line)
        if not m or cur is None:
            raise ValueError(f'cannot read decision: {line}')
        out[cur][int(m.group(1))] = m.group(2).split()
    return out


def cmd_merge(a):
    ps = load_passages(a.passages)
    A = json.load(open(a.a))
    B = json.load(open(a.b))
    dec = read_decisions(a.decisions) if a.decisions else {}
    out, problems = [], []
    nstat = Counter()
    for p in ps:
        pid = p['id']
        pairs, groups = conflicts_of(p, p, A[pid], B[pid])
        spans = []
        for x, y in pairs:
            s = {k: v for k, v in x.items()}
            s['certainty'] = 'doubt' if 'doubt' in (x['certainty'], y['certainty']) else 'sure'
            if 'alt_start' not in s and 'alt_start' in y:
                s['alt_start'] = y['alt_start']
                s['alt_char_start'] = y['alt_char_start']
            s['agreement'] = 'both'
            spans.append(s)
        d = dec.get(pid, {})
        for gi, g in enumerate(groups, 1):
            if gi not in d:
                problems.append(f'{pid} c{gi}: no decision')
                continue
            for code in d[gi]:
                if code == '-':
                    continue
                m = re.match(r'^([AB])(\d+)(\?)?$', code)
                if not m:
                    problems.append(f'{pid} c{gi}: bad code {code}')
                    continue
                side, k = m.group(1), int(m.group(2))
                if k > len(g[side]):
                    problems.append(f'{pid} c{gi}: {code} does not exist')
                    continue
                s = dict(g[side][k - 1])
                s['certainty'] = 'doubt' if m.group(3) else 'sure'
                s['agreement'] = 'adjudicated_' + side
                spans.append(s)
        spans.sort(key=lambda s: (s['start'], s['end']))
        for s in spans:
            s['pass_a'] = s['agreement'] in ('both', 'adjudicated_A')
            s['pass_b'] = s['agreement'] in ('both', 'adjudicated_B')
            nstat[(p['lang'], s['label'], s['agreement'])] += 1
        r = {k: p[k] for k in p if k not in ('spans',)}
        r['span_convention'] = 'start inclusive, end exclusive, token indices into tokens; char offsets from token_starts; alt_start/alt_char_start = extended span (optional leading generic word or honorific), either segmentation accepted'
        r['spans'] = spans
        n = len(p['tokens'])
        for lab in ('LOC', 'PER'):
            tags = ['O'] * n
            for s in spans:
                if s['label'] == lab:
                    tags[s['start']] = 'B-' + lab
                    for i in range(s['start'] + 1, s['end']):
                        tags[i] = 'I-' + lab
            r['ner_tags_' + lab.lower()] = tags
        r['annotation'] = 'agent-annotated (two passes by the same agent, adjudicated); not reviewed by the owner'
        out.append(r)
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for pr in problems:
        print('PROBLEM', pr)
    print({f'{k[0]}/{k[1]}/{k[2]}': v for k, v in sorted(nstat.items())})
    print(len(out), 'passages written;', len(problems), 'problems')


def cmd_audit(a):
    """Recall audit: capitalised tokens that no pass tagged (and whose lower-case form is not a frequent common word of the set)."""
    ps = load_passages(a.passages)
    A = json.load(open(a.a))
    B = json.load(open(a.b))
    lowercount = Counter()
    for p in ps:
        for t in p['tokens']:
            if t[:1].islower():
                lowercount[t] += 1
    for p in ps:
        if a.lang and p['lang'] != a.lang:
            continue
        cov = set()
        for sp in A.get(p['id'], []) + B.get(p['id'], []):
            cov.update(range(sp.get('alt_start', sp['start']), sp['end']))
        toks = p['tokens']
        out = []
        for i, t in enumerate(toks):
            if i in cov or len(t) < 2 or not t[0].isupper() or any(unicodedata.category(c) == 'Mn' for c in t[:1]):
                continue
            initial = i == 0 or toks[i - 1] in ('.', '!', '?', ':', ';', '¶', '(', '[', '«', '"', '“', '—', '–', '-')
            if lowercount[t.lower()] >= (1 if initial else 2):
                continue
            if t.isupper() and len(t) >= 4 and a.skip_caps:
                continue
            ctx = ' '.join(toks[max(0, i - 1):i + 2])
            out.append(f'{t}  ({ctx})')
        print(f'@{p["id"]}: ' + ' ; '.join(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('view')
    v.add_argument('passages')
    v.add_argument('--lang')
    v.add_argument('--start', type=int, default=0)
    v.add_argument('--n', type=int, default=10)
    v.add_argument('--reverse', action='store_true')
    v.add_argument('--ids-only', action='store_true')
    c = sub.add_parser('compare')
    c.add_argument('passages')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--out-json', required=True)
    c.add_argument('--out-todo', required=True)
    au = sub.add_parser('audit')
    au.add_argument('passages')
    au.add_argument('a')
    au.add_argument('b')
    au.add_argument('--lang')
    au.add_argument('--skip-caps', action='store_true')
    m = sub.add_parser('merge')
    m.add_argument('passages')
    m.add_argument('a')
    m.add_argument('b')
    m.add_argument('--decisions')
    m.add_argument('--out', required=True)
    pa = sub.add_parser('parse')
    pa.add_argument('passages')
    pa.add_argument('ann')
    pa.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'view':
        ps = load_passages(a.passages)
        if a.lang:
            ps = [p for p in ps if p['lang'] == a.lang]
        if a.reverse:
            ps = ps[::-1]
        for p in ps[a.start:a.start + a.n]:
            print(f'@{p["id"]}   [{p["year"]}]')
            if not a.ids_only:
                print(p['text'])
            print()
    elif a.cmd == 'compare':
        cmd_compare(a)
    elif a.cmd == 'merge':
        cmd_merge(a)
    elif a.cmd == 'audit':
        cmd_audit(a)
    elif a.cmd == 'parse':
        ps = load_passages(a.passages)
        ann = read_ann(a.ann)
        res, report, missing = parse(ps, ann)
        for pid, msg in report:
            print(pid, msg)
        print(f'{len(res)} passages parsed, {sum(len(v) for v in res.values())} spans, {len(missing)} passages without annotation in this file', file=sys.stderr)
        if a.out:
            json.dump(res, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
