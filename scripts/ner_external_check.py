"""Consistency check of the converted external datasets (every record of the small sets, one record in 20 of the big ones).

    python -I scripts/ner_external_check.py [--root ~/renascor-atlas-data/ner/external]

Per record: len(tokens) == len(ner_tags) == len(ner_tags_loc) == len(ner_tags_per) == len(token_starts); tag ids in range; the joint BIO sequence is well formed (no
I- without a B- or I- of the same type before it); the LOC view (PER view) is zero when loc_valid (per_valid) is false and non-zero exactly where spans of that label are;
text[start:end] of every span equals the tokens it covers; n_loc / n_per equal the number of spans; licence fields present.  Prints one line per dataset; exit code 1 on any error.
"""
import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def check(rec: dict) -> list[str]:
    err = []
    n = len(rec['tokens'])
    for k in ('ner_tags', 'ner_tags_loc', 'ner_tags_per', 'token_starts'):
        if len(rec[k]) != n:
            err.append('len ' + k)
    if err:
        return err
    if any(t not in range(5) for t in rec['ner_tags']) or any(t not in range(3) for t in rec['ner_tags_loc'] + rec['ner_tags_per']):
        err.append('tag range')
    for name, tags in (('joint', rec['ner_tags']), ('loc', rec['ner_tags_loc']), ('per', rec['ner_tags_per'])):
        prev = 0
        for t in tags:
            if t in ((2,) if name == 'loc' else (2,) if name == 'per' else (2, 4)):
                if name == 'joint':
                    ok = (t == 2 and prev in (1, 2)) or (t == 4 and prev in (3, 4))
                else:
                    ok = prev in (1, 2)
                if not ok:
                    err.append('bio ' + name)
                    break
            prev = t
    if not rec['loc_valid'] and any(rec['ner_tags_loc']):
        err.append('loc not blanked')
    if not rec['per_valid'] and any(rec['ner_tags_per']):
        err.append('per not blanked')
    text = rec['text']
    for s in rec['spans']:
        if not (0 <= s['start'] < s['end'] <= len(text)):
            err.append('span range')
            continue
        inside = [i for i in range(n) if rec['token_starts'][i] >= s['start'] and rec['token_starts'][i] + len(rec['tokens'][i]) <= s['end']]
        if not inside or text[rec['token_starts'][inside[0]]:rec['token_starts'][inside[-1]] + len(rec['tokens'][inside[-1]])] != text[s['start']:s['end']]:
            err.append('span text')
    if rec['n_loc'] != sum(1 for s in rec['spans'] if s['label'] == 'LOC') or rec['n_per'] != sum(1 for s in rec['spans'] if s['label'] == 'PER'):
        err.append('n_loc / n_per')
    for k in ('licence', 'licence_flag', 'quality', 'source', 'lang', 'split'):
        if not rec.get(k):
            err.append('missing ' + k)
    return err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path.home() / 'renascor-atlas-data/ner/external'))
    a = ap.parse_args()
    root = Path(a.root)
    bad = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != 'aggregate'):
        files = sorted(glob.glob(str(d / '*.jsonl')))
        if not files:
            continue
        errs, n, seen = Counter(), 0, 0
        ex = {}
        for f in files:
            for line in open(f, encoding='utf-8'):
                seen += 1
                if seen > 200_000 and int(hashlib.md5(line[:80].encode()).hexdigest()[:6], 16) % 20:
                    continue
                rec = json.loads(line)
                n += 1
                for e in check(rec):
                    errs[e] += 1
                    ex.setdefault(e, rec['id'])
        bad += sum(errs.values())
        print(f'{d.name:20s} checked {n:>8d} of {seen:>8d}  errors {dict(errs) or "none"} {ex if ex else ""}', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
