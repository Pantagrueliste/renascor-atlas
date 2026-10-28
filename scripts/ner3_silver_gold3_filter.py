"""Withdraw from the RenNER silver training files every passage of a document kept for the human test set gold v3.

    python scripts/ner3_silver_gold3_filter.py --docs gold3/option_retrait_silver/silver_docs_a_retirer.tsv \
        --dir silver/final --report silver/final/report_gold3_filter.json

For each silver_*.jsonl of --dir (not *.pre_gold3.jsonl): the unfiltered file is kept as <name>.pre_gold3.jsonl (once: an existing
.pre_gold3 file is the source on a rerun), and <name>.jsonl is rewritten without the passages whose doc_id is in the doc_id column of --docs.
Matching is on doc_id; the lang column is only checked. Nothing else changes (same lines, same order).
--drop-ids FILE (repeatable, JSONL with an "id" field, e.g. the --details output of ner3_silver_verify.py against the gold v3 passages)
also removes those passages: as for the earlier leak check, a passage sharing even one 8-gram with a reference set is left out.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', required=True)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--drop-ids', action='append', default=[])
    a = ap.parse_args(argv)
    drop = set()
    for p in a.drop_ids:
        for line in open(p, encoding='utf8'):
            if line.strip():
                drop.add(json.loads(line)['id'])
    want = {}
    with open(a.docs, encoding='utf8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            want[row['doc_id']] = row['lang']
    d = Path(a.dir)
    rep = {'docs_listed': len(want), 'files': {}}
    found = Counter()
    lang_mismatch = Counter()
    for out in sorted(d.glob('silver_*.jsonl')):
        if out.name.endswith('.pre_gold3.jsonl'):
            continue
        pre = out.with_name(out.stem + '.pre_gold3.jsonl')
        if not pre.exists():
            os.replace(out, pre)
        c = Counter()
        by_doc = Counter()
        tmp = out.with_suffix('.jsonl.tmp')
        with open(pre, encoding='utf8') as fi, open(tmp, 'w', encoding='utf8') as fo:
            for line in fi:
                if not line.strip():
                    continue
                r = json.loads(line)
                c['read'] += 1
                did = r.get('doc_id') or f"{r.get('collection')}/{r.get('doc')}"
                if did in want:
                    c['removed'] += 1
                    by_doc[did] += 1
                    found[did] += 1
                    if want[did] != r.get('lang'):
                        lang_mismatch[did] += 1
                    continue
                if r['id'] in drop:
                    c['removed_8gram'] += 1
                    continue
                c['kept'] += 1
                fo.write(line if line.endswith('\n') else line + '\n')
        os.replace(tmp, out)
        rep['files'][out.name] = {'pre_gold3': pre.name, 'read': c['read'], 'removed': c['removed'], 'removed_8gram': c['removed_8gram'], 'kept': c['kept'],
                                  'docs_hit': len(by_doc), 'md5_pre': md5(pre), 'md5': md5(out)}
        print(out.name, rep['files'][out.name], flush=True)
    rep['docs_found'] = len(found)
    rep['docs_not_found'] = sorted(set(want) - set(found))
    rep['lang_mismatch'] = dict(lang_mismatch)
    rep['drop_ids'] = {'files': a.drop_ids, 'ids': len(drop)}
    json.dump(rep, open(a.report, 'w'), indent=1, ensure_ascii=False)
    print('docs found', len(found), 'of', len(want), '; not found', len(rep['docs_not_found']), '; lang mismatch', len(lang_mismatch))


if __name__ == '__main__':
    main()
