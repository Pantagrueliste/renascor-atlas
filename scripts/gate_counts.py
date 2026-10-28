"""gate_log.tsv counted: ``gate_counts_by_reason.tsv`` (kind, status, reason, n) and ``gate_counts_by_edition.tsv`` (one row per edition, one column per
rej: / sug: reason, total). Run after every final build:  .venv/bin/python -I scripts/gate_counts.py [--v2 DIR]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', type=Path, default=Path('/Users/clem/Projects/Papers/RQ/derived/atlas/v2'))
    a = ap.parse_args()
    g = pd.read_csv(a.v2 / 'gate_log.tsv', sep='\t', dtype=str, keep_default_na=False)
    g = g[g['status'].isin(['rejected', 'suggested'])]
    g.groupby(['kind', 'status', 'reason']).size().rename('n').reset_index().to_csv(a.v2 / 'gate_counts_by_reason.tsv', sep='\t', index=False)
    g = g.assign(col=g['status'].map({'rejected': 'rej:', 'suggested': 'sug:'}) + g['reason'])
    t = pd.crosstab(g['coll_slug'], g['col'])
    t['total'] = t.sum(axis=1)
    t.sort_values('total', ascending=False).reset_index().to_csv(a.v2 / 'gate_counts_by_edition.tsv', sep='\t', index=False)
    print(len(g), 'gate rows;', t['total'].head(5).to_dict())


if __name__ == '__main__':
    main()
