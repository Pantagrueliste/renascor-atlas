#!/usr/bin/env python3
"""v2.2: turn the 'wrong' verdicts of the manual review into hard overrides (``data/id_overrides.tsv``).

    python -I scripts/make_hard_overrides.py --review RQ/derived/atlas/v2/manual_review.tsv --pairs RQ/derived/atlas/v2.1/gate_pairs.parquet \
        [--write]

* every ``wrong`` verdict on an authority id (a (scheme, id) pair of an entity) is a ``reject`` (a ``suggest`` when the v2 gate had only suggested it:
  the id may be right for another entity of the cluster, but it is never used for this one);
* every ``undecidable`` verdict whose evidence says that the editor marks the identification as uncertain ("Identifizierung unsicher") is a ``suggest``
  (the editor's own doubt: kept as a suggestion, never used);
* rows whose pair is not in the v2.1 gate pairs (the key changed or the id was never extracted) are listed and skipped: they cannot match;
* rows already in ``id_overrides.tsv`` (same entity, scheme, id) are left alone.
The output keeps the existing file and appends the new rows after a comment line; running it twice adds nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas import manual_overrides as MO   # noqa: E402

UNSURE = 'Identifizierung unsicher'
ID_GROUPS_SKIPPED = {'bdpn_placed', 'cavriana_place', 'bdpn_residual'}      # verdicts on a placement / a class, not on an id


def build(review: pd.DataFrame, pairs: pd.DataFrame, existing: pd.DataFrame):
    have = set(zip(existing['coll_slug'], existing['entity_key'], existing['scheme'], existing['id']))
    st = pairs.groupby(['coll_slug', 'entity_key', 'scheme', 'id'])['status'].agg(lambda s: 'accepted' if (s == 'accepted').any() else s.iloc[0]).to_dict()
    rows, skipped = [], []
    for r in review.itertuples(index=False):
        if r.group in ID_GROUPS_SKIPPED or not r.scheme or not r.id:
            continue
        unsure = r.verdict == 'undecidable' and UNSURE in str(r.evidence)
        if r.verdict != 'wrong' and not unsure:
            continue
        k = (r.coll_slug, r.entity_key, r.scheme, r.id)
        if k in have:
            continue
        if k not in st:
            skipped.append(r.case_id)
            continue
        if unsure:
            action, why = 'suggest', 'editor marks the identification as uncertain'
        elif st[k] == 'suggested' and r.outcome == 'true_suggest':
            action, why = 'suggest', 'wrong id (verdict wrong); the v2 gate had suggested it'
        else:
            action, why = 'reject', 'wrong id (verdict wrong)'
        ev = ' '.join(str(r.evidence).split())[:160]
        rows.append(dict(coll_slug=r.coll_slug, kind=r.kind, entity_key=r.entity_key, scheme=r.scheme, id=r.id, qid='', action=action,
                         case_id=r.case_id, note=f'v2.2 hard override: {why}. {ev}'.replace('\t', ' ')))
        have.add(k)
    return pd.DataFrame(rows, columns=MO.COLS), skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--overrides', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'id_overrides.tsv')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args(argv)
    review = pd.read_csv(a.review, sep='\t', dtype=str, keep_default_na=False)
    pairs = pd.read_parquet(a.pairs, columns=['coll_slug', 'entity_key', 'scheme', 'id', 'status'])
    existing = MO.read_tsv(a.overrides)
    new, skipped = build(review, pairs, existing)
    print(f'{len(new)} new rows ({new["action"].value_counts().to_dict()}); skipped (pair not in the gate pairs): {len(skipped)}')
    if a.write and len(new):
        with open(a.overrides, 'a', encoding='utf-8') as f:
            f.write('# v2.2 (2026-10-19): hard overrides of the remaining wrong verdicts and of the editor\'s uncertain identifications (scripts/make_hard_overrides.py)\n')
            new.to_csv(f, sep='\t', index=False, header=False)
        MO.validate(MO.read_tsv(a.overrides))
    return 0


if __name__ == '__main__':
    sys.exit(main())
