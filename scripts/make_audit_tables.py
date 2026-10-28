"""One-off generator of the three small tables that ``audit_v2`` reads from ``data/`` (run on the Mac, reads the audit folder read-only).

    python scripts/make_audit_tables.py --audit RQ/derived/atlas/audit --v2 RQ/derived/atlas/v2

* ``data/audit_confirmed_wrong.tsv``  the pairs the audit confirmed as wrong (place MISMATCH class; person rows of severity high for
                                      nonhuman_qid_error and chronology_birth_after_documents), one row each. ``bad_qid`` is the wrong item
                                      the id led to (the id itself may be right: Balingen's GeoNames id sits on the Wikidata item Bisingen).
                                      The audit's person ``name_mismatch`` rows are left out: they were judged with 4 label languages and
                                      the names are now judged by the frozen matcher on 10 (check 1 itself).
* ``data/freshness_audit.tsv``        slug, changed_since_harvest, harvest_date, harvest_basis of the freshness audit.
* ``data/legacy_sources.tsv``         the harvest folders without a SOURCES.tsv (read from the v2 mentions), with what the freshness audit says.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / 'data'


def prefix(p: str) -> str:
    q = p.split('/')
    return '/'.join(q[:2]) if q[0].startswith('originals_') and len(q) > 1 else q[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--v2', type=Path, required=True)
    ap.add_argument('--raw', type=Path, required=True)
    a = ap.parse_args()
    pa = pd.read_csv(a.audit / 'place_authority_check.tsv', sep='\t', dtype=str, keep_default_na=False)
    pa = pa[pa['class'] == 'MISMATCH']
    rows = [dict(kind='place', coll_slug=r.edition, entity_key=r.entry_id, entry_name=r.entry_name, scheme=r.scheme, id=r.authority_id, qid='',
                 bad_qid=(r.resolves_to if r.resolves_to[:1] == 'Q' and r.resolves_to[1:].isdigit() else ''),
                 audit_check='place MISMATCH', note=(r.review or r.evidence)[:200]) for r in pa.itertuples(index=False)]
    pe = pd.read_csv(a.audit / 'person_authority_check.tsv', sep='\t', dtype=str, keep_default_na=False)
    pe = pe[(pe['severity'] == 'high') & pe['check'].isin(['nonhuman_qid_error', 'chronology_birth_after_documents'])]
    rows += [dict(kind='person', coll_slug=r.edition, entity_key=r.entity_key, entry_name=r.entry_name, scheme='wikidata', id=r.qid, qid=r.qid, bad_qid=r.qid,
                  audit_check=r.check, note=r.detail[:200]) for r in pe.itertuples(index=False)]
    cw = pd.DataFrame(rows).drop_duplicates(['kind', 'coll_slug', 'entity_key', 'scheme', 'id', 'audit_check'])
    cw.to_csv(DATA / 'audit_confirmed_wrong.tsv', sep='\t', index=False)
    fr = pd.read_csv(a.audit / 'freshness.tsv', sep='\t', dtype=str, keep_default_na=False)
    fr[['slug', 'changed_since_harvest', 'harvest_date', 'harvest_basis', 'source', 'local_ref']].to_csv(DATA / 'freshness_audit.tsv', sep='\t', index=False)
    m = pq.read_table(a.v2 / 'mentions.parquet', columns=['coll_slug', 'source_path']).to_pandas().drop_duplicates()
    m['prefix'] = m['source_path'].map(prefix)
    m = m.drop_duplicates(['coll_slug', 'prefix'])
    leg = []
    for r in m.itertuples(index=False):
        if (a.raw / r.prefix / 'SOURCES.tsv').exists():
            continue
        f = fr[fr['slug'] == r.coll_slug].iloc[0]
        leg.append(dict(slug=r.coll_slug, source_prefix=r.prefix, harvest_date=f.harvest_date, harvest_basis=f.harvest_basis,
                        changed_since_harvest=f.changed_since_harvest, source=f.source))
    pd.DataFrame(leg).sort_values(['slug', 'source_prefix']).to_csv(DATA / 'legacy_sources.tsv', sep='\t', index=False)
    print(len(cw), 'confirmed rows;', len(leg), 'legacy rows;', len(fr), 'freshness rows')


if __name__ == '__main__':
    main()
