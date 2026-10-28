"""Manual-review overrides (v2, 2026-10-17): corrections of the gates that a person confirmed by hand, so that they survive regeneration.

Three small tables sit next to ``coord_overrides.tsv`` in ``--rdir`` (``data/`` of the repository is the source of truth; copy the files to
the reconcile directory before ``reconcile build2``):

* ``id_overrides.tsv`` -- changes the decision of the gates for one authority id of one entity.
  Columns: ``coll_slug kind entity_key scheme id qid action case_id note``.
  ``action``: ``accept`` | ``reject`` | ``suggest``. A row matches the gate rows of the pair (``scheme``, ``id``) of the entity; with
  ``scheme = *`` it matches every pair of the entity that leads to ``qid`` (an id that leads to two items is then handled item by item).
  ``accept`` with ``qid = -`` accepts the id and drops the Wikidata item it leads to (the item is a bot duplicate or a wrong
  mapping; the id itself, a GeoNames or TGN record, is right). ``accept`` of a pair the gates did not produce (a Wikipedia link lost at
  extraction, a QID that replaces a wrong one) needs ``scheme = wikidata``, ``id = qid = Q...``: the pair is created.
  A manual decision has the reason ``manual_review`` and is listed in ``gate_log.tsv`` when it is not an acceptance.
* ``coord_overrides.tsv`` -- unchanged (a register coordinate corrected by hand).
* ``geocode_rejects.tsv`` -- ``coll_slug entity_key rule note``: a name-based result of the geocoder that a person rejected; it is treated
  like the results taken back by the audit rules (``geocode_verify.tsv``): kept as a low-confidence candidate, the place stays unplaced.

``manual_review.tsv`` (``RQ/derived/atlas/v2/``) lists every case with its evidence; each override row carries its ``case_id``.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ACTIONS = ('accept', 'reject', 'suggest')
KINDS = ('place', 'person')
COLS = ['coll_slug', 'kind', 'entity_key', 'scheme', 'id', 'qid', 'action', 'case_id', 'note']
REASON = 'manual_review'
_QID = re.compile(r'^Q[1-9]\d*$')


def read_tsv(path: Path) -> pd.DataFrame:
    """A tab-separated table whose leading lines may be comments (``#`` at the start of a line only: notes may contain ``#``)."""
    text = Path(path).read_text(encoding='utf-8')
    lines = [l for l in text.splitlines() if not l.startswith('#')]
    return pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', dtype=str, keep_default_na=False)


def load_id_overrides(rdir: Path) -> pd.DataFrame:
    """The validated table (empty when the file is missing). Raises ValueError on a malformed row."""
    f = Path(rdir) / 'id_overrides.tsv'
    if not f.exists():
        return pd.DataFrame(columns=COLS)
    return validate(read_tsv(f))


def validate(o: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in COLS if c not in o.columns]
    if miss:
        raise ValueError(f'id_overrides: missing columns {miss}')
    o = o[COLS].copy()
    bad = o[~o['action'].isin(ACTIONS)]
    if len(bad):
        raise ValueError(f'id_overrides: unknown action {sorted(set(bad["action"]))}')
    bad = o[~o['kind'].isin(KINDS)]
    if len(bad):
        raise ValueError(f'id_overrides: unknown kind {sorted(set(bad["kind"]))}')
    for c in ('coll_slug', 'entity_key', 'scheme'):
        if (o[c].str.strip() == '').any():
            raise ValueError(f'id_overrides: empty {c}')
    star = o['scheme'] == '*'
    if (star & ~o['qid'].str.match(_QID)).any():
        raise ValueError('id_overrides: scheme * needs a QID')
    if (~star & (o['id'].str.strip() == '')).any():
        raise ValueError('id_overrides: empty id')
    if (~o['qid'].isin(['', '-']) & ~o['qid'].str.match(_QID)).any():
        raise ValueError('id_overrides: malformed qid')
    if ((o['qid'] == '-') & (o['action'] != 'accept')).any():
        raise ValueError('id_overrides: qid - only with accept')
    both = o.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id', 'qid'])['action'].nunique()
    if (both > 1).any():
        raise ValueError(f'id_overrides: contradictory rows {list(both[both > 1].index[:3])}')
    return o.drop_duplicates().reset_index(drop=True)


def apply_id_overrides(pairs: pd.DataFrame, ov: pd.DataFrame, ents: pd.DataFrame, kind: str, known_qids=None,
                       log=lambda m: None) -> tuple[pd.DataFrame, dict]:
    """Apply the overrides of ``kind`` to the gate pairs (before ``choose``). ``ents``: the local table of the kind (name, n_mentions ...).
    ``known_qids``: the Wikidata items that the build can read (a created pair for another QID would lead nowhere); None = no check.
    Returns (pairs, report)."""
    rep = {'rows': 0, 'accepted': 0, 'rejected': 0, 'suggested': 0, 'created': 0, 'unmatched': 0, 'unknown_qid': 0}
    o = ov[ov['kind'] == kind] if len(ov) else ov
    if o is None or o.empty:
        return pairs, rep
    pairs = pairs.copy()
    if 'qid' in pairs and pairs['qid'].dtype != object:
        pairs['qid'] = pairs['qid'].astype(object)
    ix = {}
    for i, k in enumerate(zip(pairs['coll_slug'], pairs['entity_key'])):
        ix.setdefault(k, []).append(i)
    E = ents.drop_duplicates(['coll_slug', 'entity_key']).set_index(['coll_slug', 'entity_key']) if len(ents) else None
    new_rows = []
    status = {'accept': 'accepted', 'reject': 'rejected', 'suggest': 'suggested'}
    for r in o.itertuples(index=False):
        rep['rows'] += 1
        rows = ix.get((r.coll_slug, r.entity_key), [])
        sub = pairs.iloc[rows] if rows else pairs.iloc[0:0]
        if r.scheme == '*':
            hit = sub.index[(sub['qid'] == r.qid)] if len(sub) else []
        else:
            hit = sub.index[(sub['scheme'] == r.scheme) & (sub['id'] == r.id)] if len(sub) else []
        detail = f'{r.case_id}: {r.note}'[:300]
        if len(hit):
            loc = pairs.index.get_indexer(hit)
            pairs.iloc[loc, pairs.columns.get_loc('status')] = status[r.action]
            pairs.iloc[loc, pairs.columns.get_loc('reason')] = REASON
            pairs.iloc[loc, pairs.columns.get_loc('detail')] = detail
            if r.qid == '-':
                pairs.iloc[loc, pairs.columns.get_loc('qid')] = None
            rep[status[r.action]] += len(hit)
        elif r.action == 'accept' and r.scheme == 'wikidata' and _QID.match(r.id) and r.qid == r.id:
            if known_qids is not None and r.id not in known_qids:
                rep['unknown_qid'] += 1
                log(f'manual overrides: {r.case_id} {r.id} is not in the Wikidata table: skipped')
                continue
            name, n = r.entity_key, 0
            if E is not None and (r.coll_slug, r.entity_key) in E.index:
                e = E.loc[(r.coll_slug, r.entity_key)]
                name = e['name']
                n = int(e.get('n_mentions_all', e.get('n_mentions', 0)) or 0)
            new_rows.append(dict(coll_slug=r.coll_slug, kind=kind, entity_key=r.entity_key, name=name, scheme='wikidata', id=r.id,
                                 n_mentions=n, n_in_period=n, qid=r.id, qid_raw=r.id, src='manual', level='manual', label=None,
                                 second_id='', k=1.0, authority_lat=np.nan, authority_lon=np.nan, status='accepted', reason=REASON,
                                 detail=detail))
            rep['created'] += 1
        else:
            rep['unmatched'] += 1
            log(f'manual overrides: {r.case_id} {r.coll_slug} {r.entity_key} {r.scheme}:{r.id}: no gate row')
    if new_rows:
        add = pd.DataFrame(new_rows)
        for c in pairs.columns:
            if c not in add:
                add[c] = None
        pairs = pd.concat([pairs, add[pairs.columns]], ignore_index=True)
    log(f'manual overrides ({kind}): {rep}')
    return pairs, rep


def effective_verify(verify: Optional[Path], rdir: Path, dest: Path, log=lambda m: None) -> Optional[Path]:
    """geocode_verify.tsv plus the hand-made ``geocode_rejects.tsv`` of ``rdir``; the merged file is written to ``dest`` and returned."""
    f = Path(rdir) / 'geocode_rejects.tsv'
    if not f.exists():
        return verify
    rej = read_tsv(f)
    if rej.empty:
        return verify
    for c in ('coll_slug', 'entity_key'):
        if c not in rej:
            raise ValueError(f'geocode_rejects: missing column {c}')
    if 'rule' not in rej:
        rej['rule'] = REASON
    rej['rule'] = rej['rule'].replace('', REASON)
    if verify is not None and Path(verify).exists():
        v = pd.read_csv(verify, sep='\t', dtype=str, keep_default_na=False)
        out = pd.concat([v, rej[[c for c in rej.columns if c in v.columns or c in ('coll_slug', 'entity_key', 'rule')]]], ignore_index=True)
    else:
        out = rej[['coll_slug', 'entity_key', 'rule']]
    out = out.drop_duplicates(['coll_slug', 'entity_key'], keep='first')
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / 'geocode_verify_effective.tsv'
    out.to_csv(p, sep='\t', index=False)
    log(f'geocode rejects: {len(rej)} manual rows merged into {len(out)} -> {p}')
    return p
