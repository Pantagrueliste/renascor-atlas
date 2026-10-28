"""Duplicate points merged, labels and QIDs fixed (owner decision 3 of 2026-10-28), and the way model points are found again in a rebuilt layer.

``data/curation/place_merges.json`` holds, for the points of the globe page:

* ``merges`` -- ``keep`` absorbs ``absorb``. Ids: ``e:N`` = row N of D.rows of the page build after the homonym audit (``info`` gives the name and the
  point of each id, checked at build time); ``ml2:N`` = a point of the model layer, found again by ``model_keys`` (see below). ``reading`` says in which
  reading the merge matters (both | augmented); ``source`` is duplicates (systematic search), dicotopo (same DicoTopo article) or integration (same point and name, found when the build was checked).
* ``labels`` -- point-level label fixes: ``Arragon`` -> ``Aragon``, ``Country, Province, Place`` register labels reduced to the place name ...
* ``qid_fixes`` -- a wrong QID (the point Belgium of the editors is Brussels, Q239, not Q31).
* ``model_keys`` -- for every ml2 id used by this file (and by ``dicotopo_names.tsv``), the ``prev`` list of the point: the units of the layer it was made of.
  The ml2 ids are the rank of the point by mentions, so they move when the layer is rebuilt (titles removed, verdicts added);
  ``match_model_points`` finds each point again by the units it holds.

``data/curation/place_keep_apart.tsv`` is the list of pairs read and kept apart; ``plan`` refuses to merge a pair that is on it.

The functions work on the plain structures of the page builder (a list of rows ``[name, lat, lon, q10, editions, top, qid, code, tier, mentions, original name,
kind, per-edition counts, docs shard, members, gone]``), so that they can be tested without the atlas tables.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'curation'
FILE = DATA / 'place_merges.json'
KEEP_APART = DATA / 'place_keep_apart.tsv'
MEMBERS, GONE = 14, 15                  # fields of a row


def load(path=None) -> dict:
    d = json.loads(Path(path or FILE).read_text('utf-8'))
    for m in d['merges']:
        if m['keep'] in m['absorb']:
            raise ValueError(f'place_merges: {m["keep"]} absorbs itself')
    d.setdefault('unlink', [])
    return d


def keep_apart_pairs(path=None) -> set:
    t = pd.read_csv(path or KEEP_APART, sep='\t', keep_default_na=False, dtype=str)
    return {frozenset((a, b)) for a, b in zip(t['a'], t['b'])}


# ----------------------------------------------------------------------------------------------------------------------- model points
def match_model_points(model_keys: dict, places: list) -> dict:
    """old ml2 id -> id of the point of ``places`` (a rebuilt layer) that holds most of its units, or None when no unit is left.
    A tie goes to the point with more mentions."""
    by_unit = collections.defaultdict(list)
    for p in places:
        for u in p['prev']:
            by_unit[u].append(p['id'])
    n = {p['id']: p.get('n', 0) for p in places}
    out = {}
    for old, units in model_keys.items():
        cnt = collections.Counter(pid for u in units for pid in by_unit.get(u, ()))
        out[old] = max(cnt, key=lambda pid: (cnt[pid], n[pid])) if cnt else None
    return out


# ----------------------------------------------------------------------------------------------------------------------- merges
def _km(a, b, c, d):
    r = math.pi / 180
    h = math.sin((c - a) * r / 2) ** 2 + math.cos(a * r) * math.cos(c * r) * math.sin((d - b) * r / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _row_id(s):
    return int(s[2:]) if s.startswith('e:') else None


def lead_map(rows) -> dict:
    """row index -> index of the row that leads its merged point (itself when alone)."""
    lead = {}
    for i, r in enumerate(rows):
        if len(r) > MEMBERS and r[MEMBERS]:
            for j, _ in r[MEMBERS]:
                lead[j] = i
    return lead


def plan(merges: list, rows: list, model_map: dict, model_info: dict | None = None, keep_apart: set | None = None,
         sources: tuple = ('duplicates', 'dicotopo', 'integration')) -> tuple[list, list]:
    """The merges to apply, in order, as dicts (keep: row index, absorb: row indices, models: new ml2 ids, entry), and a list of the entries left out with the reason.

    An entry is left out when the id of a row does not carry the name or the point it had when the entry was made (the rows moved), when every point of
    it is gone, when a model point cannot be found again, or when the pair is on the keep-apart list. Points already joined by an earlier entry
    are followed to their present lead, so two entries about the same point do not clash.
    ``model_info`` = new ml2 id -> (lat, lon), to check the model points."""
    keep_apart = keep_apart or set()
    lead = lead_map(rows)
    L = lambda i: lead.get(i, i)
    done, left = [], []
    absorbed_into = {}                                        # row -> lead it was absorbed into by an earlier entry

    def follow(i):
        while i in absorbed_into:
            i = absorbed_into[i]
        return L(i)

    for e in merges:
        if e['source'] not in sources:
            continue
        ids = [e['keep']] + list(e['absorb'])
        reason = None
        rws, mods = [], []
        for pid in ids:
            info = e.get('info', {}).get(pid)
            if pid.startswith('e:'):
                i = _row_id(pid)
                if i >= len(rows):
                    reason = f'{pid} does not exist'
                    break
                r = rows[i]
                if info and (r[0] != info['name'] or abs(r[1] - info['lat']) > 0.01 or abs(r[2] - info['lon']) > 0.01):
                    reason = f'{pid} is {r[0]!r} at {r[1]},{r[2]}, the entry was made on {info["name"]!r} at {info["lat"]},{info["lon"]}'
                    break
                rws.append(i)
            else:
                nid = model_map.get(pid)
                if nid is None:
                    reason = f'{pid} is not in the layer any more'
                    break
                if info and model_info and nid in model_info and _km(info['lat'], info['lon'], *model_info[nid]) > 20:
                    reason = f'{pid} -> {nid} is {_km(info["lat"], info["lon"], *model_info[nid]):.0f} km from the point of the entry'
                    break
                mods.append(nid)
        if reason:
            left.append({'keep': e['keep'], 'absorb': e['absorb'], 'reason': reason})
            continue
        rl = []
        for i in rws:
            j = follow(i)
            if j not in rl:
                rl.append(j)
        live = [j for j in rl if not (len(rows[j]) > GONE and rows[j][GONE] == 1 and not rows[j][MEMBERS])]
        if not live and not mods:
            left.append({'keep': e['keep'], 'absorb': e['absorb'], 'reason': 'every point of the entry is gone'})
            continue
        if not live:
            left.append({'keep': e['keep'], 'absorb': e['absorb'], 'reason': 'no editor row left to merge into'})
            continue
        bad = next((pr for pr in keep_apart if len(pr) == 2 and all(x in ids for x in pr)), None)
        if bad:
            left.append({'keep': e['keep'], 'absorb': e['absorb'], 'reason': f'{sorted(bad)} is on the keep-apart list'})
            continue
        keep, ab = live[0], live[1:]
        for j in ab:
            absorbed_into[j] = keep
        done.append({'keep': keep, 'absorb': ab, 'models': mods, 'entry': e})
    return done, left


def merge_rows(rows: list, keep: int, absorbed: list) -> None:
    """Join the points ``absorbed`` (lead rows) to the point ``keep``: their members become members of ``keep`` and their rows are gone.
    The figures of ``keep`` are recomputed by the caller from the mentions of its members."""
    def pad(r):
        while len(r) <= GONE:
            r.append(None)
    r = rows[keep]
    pad(r)
    if not r[MEMBERS]:
        r[MEMBERS] = [[keep, r[13]]]
    seen = {j for j, _ in r[MEMBERS]}
    for a in absorbed:
        ra = rows[a]
        pad(ra)
        for j, sh in (ra[MEMBERS] or [[a, ra[13]]]):
            if j not in seen:
                r[MEMBERS].append([j, sh])
                seen.add(j)
        ra[MEMBERS] = None
        ra[GONE] = 1


# ----------------------------------------------------------------------------------------------------------------------- labels and QIDs
def apply_labels(rows: list, labels: list, model_names: dict | None = None, model_map: dict | None = None) -> dict:
    """Rename rows (``e:N``) and model points (``ml2:N`` through ``model_map``, into ``model_names`` = new id -> name).
    A fix is applied when the point still carries the label it was written for; the others are counted, with the reason."""
    rep = {'rows': 0, 'model': 0, 'skipped_gone': 0, 'skipped_changed': 0, 'skipped_missing': 0, 'skipped': []}
    for x in labels:
        pid = x['point']
        if pid.startswith('e:'):
            i = _row_id(pid)
            r = rows[i] if i < len(rows) else None
            if r is None:
                rep['skipped_missing'] += 1
            elif len(r) > GONE and r[GONE] == 1:
                rep['skipped_gone'] += 1
            elif r[0] != x['old']:
                rep['skipped_changed'] += 1
                rep['skipped'].append((pid, r[0], x['old'], x['new']))
            else:
                r[0] = x['new']
                rep['rows'] += 1
        elif model_names is not None and model_map is not None:
            nid = model_map.get(pid)
            if nid is None or nid not in model_names:
                rep['skipped_missing'] += 1
            else:
                model_names[nid] = x['new']
                rep['model'] += 1
    return rep


def apply_qid_fixes(rows: list, fixes: list) -> int:
    n = 0
    for f in fixes:
        i = _row_id(f['point'])
        if i < len(rows) and rows[i][6] == f['old']:
            rows[i][6] = f['new']
            n += 1
    return n
