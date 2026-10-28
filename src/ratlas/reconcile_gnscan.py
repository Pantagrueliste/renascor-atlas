"""Reconcile stage, part 2d: names of the GeoNames features behind the ids, and their same-name neighbours (Hypatia, Slurm).

    python -m ratlas reconcile gnscan --ids ids.tsv --local places_local.parquet --zip allCountries.zip --dest RDIR

One pass over ``allCountries.txt`` (GeoNames, CC BY 4.0, https://www.geonames.org) writes

* ``gn_ids.parquet``       one row per GeoNames id of ``ids.tsv`` (places): gid, name, names (normalised name, ascii name and every
                           alternate name, ``|``-joined), lat, lon, pop, cc, fcode, fclass. The name gate compares the entry's names
                           with these (the Latin and old forms are alternate names).
* ``gn_samename.parquet``  key, gid, lat, lon, pop, cc, fcode, names for the features of classes P A T H L whose normalised name or
                           alternate name equals a name of a place entity that has an authority id (``names``: all the normalised names
                           of the feature, ``|``-joined). Used by the homonym gate (a pop-0 hamlet with the town's name) and as independent
                           evidence for old forms: an entry name that is an alternate name of a feature next to the authority's point,
                           and that same feature also carries one of the authority item's names (Venedig and Venezia; not Halle and
                           Altstadt). v2 audit, check 1.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import namegate as NG

CLASSES = frozenset('PATHL')
AUTH_COLS = ('gnd', 'viaf', 'wikidata', 'geonames', 'tgn', 'pleiades', 'hls')


def _t(msg):
    print(f'[gnscan {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def target_keys(local: Path) -> set:
    t = pq.read_table(local, columns=['name', 'variants', 'other_auth'] + list(AUTH_COLS)).to_pandas()
    has = t[list(AUTH_COLS)].notna().any(axis=1) | t['other_auth'].map(lambda v: v is not None and len(v) > 0)
    t = t[has]
    keys: set = set()
    for n, v in zip(t['name'], t['variants']):
        keys |= NG.name_keys(n, v)
    return {k for k in keys if len(k) >= 3}


def run(ids: Path, local: Path, zpath: Path, dest: Path) -> None:
    t0 = time.time()
    need = set()
    with open(ids, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['scheme'] == 'geonames' and r['kind'] == 'place':
                need.add(r['id'])
    keys = target_keys(local)
    _t(f'{len(need)} geonames ids, {len(keys)} entry name keys')
    nrm = lru_cache(maxsize=2_000_000)(NG.nrm)
    gn_rows, same_rows = [], []
    with zipfile.ZipFile(zpath) as z, z.open('allCountries.txt') as fh:
        for n, line in enumerate(io.TextIOWrapper(fh, encoding='utf-8', newline='')):
            p = line.split('\t', 15)
            if len(p) < 15:
                continue
            gid = p[0]
            want_id = gid in need
            if not want_id and p[6] not in CLASSES:
                continue
            alts = [a for a in p[3].split(',') if a]
            nm = {nrm(p[1]), nrm(p[2])} | {nrm(a) for a in alts}
            nm.discard('')
            pop = int(p[14]) if p[14].isdigit() else 0
            if want_id:
                gn_rows.append((gid, p[1], '|'.join(sorted(nm)), float(p[4]), float(p[5]), pop, p[8], p[7], p[6]))
            if p[6] in CLASSES:
                hit = nm & keys
                for k in hit:
                    same_rows.append((k, gid, float(p[4]), float(p[5]), pop, p[8], p[7], '|'.join(sorted(nm))))
            if n % 3_000_000 == 0:
                _t(f'{n} lines, {len(gn_rows)} ids, {len(same_rows)} same-name rows ({time.time() - t0:.0f}s)')
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gn_rows, columns=['gid', 'name', 'names', 'lat', 'lon', 'pop', 'cc', 'fcode', 'fclass']).to_parquet(
        dest / 'gn_ids.parquet', index=False)
    pd.DataFrame(same_rows, columns=['key', 'gid', 'lat', 'lon', 'pop', 'cc', 'fcode', 'names']).to_parquet(
        dest / 'gn_samename.parquet', index=False)
    _t(f'{len(gn_rows)}/{len(need)} ids found, {len(same_rows)} same-name rows ({time.time() - t0:.0f}s)')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile gnscan')
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--local', type=Path, required=True)
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args(argv)
    run(a.ids, a.local, a.zip, a.dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
