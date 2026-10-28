"""Reconcile stage, part 2b: does a coordinate agree with the place NAME? (run on Hypatia through Slurm)

    python -m ratlas reconcile namecheck --geo places_local_geo.parquet --local places_local.parquet \\
        --geonames-zip allCountries.zip --dest namecheck_local.tsv

For every edition-scoped place with a register coordinate and/or an authority (Wikidata / GeoNames) coordinate, the
distance in km from that coordinate to the nearest GeoNames feature (classes P A T H L S) that carries the same
normalised name (name, ASCII name, alternate names; the place name, its variants, the part before a comma, the text
outside and inside parentheses). d = -1: no GeoNames feature of that name; empty: no such coordinate.
It is independent of Wikidata and of the ids the editors gave, so it can tell which of two conflicting coordinates is
right (cavriana: TGN ids that Wikidata maps to Illinois; selbstzeugnisse_hab_diarium: copied register coordinates).
Run it on the output of a first ``reconcile build`` (pass 1, without namecheck_local.tsv), then build again (pass 2).
Needs the GeoNames dump (CC BY 4.0), downloaded once by ``reconcile geonames``.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEATURE_CLASSES = ('P', 'A', 'T', 'H', 'L', 'S')
# words that name a kind of place, not a place: a GeoNames feature called 'Stadt' proves nothing
STOP = {'stadt', 'kloster', 'bistum', 'fluss', 'insel', 'kirche', 'schloss', 'burg', 'haus', 'republik', 'furstentum',
        'konigreich', 'herzogtum', 'grafschaft', 'church', 'castle', 'river', 'island', 'abbey', 'palace', 'street',
        'bridge', 'gate', 'parish', 'hall', 'manor', 'hospital', 'house', 'inn', 'school', 'college', 'market'}


def _t(msg):
    print(f'[namecheck {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def nrm(s) -> str:
    """Same folding as namegate.nrm (v2): small combining letters of print (o + U+0364 = oe), ligatures, long s."""
    from .namegate import nrm as _nrm
    return _nrm(s)


def name_keys(name, variants) -> set:
    out = set()
    for n in [name] + list(variants if variants is not None else []):
        if not isinstance(n, str) or not n:
            continue
        for c in [n, re.sub(r'\([^)]*\)', ' ', n)] + re.split(r'[,/()]', n):
            k = nrm(c)
            if len(k) > 1 and k not in STOP:
                out.add(k)
    return out


def min_km(lat: float, lon: float, pts: np.ndarray) -> float:
    p1, p2 = np.radians(lat), np.radians(pts[:, 0])
    x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(pts[:, 1] - lon) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(x, 0, 1))).min())


def run(geo_path: Path, local_path: Path, zip_path: Path, dest: Path) -> pd.DataFrame:
    geo = pq.read_table(geo_path).to_pandas()
    pl = pq.read_table(local_path, columns=['coll_slug', 'entity_key', 'variants']).to_pandas()
    geo = geo.merge(pl, on=['coll_slug', 'entity_key'], how='left')
    geo = geo[geo['register_has_coords'] | geo['auth_lat'].notna()].copy()
    _t(f'{len(geo)} places with a coordinate')
    keys = [name_keys(n, v) for n, v in zip(geo['name'], geo['variants'])]
    target = set().union(*keys) if keys else set()
    hits: dict = {}
    with zipfile.ZipFile(zip_path) as z, z.open('allCountries.txt') as fh:
        for line in io.TextIOWrapper(fh, encoding='utf-8'):
            f = line.split('\t', 8)
            if len(f) < 8 or f[6] not in FEATURE_CLASSES:
                continue
            nm = {nrm(f[1]), nrm(f[2])} | {nrm(a) for a in f[3].split(',') if a}
            nm &= target
            if nm:
                pt = (float(f[4]), float(f[5]))
                for n in nm:
                    hits.setdefault(n, []).append(pt)
    _t(f'{len(hits)} of {len(target)} names found in GeoNames')
    arr = {k: np.array(v) for k, v in hits.items()}

    def dist(ks, la, lo):
        if not (la == la and lo == lo):
            return np.nan
        best, found = np.inf, False
        for k in ks:
            a = arr.get(k)
            if a is not None:
                found = True
                best = min(best, min_km(la, lo, a))
        return best if found else -1.0
    d_reg = [dist(k, la, lo) for k, la, lo in zip(keys, geo['register_lat'], geo['register_lon'])]
    d_auth = [dist(k, la, lo) for k, la, lo in zip(keys, geo['auth_lat'], geo['auth_lon'])]
    out = pd.DataFrame({'coll_slug': geo['coll_slug'].to_numpy(), 'entity_key': geo['entity_key'].to_numpy(),
                        'd_reg': np.round(d_reg, 1), 'd_auth': np.round(d_auth, 1)})
    out.to_csv(dest, sep='\t', index=False)
    _t(f'written {dest}')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile namecheck')
    ap.add_argument('--geo', type=Path, required=True, help='places_local_geo.parquet of a first build')
    ap.add_argument('--local', type=Path, required=True, help='places_local.parquet (variants)')
    ap.add_argument('--geonames-zip', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args(argv)
    run(a.geo, a.local, a.geonames_zip, a.dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
