"""Reconcile stage, part 2b: coordinates for GeoNames ids from the GeoNames dump (run on Hypatia through Slurm).

    python -m ratlas reconcile geonames --ids ids.tsv --zip allCountries.zip --out geonames_lookup.tsv

Streams ``allCountries.txt`` once and keeps the rows of the ids found on places (and on persons, as place ids).
GeoNames data: CC BY 4.0, https://www.geonames.org (attribution is recorded in the output header comment and in
globe_places.json).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

COLS = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature_class', 'feature_code',
        'country_code', 'cc2', 'admin1', 'admin2', 'admin3', 'admin4', 'population', 'elevation', 'dem', 'timezone',
        'modification']
KEEP = ['geonameid', 'name', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country_code', 'population']


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile geonames')
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    need = set()
    with open(a.ids, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['scheme'] == 'geonames' and r['kind'] == 'place':
                need.add(r['id'])
    print(f'{len(need)} geonames ids wanted', flush=True)
    found = 0
    with zipfile.ZipFile(a.zip) as z, open(a.out, 'w', encoding='utf-8', newline='') as out:
        out.write('# GeoNames gazetteer extract, CC BY 4.0, https://www.geonames.org (allCountries.zip)\n')
        w = csv.writer(out, delimiter='\t')
        w.writerow(KEEP)
        with z.open('allCountries.txt') as raw:
            for n, line in enumerate(io.TextIOWrapper(raw, encoding='utf-8', newline='')):
                i = line.split('\t', 1)[0]
                if i in need:
                    p = line.rstrip('\n').split('\t')
                    w.writerow([p[COLS.index(k)] for k in KEEP])
                    found += 1
                if n % 5_000_000 == 0:
                    print(f'  {n} lines, {found} found ({time.time() - t0:.0f}s)', flush=True)
    print(f'{found}/{len(need)} found in {time.time() - t0:.0f}s -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
