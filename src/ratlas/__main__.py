"""python -m ratlas list | extract <slug> [--limit N] [--out DIR] [--workers N] | files <coll>... | inspect <coll-or-prefix> """
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import common as C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list', help='list registered collection modules')
    p.add_argument('--counts', action='store_true', help='also count files from scan.jsonl.gz (slow)')
    p = sub.add_parser('extract', help='extract one collection module')
    p.add_argument('slug')
    p.add_argument('--limit', type=int, default=None, help='only the first N files (sorted by path)')
    p.add_argument('--out', type=Path, default=None, help='output root (default: RATLAS_OUT or derived/atlas/out)')
    p.add_argument('--workers', type=int, default=1, help='worker processes (Hypatia only; max 8)')
    p.add_argument('--include-dups', action='store_true')
    p = sub.add_parser('files', help='list raw files (archives, not members) needed for collections / prefixes')
    p.add_argument('selectors', nargs='+')
    p.add_argument('--siblings', action='store_true', help='also non-XML files next to the XML files (register csv/json)')
    p.add_argument('--extra', action='append', default=[], help='glob relative to the raw root to add')
    p = sub.add_parser('merge', help='integration: merge out/<slug> into out/all + collections table + local tables')
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--dest', type=Path, default=None)
    p.add_argument('--no-overlaps', action='store_true')
    p.add_argument('--v1-collections', type=Path, default=None, help='v1 collections.tsv: also writes deviations_vs_v1.tsv')
    p = sub.add_parser('reconcile', help='reconcile stage: ids | fetch | authority | geonames | gnscan | namecheck | build | geocode | verify (see ratlas.reconcile_*)')
    p.add_argument('step', choices=['ids', 'fetch', 'authority', 'geonames', 'gnscan', 'namecheck', 'build', 'build2', 'geocode', 'verify'])
    p.add_argument('rest', nargs=argparse.REMAINDER)
    p = sub.add_parser('audit', help='audit v2: the nine blocking checks on the v2 tables (ratlas.audit_v2)')
    p.add_argument('step', choices=['v2', 'editions'])
    p.add_argument('rest', nargs=argparse.REMAINDER)
    p = sub.add_parser('curation', help='homonym audit: check (rules and outliers of a build) | plan (what the fixes and rules change) (ratlas.curation)')
    p.add_argument('rest', nargs=argparse.REMAINDER)
    p = sub.add_parser('inspect', help='what the scan says about a coll / path prefix (files, dups, tag counts, samples)')
    p.add_argument('selector')
    a = ap.parse_args(argv)

    if a.cmd == 'list':
        from .collections import discover
        for slug, col in sorted(discover().items()):
            m = col.module
            n = ''
            if a.counts:
                n = f'\t{len(C.iter_collection_files(m.COLLS))} files'
            print(f'{slug}\t{m.TIER}\t{m.LICENCE}\t{",".join(m.RENASCOR_IDS)}\t{";".join(m.COLLS)}{n}')
        return 0
    if a.cmd == 'extract':
        from .runner import run
        run(a.slug, a.limit, a.out, a.workers, a.include_dups)
        return 0
    if a.cmd == 'merge':
        from .merge import main as merge_main
        return merge_main(sys.argv[2:])
    if a.cmd == 'curation':
        from .curation import main as cur_main
        return cur_main(a.rest)
    if a.cmd == 'reconcile':
        import importlib
        mod = importlib.import_module({'ids': '.reconcile_ids', 'fetch': '.reconcile_fetch', 'authority': '.authority_fetch', 'gnscan': '.reconcile_gnscan', 'geonames': '.reconcile_geonames', 'namecheck': '.reconcile_namecheck', 'build': '.reconcile_build', 'build2': '.reconcile_v2', 'geocode': '.reconcile_geocode', 'verify': '.verify_geocode'}[a.step], 'ratlas')
        return mod.main(a.rest)
    if a.cmd == 'audit':
        if a.step == 'editions':
            from .audit_editions import main as ed_main
            return ed_main(a.rest)
        from .audit_v2 import main as audit_main
        return audit_main(a.rest)
    if a.cmd == 'files':
        from .pushlist import push_files
        for p in push_files(a.selectors, siblings=a.siblings, extra=a.extra):
            print(p)
        return 0
    if a.cmd == 'inspect':
        from collections import Counter
        recs = list(C.iter_scan([a.selector], include_dups=True))
        c = Counter()
        for r in recs:
            if not r.dup:
                c.update(r.c)
        print(f'{len(recs)} records ({sum(r.dup for r in recs)} dup), {len({r.coll for r in recs})} coll key(s): '
              f'{sorted({r.coll for r in recs})[:5]}')
        print('archives:', sorted({r.archive for r in recs if "!" in r.path})[:10])
        print('tag counts (non-dup):', {k: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:25]})
        for r in recs[:5]:
            print(' ', r.path)
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
