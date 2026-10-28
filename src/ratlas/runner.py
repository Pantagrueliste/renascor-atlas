"""Run one collection module: prepare registers, extract every file, write out/<slug>/{mentions,entities}.parquet."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Optional

from . import common as C
from .collections import Collection, get

_STATE: dict = {}


def _work(arg):
    rec, data = arg
    rec.data = data
    try:
        ms, es = _STATE['col'].extract(rec, _STATE['ctx'])
        return rec.path, ms, es, None
    except Exception as e:   # noqa: BLE001  one bad file must not stop the run
        return rec.path, [], [], f'{type(e).__name__}: {e}'


def _feed(recs):
    for r in C.iter_with_data(recs):
        d = r.data
        r.data = None
        yield r, d


def run(slug: str, limit: Optional[int] = None, out: Optional[Path] = None, workers: int = 1,
        include_dups: bool = False, verbose: bool = True) -> dict:
    col: Collection = get(slug)
    ctx = col.make_context()
    t0 = time.time()
    files = col.iter_files(ctx, include_dups)
    files = [f for f in files if not C.is_forbidden(f.path)]
    ctx.all_files = files
    if verbose:
        print(f'[{slug}] {len(files)} files in {time.time() - t0:.1f}s', file=sys.stderr, flush=True)
    # a --limit run never overwrites the real output: it goes to <out_root>/_limit unless --out is given
    out_dir = Path(out) if out else (C.out_root() / '_limit' if limit else C.out_root())
    w = C.Writer(out_dir / slug, slug, col.licence, col.tier)
    reg_ents = col.prepare(ctx)
    w.add([], reg_ents)
    todo = files[:limit] if limit else files
    _STATE['col'], _STATE['ctx'] = col, ctx
    n = 0
    if workers > 1:
        pool = mp.get_context('fork').Pool(min(workers, 8))
        it = pool.imap_unordered(_work, _feed(todo), chunksize=4)
    else:
        pool = None
        it = (_work((r, r.data)) for r in C.iter_with_data(todo))
    try:
        for path, ms, es, err in it:
            w.note_file(path, err)
            w.add(ms, es)
            n += 1
            if verbose and n % 2000 == 0:
                print(f'[{slug}] {n}/{len(todo)} files, {w.s["mentions"]} mentions, {time.time() - t0:.0f}s',
                      file=sys.stderr, flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    stats = w.close()
    if verbose:
        print(json.dumps({k: stats[k] for k in ('files_read', 'file_errors', 'mentions', 'entities',
                                                'resolved_share', 'auth_share', 'in_period_share', 'seconds')}),
              file=sys.stderr, flush=True)
    return stats
