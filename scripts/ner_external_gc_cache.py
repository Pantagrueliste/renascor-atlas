"""Remove the blobs of the Hugging Face cache that no snapshot refers to any more (left behind when a repo is re-downloaded with a narrower file list).

    python3 -I scripts/ner_external_gc_cache.py [--cache ~/renascor-atlas-data/ner/hf_cache] [--apply]

Dry run by default.  Only touches <cache>/blobs; *.incomplete files younger than 1 h (a running download) are never removed.
"""
import argparse
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default=str(Path.home() / 'renascor-atlas-data/ner/hf_cache'))
ap.add_argument('--apply', action='store_true')
a = ap.parse_args()
cache = Path(a.cache)
ref = set()
for d in cache.glob('models--*'):
    for p in (d / 'snapshots').rglob('*'):
        if p.is_symlink():
            ref.add(os.path.realpath(p))
tot = freed = 0
for b in (cache / 'blobs').rglob('*'):
    if not b.is_file():
        continue
    tot += 1
    if os.path.realpath(b) in ref:
        continue
    if b.name.endswith('.incomplete') and time.time() - b.stat().st_mtime < 3600:
        continue
    freed += b.stat().st_size
    if a.apply:
        b.unlink()
print(f'{tot} blobs, {len(ref)} referenced, {"removed" if a.apply else "removable"} {freed / 1e9:.1f} GB')
