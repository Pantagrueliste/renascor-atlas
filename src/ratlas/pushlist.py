"""Files to ship to Hypatia for given collections (used by scripts/push_coll.sh)."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

from . import common as C


def push_files(selectors: Iterable[str], siblings: bool = False, extra: Iterable[str] = (),
               max_sibling_bytes: int = 200_000_000) -> list[str]:
    """Raw-root-relative paths of the plain files and archives that hold the XML of the selectors.

    Every scan record is included (duplicates too: a register may be a duplicate of a file in another collection),
    not only the tagged ones. ``siblings`` adds non-XML files from the same directories (register csv/json...).
    ``extra`` are globs relative to the raw root.
    """
    root = C.raw_root()
    files: set[str] = set()
    dirs: set[str] = set()
    for rec in C.iter_scan(list(selectors), include_dups=True):
        files.add(rec.archive)
        if '!' not in rec.path:
            dirs.add(str(Path(rec.path).parent))
    if siblings:
        for d in dirs:
            for p in (root / d).iterdir():
                if p.is_file() and not p.name.startswith('._') and p.stat().st_size <= max_sibling_bytes:
                    files.add(str(p.relative_to(root)))
    for g in extra:
        for p in glob.glob(str(root / g), recursive=True):
            if Path(p).is_file():
                files.add(str(Path(p).relative_to(root)))
    return sorted(f for f in files if not C.is_forbidden(f))
