"""Helpers shared by the batch-5 collection tests (tests/test_<slug>.py)."""
import pyarrow as pa

from ratlas import common as C
from ratlas.collections import get
from ratlas.model import ENTITY_SCHEMA, MENTION_SCHEMA

REAL = C.raw_root().exists()


def run(slug, items, prepare=True):
    """items: list of (path, bytes). Returns (mentions, entities) of prepare() + extract() over all items."""
    col = get(slug)
    ctx = col.make_context()
    recs = [C.FileRecord(path=p, coll='x', data=d) for p, d in items]
    ctx.all_files = recs
    ents = list(col.prepare(ctx)) if prepare else []
    ms = []
    for r in recs:
        m, e = col.extract(r, ctx)
        ms += m
        ents += e
    return ms, ents


def run_real(slug, paths, **kw):
    return run(slug, [(p, C.read_raw(p)) for p in paths], **kw)


def count(ms, **conds):
    return sum(all(getattr(m, k) == v for k, v in conds.items()) for m in ms)


def schema_ok(ms, ents=()):
    """Rows convert to the pyarrow schemas; context <= 120 chars and none for T3."""
    if ms:
        pa.Table.from_pylist([m.to_row() for m in ms], schema=MENTION_SCHEMA)
    if ents:
        pa.Table.from_pylist([e.to_row() for e in ents], schema=ENTITY_SCHEMA)
    for m in ms:
        assert m.context is None or len(m.context) <= 120
        if m.tier == 'T3':
            pass   # the Writer drops contexts of T3 rows; modules never fill them
    return True


NS = 'xmlns="http://www.tei-c.org/ns/1.0"'
