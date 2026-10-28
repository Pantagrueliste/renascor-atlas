"""Registry of collection modules.

A module ``src/ratlas/collections/<slug>.py`` defines (module-level):

    SLUG          str, [a-z0-9_]+ and equal to the module file name
    COLLS         list[str]  scan.jsonl 'coll' values and/or raw path prefixes the module covers
    RENASCOR_IDS  list[str]  catalogue ids (data/entries/<id>.json of the Renascor repo)
    LICENCE       str        SPDX-ish licence of the data
    TIER          'T1' | 'T2' | 'T3'
  optional:
    OPTIONS       dict       Context fields: default_years=(from,to), filename_dates=bool, opaque_keys=bool,
                             type_map={...}; any other key goes to ctx.options
    EXTRACTOR     a GenericTEI instance (default: GenericTEI())
    iter_files(ctx, include_dups=False), prepare(ctx) -> [Entity], extract(rec, ctx) -> (mentions, entities)
                  module-level functions override the extractor's methods
    DESCRIPTION   str
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Optional

from .. import common as C
from .generic_tei import GenericTEI

_SKIP = {'generic_tei'}


@dataclass
class Collection:
    """Adapter around a collection module (the runner only talks to this)."""
    module: object

    @property
    def slug(self) -> str:
        return self.module.SLUG

    @property
    def tier(self) -> str:
        return self.module.TIER

    @property
    def licence(self) -> str:
        return self.module.LICENCE

    def make_context(self) -> C.Context:
        m = self.module
        opts = dict(getattr(m, 'OPTIONS', {}) or {})
        ctx = C.Context(
            slug=m.SLUG, licence=m.LICENCE, tier=m.TIER, renascor_ids=list(getattr(m, 'RENASCOR_IDS', [])),
            default_years=opts.pop('default_years', None), filename_dates=opts.pop('filename_dates', False),
            opaque_keys=opts.pop('opaque_keys', False), type_map=opts.pop('type_map', {}), options=opts)
        ctx.options['colls'] = list(m.COLLS)
        return ctx

    @property
    def extractor(self) -> GenericTEI:
        return getattr(self.module, 'EXTRACTOR', None) or GenericTEI()

    def iter_files(self, ctx: C.Context, include_dups: bool = False):
        f = getattr(self.module, 'iter_files', None)
        return f(ctx, include_dups) if f else self.extractor.iter_files(ctx, include_dups)

    def prepare(self, ctx: C.Context):
        f = getattr(self.module, 'prepare', None)
        return (f(ctx) if f else self.extractor.prepare(ctx)) or []

    def extract(self, rec: C.FileRecord, ctx: C.Context):
        f = getattr(self.module, 'extract', None)
        return f(rec, ctx) if f else self.extractor.extract(rec, ctx)


def _validate(mod, name: str):
    for attr in ('SLUG', 'COLLS', 'RENASCOR_IDS', 'LICENCE', 'TIER'):
        if not hasattr(mod, attr):
            raise ValueError(f'collection module {name}: missing {attr}')
    if mod.SLUG != name or not re.fullmatch(r'[a-z0-9_]+', mod.SLUG):
        raise ValueError(f'collection module {name}: SLUG must be [a-z0-9_]+ and equal to the file name')
    if mod.TIER not in ('T1', 'T2', 'T3'):
        raise ValueError(f'collection module {name}: TIER must be T1, T2 or T3')
    for s in mod.COLLS:
        if C.is_forbidden(s):
            raise ValueError(f'collection module {name}: {s} is excluded from the atlas')


def discover() -> dict[str, Collection]:
    out: dict[str, Collection] = {}
    for mi in pkgutil.iter_modules(__path__):
        if mi.name.startswith('_') or mi.name in _SKIP:
            continue
        mod = importlib.import_module(f'{__name__}.{mi.name}')
        if not hasattr(mod, 'SLUG'):
            continue
        _validate(mod, mi.name)
        out[mod.SLUG] = Collection(mod)
    return out


def get(slug: str) -> Collection:
    reg = discover()
    if slug not in reg:
        raise KeyError(f'unknown slug {slug!r}; known: {", ".join(sorted(reg)) or "(none)"}')
    return reg[slug]
