"""The editions of the NER data: one edition = one collection module of the atlas (slug).

Files, licences and tag conventions come from the collection modules (``ratlas.collections``): ``COLLS``, ``LICENCE``, ``TIER`` and
``OPTIONS`` (``type_map``, ``note_zone``, ``div_zone``). The NER-only settings are in ``EXTRA``. London Lives and the Medici Archive Project
are never read (``common.is_forbidden``, and the explicit check of ``allowed_path``).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import common as C
from .tei import Options

# NER-only settings per edition. strip_lead: words dropped in front of a place span ('of Stantone' -> 'Stantone'); default_lang: see below.
EXTRA: dict[str, dict] = {
    'casebooks': {'strip_lead': ('of', 'at', 'in', 'from', 'near', 'late')},
    # default_lang: language given to the passages without any language evidence (lists of names) when the document has no other passage
    # not_test: never held out as a test edition (a register of names, not running prose: a test score on it would mostly measure list parsing)
    'hearth_tax': {'default_lang': 'en', 'not_test': True},      # English returns (the Latin headings carry xml:lang or are guessed)
}
# editions that carry the same texts under two slugs: they never fall on two sides of the train / test split
OVERLAP_GROUPS = [
    {'freem_bvh', 'epistemon_bvh', 'freem_other'},
    {'dracor_ita', 'biblioteca_italiana'},
    {'dta_1600', 'sandrart'},
    {'camena_cera', 'camena_historicapolitica', 'camena_poemata', 'camena_thesaurus'},
    {'gongora_epistolario', 'gongora_soledades', 'polemicagongorina_obvil'},
    {'selbstzeugnisse_hab_diarium', 'selbstzeugnisse_hab_sz2'},
    {'elan_crenum', 'elan_fnvr', 'elan_polygrenaille'},
]
TAGGED_KEYS = ('txt|persName', 'txt|placeName', 'txt|geogName', 'txt|settlement', 'txt|country', 'txt|region', 'txt|district',
               'txt|bloc', 'txt|rs', 'txt|name')
FORBIDDEN_SLUG_TOKENS = ('london_lives', 'londonlives', 'london-lives', 'medici_archive', 'medici-archive', 'mediciarchive')


def allowed_path(path: str) -> bool:
    low = path.lower()
    return not C.is_forbidden(path) and not any(t in low for t in FORBIDDEN_SLUG_TOKENS)


def overlap_group(slug: str) -> str:
    for g in OVERLAP_GROUPS:
        if slug in g:
            return '+'.join(sorted(g))
    return slug


@dataclass
class Edition:
    slug: str
    colls: list
    licence: str
    tier: str
    renascor_ids: list
    options: Options
    col: object = None
    ctx: object = None
    meta: dict = field(default_factory=dict)       # from edition_meta.tsv (languages, period, no_derivatives...)

    def files(self) -> list:
        """Non-duplicate XML files of the edition, sorted by path, without the stand-alone register files."""
        recs = [r for r in self.col.iter_files(self.ctx) if allowed_path(r.path)]
        isreg = getattr(self.col.extractor, 'is_register_file', None)
        if isreg is not None:
            try:
                recs = [r for r in recs if not isreg(r, self.ctx)]
            except Exception:       # noqa: BLE001  a module whose test needs more context: keep the files
                pass
        return sorted(recs, key=lambda r: r.path)


def load_meta(path: Optional[Path]) -> dict:
    out = {}
    if path and Path(path).exists():
        with open(path, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                out[row['slug']] = row
    return out


def get_edition(slug: str, meta: Optional[dict] = None) -> Edition:
    from ..collections import get
    col = get(slug)
    ctx = col.make_context()
    ctx.all_files = []
    opts = Options.from_context(ctx.options, ctx.type_map, EXTRA.get(slug))
    return Edition(slug=slug, colls=list(col.module.COLLS), licence=col.licence, tier=col.tier,
                   renascor_ids=list(getattr(col.module, 'RENASCOR_IDS', [])), options=opts, col=col, ctx=ctx,
                   meta=(meta or {}).get(slug, {}))


def all_slugs() -> list[str]:
    from ..collections import discover
    return sorted(discover())


def ner_data_root() -> Path:
    return Path(os.environ.get('RATLAS_NER') or (C.data_root() / 'ner'))
