"""Generic extractor for well-formed TEI with standard registers (listPerson/listPlace/listOrg).

Specific collection modules either use it as is (constants only) or subclass it and override a hook:

  is_register_file(rec, ctx)      which files are stand-alone registers (loaded once by prepare())
  parse_register_file(rec, ctx, root)   -> list[(Entity, aliases)]   (default: standard TEI register parsing)
  doc_date(root, rec, ctx)        -> DocDate   (default: header dates, filename if ctx.filename_dates, collection)
  scan(root, rec, ctx, counters)  -> list[Hit] (default: common.scan_mentions)
  adjust(mention, hit)            -> Mention | None   (per-mention fix-ups: ids in odd attributes, roles, drops)
  extract_root(root, rec, ctx)    -> (mentions, entities) for one parsed document

Every file of the collection is passed to extract(); register files return ([], []) because their entities were
already produced by prepare().
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .. import common as C
from ..model import Entity, Mention

_TAGGED = ('txt|persName', 'txt|placeName', 'txt|orgName', 'txt|geogName', 'txt|rs', 'txt|name', 'txt|settlement')
_REG = ('reg|person', 'reg|place', 'reg|org')


class GenericTEI:
    register_min = 5          # a file with >= this many register entries and no tagged text is a register file

    # ---- file selection -------------------------------------------------------------------------------------------
    def iter_files(self, ctx: C.Context, include_dups: bool = False) -> list[C.FileRecord]:
        return C.iter_collection_files(ctx.options['colls'], include_dups)

    def is_register_file(self, rec: C.FileRecord, ctx: C.Context) -> bool:
        return rec.count(*_REG) >= self.register_min and rec.count(*_TAGGED) == 0

    # ---- registers ------------------------------------------------------------------------------------------------
    def parse_register_file(self, rec: C.FileRecord, ctx: C.Context, root) -> list[tuple[Entity, list]]:
        ents, _ = C.parse_register(root, ctx, rec.path, doc_id=None)
        return ents

    def prepare(self, ctx: C.Context) -> list[Entity]:
        """Load the stand-alone register files into ctx.shared and return their entities."""
        out: list[Entity] = []
        reg_paths = set()
        for rec in ctx.all_files:
            if not self.is_register_file(rec, ctx):
                continue
            reg_paths.add(rec.path)
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for ent, aliases in self.parse_register_file(rec, ctx, root):
                ctx.shared.add(ent, aliases)
                out.append(ent)
        ctx.options['_register_paths'] = reg_paths
        return out

    # ---- documents ------------------------------------------------------------------------------------------------
    def doc_date(self, root, rec: C.FileRecord, ctx: C.Context) -> C.DocDate:
        return C.resolve_doc_date(root, rec, filename_dates=ctx.filename_dates, default_years=ctx.default_years)

    def scan(self, root, rec: C.FileRecord, ctx: C.Context, counters: Counter) -> list[C.Hit]:
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root), counters=counters)

    def extract_root(self, root, rec: C.FileRecord, ctx: C.Context) -> tuple[list[Mention], list[Entity]]:
        doc_id = C.doc_id_of(root, rec)
        local = C.Register()
        ents: list[Entity] = []
        if not rec.c or rec.count(*_REG):   # no scan info (tests): always look
            for ent, aliases in C.parse_register(root, ctx, rec.path, doc_id=doc_id)[0]:
                local.add(ent, aliases)
                ents.append(ent)
        hits = self.scan(root, rec, ctx, Counter())
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=local, date=date, hits=hits, adjust=self.adjust), ents

    def adjust(self, m: Mention, hit: C.Hit) -> Optional[Mention]:
        """Per-mention hook: edit and return the mention (or None to drop it). Default: unchanged."""
        return m

    def extract(self, rec: C.FileRecord, ctx: C.Context) -> tuple[list[Mention], list[Entity]]:
        if rec.path in ctx.options.get('_register_paths', ()):
            return [], []
        root = C.parse_xml(rec.read())
        if root is None:
            raise ValueError('unparseable XML')
        return self.extract_root(root, rec, ctx)
