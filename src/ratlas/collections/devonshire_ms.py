"""Devonshire Manuscript (BL Add. MS 17492), social edition TEI from github etcluvic/devonshire-ms (tar.gz).

199 poem files ('Devonshire Manuscript poems/') share an identical teiHeader with two listPerson blocks:
type="historical" (25 poets, scribes and owners with LOC and ODNB ids and dates) and type="modern" (17 researchers and
scholars). In the poems <name key="WYATT"> marks the attributed author, a scribal hand or a person named in a note
(374 tags). The 185 witness files carry no markup and are not read.
Special handling:
  * the register is read once from the first poem file (prepare) as a shared register 'devonshire_ms:<ID>'; only the
    25 historical persons become entities; LOC name-authority ids go to auth.other ('loc:n50019425'), ODNB article
    ids to auth.other ('odnb:30111'); role="poet|scribe poet|historical" -> note_type; dates from birth / death;
  * untyped <name> is a person: type="person" is set on the in-memory tree; type="author" -> role 'author';
  * name tags whose key is one of the 17 modern researchers/scholars (HB, NOTT, MUIR, AC ...: 11 tags) are dropped:
    they are citations of modern editors, not mentions of early modern persons;
  * header tags (one <rs type="country"> 'England', repeated in every file) are boilerplate: not emitted;
  * the per-file copies of the header register are NOT re-emitted as entities;
  * dating: the manuscript, c. 1532-1547 (default_years, date_source 'collection'); no per-poem dates.
Licence: T3 (LICENSE_TIERS 'Mixed'; headers: "available upon request for teaching and academic purposes only"; the
Wikibooks social edition is CC BY-SA 4.0). No context snippets are stored.
"""
from __future__ import annotations

import re

from .. import common as C
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'devonshire_ms'
COLLS = ['renascor_2026-10/devonshire-manuscript']
RENASCOR_IDS = ['devonshire-manuscript']
LICENCE = 'LicenseRef-Mixed'
TIER = 'T3'
DESCRIPTION = 'Devonshire Manuscript poems: name@key attributions resolved to the header register of 25 historical persons'
OPTIONS = {'default_years': (1532, 1547)}

_POEMS = 'Devonshire Manuscript poems/'


def _persons(root):
    out = []
    for lp in root.iter('{*}listPerson'):
        for p in lp.findall('{*}person'):
            out.append((lp.get('type') or '', p))
    return out


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if _POEMS in r.path]

    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        return C.DocDate(1532, 1547, 'collection')

    def prepare(self, ctx):
        recs = [r for r in ctx.all_files if _POEMS in r.path]
        ents, modern = [], set()
        for rec in recs:                                  # the header register is identical in all files: take the first
            root = C.parse_xml(rec.read())
            ps = _persons(root) if root is not None else []
            if not ps:
                continue
            for typ, p in ps:
                ident = p.get(C.XML_ID)
                if typ == 'modern':
                    modern.add(ident)
                    continue
                pn = next((c for c in p if C.ln(c.tag) == 'persName'), None)
                name = C.name_text(pn) if pn is not None else ident
                toks = []
                if pn is not None:
                    for r in pn.findall('{*}ref'):
                        v = C.norm(''.join(r.itertext()))
                        t = (r.get('type') or '').upper()
                        if t == 'LOC' and v:
                            toks.append('loc:' + v.replace(' ', ''))
                        elif t == 'DNB':
                            m = re.search(r'article/(\d+)', v)
                            if m:
                                toks.append('odnb:' + m.group(1))
                auth = C.parse_auth(toks)
                b, d = C.child(p, 'birth'), C.child(p, 'death')
                role = (p.get('role') or '').strip().lower().replace(' ', '+') or None
                e = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='person', name=name,
                           source_path=rec.path, renascor_id=ctx.renascor_id, birth=C._date_string(b),
                           death=C._date_string(d), auth=auth, note_type=role, licence=ctx.licence, tier=ctx.tier)
                ctx.shared.add(e, [ident])
                ents.append(e)
            break
        ctx.options['_modern'] = modern
        ctx.options['_register_paths'] = set()
        return ents

    def scan(self, root, rec, ctx, counters):
        for el in root.iter('{*}name'):
            if not any(C.ln(a.tag) == 'teiHeader' for a in el.iterancestors()):
                if el.get('type') == 'author':
                    el.set('role', 'author_attribution')
                el.set('type', 'person')
        hits = super().scan(root, rec, ctx, counters)
        modern = ctx.options.get('_modern', ())
        out = []
        for h in hits:
            if h.zone == 'hdr':           # identical boilerplate of the shared header (rs country England ...)
                continue
            if (h.refs.get('key') or '').strip() in modern:
                continue
            if h.zone == 'txt' and h.el.get('role') == 'author_attribution':
                h.role = 'author'
            out.append(h)
        return out

    def extract_root(self, root, rec, ctx):
        # no per-file register: the header list is the shared one loaded by prepare()
        hits = self.scan(root, rec, ctx, __import__('collections').Counter())
        return C.build_mentions(root, rec, ctx, date=self.doc_date(root, rec, ctx), hits=hits), []


EXTRACTOR = Extractor()
