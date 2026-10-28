"""Helpers of batch 4 (label 'batch4'): shared pieces of the batch-4 collection modules and a census tool.

Run a census (read-only, one process; on Hypatia through srun, never on the login node):
    python -I -m ratlas.helpers_batch4 census <selector>... [--limit N]
It prints, per entity-candidate tag, the @type/@role values, the share with a reference and the shapes of the
references (digits replaced by 9), so type typos and reference conventions can be found before writing a module.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

from . import common as C
from .collections.generic_tei import GenericTEI
from .model import Entity

def title_year_date(root, lo=1400, hi=1800):
    """DocDate from the year(s) written in the titleStmt titles (e.g. 'Edikt von Amboise (19. Maerz 1563)').

    Skips level="s" series titles and <note> children; source 'header'. Undated when no year is found."""
    hdr = next((e for e in root.iter() if C.ln(e.tag) == 'teiHeader'), None)
    if hdr is None:
        return C.DocDate()
    ts = next((e for e in hdr.iter() if C.ln(e.tag) == 'titleStmt'), None)
    if ts is None:
        return C.DocDate()
    for t in ts:
        if C.ln(t.tag) != 'title' or t.get('level') == 's':
            continue
        parts = []

        def rec(e):
            if e.text:
                parts.append(e.text)
            for c in e:
                if isinstance(c.tag, str) and C.ln(c.tag) not in ('note', 'ref'):
                    rec(c)
                if c.tail:
                    parts.append(c.tail)
        rec(t)
        ys = [int(y) for y in C._YEAR_RX.findall(C.norm(''.join(parts))) if lo <= int(y) <= hi]
        if ys:
            return C.DocDate(min(ys), max(ys), 'header')
    return C.DocDate()


def sourcedesc_year_date(root, lo=1400, hi=1800):
    """DocDate from the imprint year written in sourceDesc/bibl (CAMENA: 'Venedig: apud Juntas, 1640.'), else from
    the short title ('Imperiali, Giovanni: Musaeum historicum et physicum. - Venedig, 1640.'). Source 'header'."""
    hdr = next((e for e in root.iter() if C.ln(e.tag) == 'teiHeader'), None)
    if hdr is None:
        return C.DocDate()
    cands = []
    for e in hdr.iter():
        n = C.ln(e.tag)
        if n == 'sourceDesc':
            cands.append(C.surface_text(e))
    for e in hdr.iter():
        if C.ln(e.tag) == 'title' and e.get('type') == 'short':
            cands.append(C.surface_text(e))
    for txt in cands:
        ys = [int(y) for y in C._YEAR_RX.findall(txt) if lo <= int(y) <= hi]
        if ys:
            return C.DocDate(ys[0], ys[0], 'header')
    return C.DocDate()


def text_dates_doc_date(root, lo=1400, hi=1800):
    """DocDate (source 'text') from the @when/@notBefore/@notAfter years of <date> elements outside the header."""
    ys = []
    for el in root.iter():
        if C.ln(el.tag) != 'date':
            continue
        if any(C.ln(a.tag) == 'teiHeader' for a in el.iterancestors()):
            continue
        for a in ('when', 'notBefore', 'notAfter', 'from', 'to'):
            y = C._attr_year(el.get(a))
            if y and lo <= y <= hi:
                ys.append(y)
    return C.DocDate(min(ys), max(ys), 'text') if ys else C.DocDate()


class SharedRegisterTEI(GenericTEI):
    """GenericTEI for editions whose register sits INSIDE the document files but whose ids are edition-wide:
    prepare() reads the register of every file once into ctx.shared (keys '<slug>:<id>', first one wins, entities
    returned once); extract_root() then resolves against ctx.shared only (no per-file register, no duplicates).
    ``register_root(root)`` can narrow the part of the tree that holds the register (default: whole tree)."""

    def register_root(self, root):
        return root

    def prepare(self, ctx):
        out = []
        for rec in ctx.all_files:
            if not self.is_register_candidate(rec):
                continue
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for ent, aliases in C.parse_register(self.register_root(root), ctx, rec.path, doc_id=None)[0]:
                if ent.local_key in ctx.shared.entities:
                    continue
                ctx.shared.add(ent, aliases)
                out.append(ent)
        return out

    def is_register_candidate(self, rec):
        return True

    def extract_root(self, root, rec, ctx):
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root))


_CAND = ('rs', 'name', 'persName', 'placeName', 'orgName', 'geogName', 'settlement', 'country', 'region')


def ref_shape(v: str) -> str:
    v = re.sub(r'\d+', '9', v.strip())
    v = re.sub(r'[A-Za-z_À-ɏ]+', lambda m: m.group(0) if len(m.group(0)) <= 4 else 'w', v)
    return v[:40]


def census(selectors, limit=None):
    types = defaultdict(Counter)
    refs = defaultdict(Counter)
    nfiles = 0
    for rec in C.iter_collection_files(selectors):
        if limit and nfiles >= limit:
            break
        root = C.parse_xml(rec.read())
        if root is None:
            continue
        nfiles += 1
        for el in root.iter():
            n = C.ln(el.tag)
            if n not in _CAND:
                continue
            key = n
            if n in ('rs', 'name'):
                key = f"{n}@{(el.get('type') or el.get('role') or '-').lower()}"
            types[n][key] += 1
            for a in C.REF_ATTRS:
                v = el.get(a)
                if v:
                    refs[key][f'{a}:{ref_shape(v.split()[0])}'] += 1
            if not any(el.get(a) for a in C.REF_ATTRS):
                refs[key]['(none)'] += 1
    return {'files': nfiles,
            'types': {k: dict(v.most_common(40)) for k, v in types.items()},
            'refs': {k: dict(v.most_common(8)) for k, v in refs.items()}}


_INV_TAGS = {'persName', 'placeName', 'orgName', 'geogName', 'settlement', 'country', 'region', 'rs', 'name'}


def inventory(slug):
    """Independent count of candidate tags per file set of a module (outside register containers), by zone and
    'tag@type'; used to reconcile the extracted mentions with the source. Read-only, one process."""
    from .collections import get
    col = get(slug)
    ctx = col.make_context()
    files = col.iter_files(ctx)
    out = defaultdict(Counter)
    for rec in files:
        root = C.parse_xml(rec.read())
        if root is None:
            continue
        stack = []
        for ev, el in __import__('lxml.etree', fromlist=['x']).iterwalk(root, events=('start', 'end')):
            n = C.ln(el.tag)
            if n is None:
                continue
            if ev == 'end':
                stack.pop()
                continue
            stack.append(n)
            if n not in _INV_TAGS:
                continue
            if set(stack[:-1]) & C.REG_ANCESTORS:
                zone = 'reg'
            else:
                zone = 'hdr' if 'teiHeader' in stack else 'txt'
            key = n
            if n in ('rs', 'name'):
                key = f"{n}@{(el.get('type') or el.get('role') or '-').lower()}"
            out[zone][key] += 1
    return {z: dict(c.most_common()) for z, c in out.items()} | {'files': len(files)}


def main(argv):
    if len(argv) >= 2 and argv[0] == 'inventory':
        print(json.dumps({s: inventory(s) for s in argv[1:]}, ensure_ascii=False, indent=1))
        return 0
    if len(argv) >= 2 and argv[0] == 'census':
        lim = None
        sel = []
        i = 1
        while i < len(argv):
            if argv[i] == '--limit':
                lim = int(argv[i + 1])
                i += 2
            else:
                sel.append(argv[i])
                i += 1
        print(json.dumps(census(sel, lim), ensure_ascii=False, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
