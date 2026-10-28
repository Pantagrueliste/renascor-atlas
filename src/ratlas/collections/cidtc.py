"""CIDTC: Johannes Dantiscus' Texts and Correspondence (Univ. of Warsaw, Laboratory for Source Editing), 7,096 letters.

Licence: LicenseRef-AllRightsReserved, tier T3 (LICENSE_TIERS.tsv: CIDTC, redistribution not permitted). Extracted for
internal work only; no context snippets are stored.

Conventions handled here:
  * every mention is an opaque numeric key: text <name|rs type="person|place" key="N"> and header
    <persName ref="PeN" role="sender|addressee" ana="Surname, Forename"> / <placeName ref="PlN">. Persons and places
    have separate number spaces and the text keys use the same numbers as the header refs, so
    local_key = 'cidtc:pe<N>' / 'cidtc:pl<N>' (in_register False: the edition's person and place registers are not in
    the harvest, hence no entities.parquet rows);
  * header persName of correspAction both sit under correspAction[@type=sent]; the role comes from @role
    (sender / addressee), the place under correspAction is 'sent_from'; the titleStmt <persName type=sender|addressee>
    only repeats them and is skipped;
  * empty header persName carry the name in @ana; it is used as surface;
  * a plural key (key="24 12 7", 'maiestates regias') gives one mention row per referenced id (same surface), so
    rows can exceed the number of tags; ref_raw keeps the full key string;
  * rs marks epithets and offices ('regiae maiestatis'): kept as mentions of the person who holds the key;
  * dating: correspAction/date (letters 1511-1548; 4 letters without date).
No authority ids exist in the files.

v2 (issue 217; the owner decided to leave CIDTC as it is: no request to the editors, no new harvest, only a better reading of the files we
hold): the edition's registers are still absent, but ``prepare()`` builds a local register from the files themselves, so that an entity is
named by a NAME and not by its commonest word form (pe24 was 'maiestatem regiam', pe30 'caesarem', 5,115 and 3,202 mentions): the name of
a person is the commonest header ``@ana`` ('Surname, Forename') of its id; failing that, the commonest capitalised text surface; failing
that, the commonest surface with ``note_type`` 'epithet_only' (a title, an office: not a name). Places take the commonest header
<placeName> text, else the commonest capitalised surface. Variants are the other surfaces. 921 pseudo-places and the epithet-only
persons remain visible through note_type.
"""
from __future__ import annotations

import dataclasses
import re
from collections import Counter

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'cidtc'
COLLS = ['cidtc']
RENASCOR_IDS = ['cidtc']
LICENCE = 'LicenseRef-AllRightsReserved'
TIER = 'T3'
DESCRIPTION = 'Dantiscus letters 1511-1548, numeric person/place keys, no registers'
OPTIONS = {}

_NUM = re.compile(r'(?:Pe|Pl)?(\d+)', re.I)


def _capitalised(s: str) -> bool:
    return bool(s) and s[:1].isupper()


class Extractor(GenericTEI):
    def prepare(self, ctx):
        from collections import defaultdict
        from ..model import Entity
        hdr = defaultdict(Counter)
        txt = defaultdict(Counter)
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for h in C.scan_mentions(root, type_map=ctx.type_map, skip_ancestors=('titleStmt',)):
                if h.kind not in ('person', 'place'):
                    continue
                pre = 'pe' if h.kind == 'person' else 'pl'
                toks = (h.refs.get('ref') or h.refs.get('key') or '').split()
                if not toks or not all(_NUM.fullmatch(t) for t in toks):
                    continue
                if len(toks) == 1:
                    k = f'{pre}{int(_NUM.fullmatch(toks[0]).group(1))}'
                    if h.zone == 'hdr':
                        s = C.norm(h.el.get('ana')) if h.kind == 'person' and h.el.get('ana') else h.surface
                        if s:
                            hdr[k][s] += 1
                    elif h.surface:
                        txt[k][h.surface] += 1
        out = []
        for k in sorted(set(hdr) | set(txt), key=lambda x: (x[:2], int(x[2:]))):
            kind = 'person' if k.startswith('pe') else 'place'
            if hdr.get(k):
                name, nt = hdr[k].most_common(1)[0][0], kind
            else:
                cap = [(s, n) for s, n in txt[k].most_common() if _capitalised(s)]
                if cap:
                    name, nt = cap[0][0], kind
                else:
                    name, nt = txt[k].most_common(1)[0][0], 'epithet_only'
            variants = [s for s, _ in (hdr.get(k, Counter()) + txt.get(k, Counter())).most_common(8) if s != name][:7]
            out.append(Entity(local_key=f'{SLUG}:{k}', coll_slug=ctx.slug, kind=kind, name=name, variants=variants,
                              source_path='(header @ana and text forms)', renascor_id=ctx.renascor_id, note_type=nt,
                              licence=ctx.licence, tier=ctx.tier))
        ctx.shared.entities.update({e.local_key: e for e in out})
        return out

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=False, doc_lang=C.doc_language(root),
                               counters=counters, skip_ancestors=('titleStmt',))

    def adjust(self, m, hit):
        el = hit.el
        if m.kind == 'person' and m.zone == 'hdr':
            r = (el.get('role') or '').lower()
            if r in ('sender', 'addressee'):
                m.role = r
            if not m.surface:
                m.surface = C.norm(el.get('ana'))
        ids = self._ids(m)
        if ids:
            m.local_key = ids[0]
            m.in_register = ids[0] in self._known
        return m

    @staticmethod
    def _ids(m):
        if not m.ref_raw or m.kind not in ('person', 'place'):
            return []
        toks = m.ref_raw.split()
        if not all(_NUM.fullmatch(t) for t in toks):
            return []
        pre = 'pe' if m.kind == 'person' else 'pl'
        return [f'{SLUG}:{pre}{int(_NUM.fullmatch(t).group(1))}' for t in dict.fromkeys(toks)]

    def extract_root(self, root, rec, ctx):
        self._known = ctx.shared.entities
        ms, es = super().extract_root(root, rec, ctx)
        out = []
        for m in ms:
            ids = self._ids(m)
            out.append(m)
            for k in ids[1:]:       # plural reference (key="24 12 7"): one row per referenced person
                out.append(dataclasses.replace(m, local_key=k, in_register=k in self._known))
        return out, es


EXTRACTOR = Extractor()
