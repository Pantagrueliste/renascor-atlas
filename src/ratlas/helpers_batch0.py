"""Small helpers shared by the batch0 collection modules (additive; common.py is untouched)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional

from . import common as C
from .model import Entity, Mention, empty_auth

_NAME_CACHE: dict = {}


def fold(s: str) -> str:
    """Lower-case, accent-free, punctuation-free key for exact name matching."""
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def name_index(reg: C.Register, kind: Optional[str] = None) -> dict:
    """folded name/variant -> list of Entity (cached per register object and kind)."""
    key = (id(reg), kind, len(reg))
    idx = _NAME_CACHE.get(key)
    if idx is None:
        idx = {}
        for e in reg.entities.values():
            if kind and e.kind != kind:
                continue
            for n in [e.name, *e.variants]:
                f = fold(n)
                if f:
                    lst = idx.setdefault(f, [])
                    if e not in lst:
                        lst.append(e)
        _NAME_CACHE[key] = idx
    return idx


def unique_by_name(reg: C.Register, name: str, kind: str) -> Optional[Entity]:
    """The single entity of ``kind`` whose name or variant equals ``name`` (folded), else None."""
    lst = name_index(reg, kind).get(fold(name))
    return lst[0] if lst and len(lst) == 1 else None


def attach(m: Mention, ent: Entity) -> Mention:
    """Resolve mention ``m`` to ``ent`` (local_key, in_register, merged authority ids)."""
    m.local_key = ent.local_key
    m.in_register = True
    m.auth = C.merge_auth(m.auth or empty_auth(), ent.auth)
    return m


def first_year(s: Optional[str]) -> Optional[int]:
    m = re.search(r'(?<!\d)(\d{4})(?!\d)', s or '')
    return int(m.group(1)) if m else None


def text_of(el) -> str:
    return C.surface_text(el)


def iter_local(root, name: str) -> Iterable:
    for el in root.iter():
        if C.ln(el.tag) == name:
            yield el


def extract_paths(slug: str, paths: Iterable[str], with_registers: bool = True):
    """Test helper: run a collection module on a few files. Returns (mentions, entities, ctx).

    Reads the scan for the module's files (needed for register detection), calls prepare(), then extract() for
    the given raw-relative ``paths`` only (matched exactly against the scan records).
    """
    from .collections import get
    col = get(slug)
    ctx = col.make_context()
    files = col.iter_files(ctx, False)
    ctx.all_files = files
    reg_ents = col.prepare(ctx) if with_registers else []
    want = set(paths)
    mentions: list = []
    ents: list = list(reg_ents)
    seen = set()
    for rec in files:
        if rec.path in want:
            seen.add(rec.path)
            ms, es = col.extract(rec, ctx)
            mentions += ms
            ents += es
    missing = want - seen
    if missing:
        raise FileNotFoundError(f'not in scan: {sorted(missing)[:3]}')
    return mentions, ents, ctx


def make_hit(el, kind: str, tag: str, role: str, zone: str, refs: Optional[dict] = None, lang: Optional[str] = None,
             ancestors: tuple = (), context: Optional[str] = None) -> C.Hit:
    """Build a scanner Hit by hand (for names the generic scanner skips, e.g. an author given in respStmt)."""
    if refs is None:
        refs = {a: v.strip() for a in C.REF_ATTRS if (v := el.get(a)) and v.strip() and v.strip() != '#'}
    return C.Hit(kind=kind, tag=tag, name=C.ln(el.tag), surface=C.surface_text(el), zone=zone, role=role, refs=refs,
                 lang=lang, context=context, el=el, ancestors=ancestors)


def author_respstmt_hits(root, resp_words=('author',)) -> list:
    """Persons named in header respStmt whose <resp> says they are the author (e.g. 'attributed author')."""
    out = []
    for rs in iter_local(root, 'respStmt'):
        resp = ' '.join(C.surface_text(c).lower() for c in rs if C.ln(c.tag) == 'resp')
        if not any(w in resp for w in resp_words):
            continue
        for c in rs:
            if C.ln(c.tag) in ('name', 'persName'):
                out.append(make_hit(c, 'person', 'name@person' if C.ln(c.tag) == 'name' else 'persName', 'author',
                                    'hdr', ancestors=('TEI', 'teiHeader', 'fileDesc', 'titleStmt', 'respStmt')))
    return out


def loc_from_url(auth: dict) -> dict:
    """Turn 'url:https://lccn.loc.gov/n85355622' entries of auth.other into 'loc:n85355622'."""
    other = []
    for x in auth.get('other') or []:
        m = re.match(r'url:https?://lccn\.loc\.gov/([a-z]{1,2}\d+)', x)
        x = f'loc:{m.group(1)}' if m else x
        if x not in other:
            other.append(x)
    auth['other'] = other
    return auth


# ======================================================================================================================
# DraCor plays (dracor_eng, dracor_ita): cast lists as register + 'cast' mentions, authors with Wikidata, event dates
# ======================================================================================================================
from .collections.generic_tei import GenericTEI   # noqa: E402  (after the helpers above; no circular import at use)


class DracorTEI(GenericTEI):
    """Shared logic of the DraCor play corpora.

    * ``particDesc/listPerson`` (person / personGrp) is the cast: register entries are fictional or mythological
      characters (``note_type`` 'character'; an ``@ana`` Wikidata URL becomes auth.wikidata) and each gives one mention
      with role ``cast`` (zone hdr) resolved to its entry;
    * the author ``persName`` in ``titleStmt/author`` is role ``author`` and takes the sibling ``idno[@type=wikidata]``;
    * speaker attributions (``sp/@who``) and untyped ``<name>`` in verse are not entity tags and are not read;
    * dates: ``standOff/listEvent/event`` written (notBefore/notAfter or when) > premiere > print, source 'header'.
    """
    event_order = ('written', 'premiere', 'print')

    def is_register_file(self, rec, ctx):
        return False           # the cast register sits in each play's own header

    def doc_date(self, root, rec, ctx):
        ev = {}
        for e in iter_local(root, 'event'):
            t = (e.get('type') or '').lower()
            ys = [y for a in ('when', 'notBefore', 'notAfter') if (y := C._attr_year(e.get(a))) is not None]
            if ys:
                ev.setdefault(t, []).extend(ys)
        for t in self.event_order:
            if ev.get(t):
                return C.DocDate(min(ev[t]), max(ev[t]), 'header')
        return C.DocDate()

    def cast_hits(self, root, doc_lang):
        out = []
        for lp in iter_local(root, 'listPerson'):
            if not any(C.ln(a.tag) == 'particDesc' for a in lp.iterancestors()):
                continue
            for p in lp:
                if C.ln(p.tag) not in ('person', 'personGrp') or not p.get(C.XML_ID):
                    continue
                nm = next((c for c in p if C.ln(c.tag) in ('persName', 'name')), None)
                el = nm if nm is not None else p
                out.append(C.Hit(kind='person', tag='person@cast', name=C.ln(el.tag), surface=C.name_text(el),
                                 zone='hdr', role='cast', refs={'corresp': '#' + p.get(C.XML_ID)}, lang=doc_lang,
                                 context=None, el=el, ancestors=('TEI', 'teiHeader', 'profileDesc', 'particDesc',
                                                                 'listPerson')))
        return out

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        return hits + self.cast_hits(root, C.doc_language(root))

    def file_licence(self, root, rec):
        """(licence, tier) override for a file, or None."""
        return None

    def file_renascor_id(self, root, rec, ctx):
        return None

    def extract_root(self, root, rec, ctx):
        doc_id = C.doc_id_of(root, rec)
        regs, _ = C.parse_register(root, ctx, rec.path, doc_id=doc_id)
        els = {e.get(C.XML_ID): e for e in root.iter() if C.ln(e.tag) in ('person', 'personGrp') and e.get(C.XML_ID)}
        local = C.Register()
        ents = []
        lic = self.file_licence(root, rec)
        rid = self.file_renascor_id(root, rec, ctx)
        for ent, aliases in regs:
            el = els.get(ent.local_key.rsplit('#', 1)[-1])
            if el is not None and el.get('ana'):
                C.parse_auth([el.get('ana')], into=ent.auth)
            ent.note_type = 'character'
            if lic:
                ent.licence, ent.tier = lic
            if rid:
                ent.renascor_id = rid
            local.add(ent, aliases)
            ents.append(ent)
        hits = self.scan(root, rec, ctx, Counter())
        date = self.doc_date(root, rec, ctx)
        ms = C.build_mentions(root, rec, ctx, local=local, date=date, hits=hits, adjust=self.adjust)
        for m in ms:
            if lic:
                m.licence, m.tier = lic
            if rid:
                m.renascor_id = rid
        return ms, ents

    def adjust(self, m, hit):
        if m.role == 'author' and m.surface.strip('() .').lower() in ('anonimo', 'anonymous', 'anon', 'anonyme', 'anonym'):
            return None             # "(Anonimo)" is not a person
        if m.role == 'author' and m.kind == 'person':
            parent = hit.el.getparent()
            for i in (parent if parent is not None else []):
                if C.ln(i.tag) == 'idno' and (i.get('type') or '').lower() == 'wikidata':
                    C.parse_auth([C.idno_to_token(i)], into=m.auth)
        m.context = None        # header-only mentions: no running text to quote
        return m


def sourcedesc_year(root, lo: int = 1400, hi: int = 1800) -> 'C.DocDate':
    """Imprint year written in sourceDesc (e.g. <bibl>Printer, Place, 1566.</bibl>): the first year in [lo, hi].
    Source 'header'. Undated when there is none."""
    for el in root.iter():
        if C.ln(el.tag) == 'sourceDesc':
            ys = [int(y) for y in C._YEAR_RX.findall(C.surface_text(el)) if lo <= int(y) <= hi]
            if ys:
                return C.DocDate(ys[0], ys[0], 'header')
    return C.DocDate()
