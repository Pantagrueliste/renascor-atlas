"""CKCC Epistolarium (Circulation of Knowledge, DANS deposit 10.17026/DANS-XFD-N8Y5): 17th-c. Dutch-Republic scholars.

Two forms of the same corpus:
  (a) 64 DBNL TEI.2 volume files (data/corpus/data/<corpus>/<corpus><vol>_01.xml; Grotius 18 vols, C. Huygens 6,
      Chr. Huygens 22, Leeuwenhoek 18), no namespace, ids in @id: every letter is a <div type="chapter"
      subtype="artifact"> with <interpGrp><interp type="id|date|sender|senderloc|recipient|recipientloc|language">;
      <persName key>, <placeName key>, <orgName>, <geogName> mark the mentions in text and in the editors' notes;
  (b) 6 pseudo-XML .txt corpora (Barlaeus, Beeckman, Descartes, Nieuwerop/Nieuwentijt, Swammerdam, Leeuwenhoek-corpus),
      one TEI-like chunk per letter after a line '### <id>.xml', with <meta type="id|date|sender|senderloc|recipient|
      recipientloc|language"> and the same persName / placeName tags. They are not in scan.jsonl; they are added here.
Registers (data/corpus/meta): persons/CKCC-persons-a..z.xml (id = CKCC key like 'bacon.francis.1561-1626', name parts,
birth / death / floruit, occupation, xref CEN ppn / CERL / bioport -> auth.other), persons/CKCC-organizations.xml
(kind org), places/CKCC-places-a..z.xml (settlement, region, country ref, geo lat lon, xref CNW), CKCC-countries.xml and
CKCC-regions.xml (country / region entries, regions with geo). Ids are in @id (not xml:id): parsed by this module.
Mentions:
  * tagged persName / placeName / orgName / geogName (zone 'txt', role 'mention'), resolved by @key against the shared
    register; a persName that resolves to an organization entry becomes kind 'org';
  * letter metadata (zone 'hdr', tags 'interp@sender' / 'meta@sender' ...): one 'sender', 'addressee', 'sent_from',
    'sent_to' mention per letter from interp / meta (the ids are CKCC keys; '?' is skipped). These rows are EXTRA to the
    tagged counts of the inventory (they are not tags in the text);
  * keys of the old coding scheme ('amste001', 'swam001', 'sten901'; txt corpora and Chr. Huygens volumes) have no
    register: opaque local keys '<slug>:<code>' (in_register False; also 'luik_001', 'viry_901'); '@Surname, Forename'
    handles and '?' stay unresolved;
  * CNW ids of the places ('amsterdam.nl', xref type CNW) are accepted as aliases of the register id;
  * place keys '<name>.<country>' that are not register ids but equal the slug of a register place name (primary or
    variant, unique in the country) are resolved ('venezia.ita' -> venetie.ita, 'milano.ita' -> milaan.ita: 910 tags);
  * not read: <name type="code"> (2,192 cipher / code names of Grotius's letters, e.g. 'Numerianus' = Amsterdam) and the
    821 untyped <rs> epithets ('Francen ambassadeur').
Dating: per letter from interp / meta date (years; date_source 'letter'); chapters that are not letters (introductions,
indexes, notes) take no date unless the volume has dated letters, then the volume span (date_source 'collection');
volumes without letters (Chr. Huygens science tomes, Leeuwenhoek '_lp') are undated. Out-of-period letters (Leeuwenhoek
1701-1721, a few early / late Huygens) are kept and flagged in_period False. Tags inside the 'index' chapters
(name indexes of the printed editions) are not emitted (apparatus).
v2 (audit points 4 and 11):
  * letters shared by two corpora are counted once: the 'id' of a letter is local to its volume (3598 is a different letter in
    Grotius and in C. Huygens), so a duplicate is a letter with the same (date, sender, recipient) in two corpora
    (the Grotius / Huygens / Descartes / Barlaeus corpora overlap: ~280 letters, 219 of them with both persons known). The
    letter of the corpus that comes first in ``_CORPUS_PRIORITY`` is the canonical one; the other copies keep their rows with
    ``excluded_reason`` 'duplicate_letter' (left out of the default counts);
  * old-scheme place codes ('denha004', 'amste001', 'leide001', 'londe002', 'stock...'; 6,530 mentions, mostly the letter header) are
    resolved to the register place when their letter stem is the beginning of exactly one place name of the register
    ('denha' -> Den Haag, 'amste' -> Amsterdam); otherwise they stay opaque keys;
  * a persName / placeName tag that points to a register entry of another kind keeps its own kind and inherits no ids.
Licence: CC-BY-4.0 (DANS deposit; LICENSE_TIERS CKCC_Epistolarium), T1.
Overlap: the Grotius and C. Huygens volumes duplicate letters of the separate catalogue entries
grotius-correspondence-huygens and briefwisseling-constantijn-huygens (other encodings); global identity is a later stage.
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

from .. import common as C
from ..helpers_batch3 import enclosing, set_years
from ..model import Entity, Mention, best_year, period_overlap
from .generic_tei import GenericTEI

SLUG = 'ckcc_epistolarium'
_BASE = 'originals_2026-10-15/repository-bulk-downloads/CKCC_Epistolarium/'
COLLS = [_BASE.rstrip('/')]
RENASCOR_IDS = ['ckcc-epistolarium']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'CKCC letters of 9 Dutch-Republic scholars: keyed persName/placeName, register with coordinates'
OPTIONS = {}

_DATA = _BASE + 'data/corpus/data/'
_META = _BASE + 'data/corpus/meta/'
_OLD = re.compile(r'^[a-z]{3,7}_?\d{3}$')
# old-scheme place code stem -> register id (audit: 'resolve the codes with a manual table'; the rest go by name prefix)
_CODE_TABLE = {'parij': 'paris.fra', 'londe': 'london.gbr', 'luik': 'luik.bel', 'flore': 'florence.ita',
               'kopen': 'kopenhagen.dnk', 'egmon': 'egmond.nld'}
_CORPUS_PRIORITY = ['groo001', 'huyg001', 'desc004', 'barl001', 'beec002', 'nier005', 'swam001', 'leeu027', 'huyg003']
_META_RX = re.compile(r'<(?:interp|meta)\b[^>]*?type="(id|date|sender|recipient)"[^>]*?value="([^"]*)"')


def _corpus_of(rec) -> str:
    p = rec.path.split('/data/corpus/data/')[-1]
    return p.split('/')[0][:7]


def _letters_of(rec):
    """(doc_id, date, sender, recipient) of every letter of a file (xml volume or txt corpus)."""
    text = rec.read().decode('utf-8', errors='replace')
    corpus = _corpus_of(rec)
    if rec.path.endswith('.txt'):
        blocks = re.split(r'(?m)^### \S+\s*$', text)[1:]
    else:
        blocks = re.findall(r'<interpGrp>(.*?)</interpGrp>', text, re.S)
    for blk in blocks:
        d = {}
        for k, v in _META_RX.findall(blk):
            d.setdefault(k, v)
        if d.get('id') is not None:
            yield f"{corpus}:{d['id']}", d.get('date'), d.get('sender'), d.get('recipient')



# ---------------------------------------------------------------------------------------------------------------
# registers
# ---------------------------------------------------------------------------------------------------------------
def _when(el):
    if el is None:
        return None
    if el.get('when'):
        return el.get('when')
    a, b = el.get('notBefore'), el.get('notAfter')
    if a or b:
        return f'{a or ""}/{b or ""}'
    return C.norm(''.join(el.itertext())) or None


def _person_name(p):
    pn = C.child(p, 'persName')
    if pn is None:
        return p.get('id') or '', []
    parts = [C.surface_text(c) for c in pn if C.ln(c.tag) in ('forename', 'nameLink', 'surname', 'roleName')]
    base = C.norm(' '.join(x for x in parts if x)) or C.surface_text(pn)
    adds = [C.surface_text(c) for c in pn if C.ln(c.tag) == 'addName']
    return base, [f'{base} {a}' for a in adds if a]


def _parse_register(root, ctx, source_path, kind_of_file):
    out = []
    for el in root.iter():
        n = C.ln(el.tag)
        if n not in ('person', 'place'):
            continue
        ident = el.get('id') or el.get(C.XML_ID)
        if not ident:
            continue
        auth_toks, variants = [], []
        for x in el:
            if C.ln(x.tag) != 'xref':
                continue
            t, st, v = (x.get('type') or ''), (x.get('subtype') or ''), C.norm(''.join(x.itertext()))
            if not v or t == 'CKCC':
                continue
            if t == 'CEN' and st == 'name':
                variants.append(v)
            elif t == 'CEN' and st == 'ppn':
                auth_toks.append('ppn:' + v)
            elif t == 'CERL':
                auth_toks.append('cerl:' + v if not v.startswith('http') else v)
            elif t == 'CNW':
                auth_toks.append('cnw:' + v)      # Dutch gazetteer id: kept as a plain string in auth.other
            else:
                auth_toks.append(f'{t.lower()}:{v}')
        auth = C.empty_auth()
        for tok in auth_toks:
            sch, _, val = tok.partition(':')
            if tok.startswith('http'):
                C.parse_auth([tok], into=auth)
            elif tok not in auth['other']:
                auth['other'].append(tok)
        if n == 'person':
            name, addv = _person_name(el)
            variants = addv + [v for v in variants if v != name]
            sex = C.child(el, 'sex')
            is_org = kind_of_file == 'org'
            birth = death = None
            b, d, f = C.child(el, 'birth'), C.child(el, 'death'), C.child(el, 'floruit')
            birth, death = _when(b), _when(d)
            if f is not None and not birth and not death:
                birth = 'fl. ' + (_when(f) or '')
            ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug,
                         kind='org' if is_org else 'person', name=name, source_path=source_path,
                         renascor_id=ctx.renascor_id, variants=variants, birth=birth, death=death, auth=auth,
                         note_type='organization' if is_org else 'person', licence=ctx.licence, tier=ctx.tier)
        else:
            pls = [c for c in el if C.ln(c.tag) == 'placeName']
            names = []
            for pl in pls:
                bits = [C.surface_text(c) for c in pl if C.ln(c.tag) in ('settlement', 'region', 'country') and C.surface_text(c)]
                if bits:
                    names.append(bits[0] if kind_of_file == 'city' or len(bits) == 1 else ', '.join(bits))
            name = names[0] if names else ident
            variants = [v for v in dict.fromkeys(names[1:] + variants) if v != name]
            g = next((x for x in el.iter() if C.ln(x.tag) == 'geo'), None)
            ll = C._parse_geo(C.surface_text(g)) if g is not None else None
            ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='place', name=name,
                         source_path=source_path, renascor_id=ctx.renascor_id, variants=variants,
                         lat=ll[0] if ll else None, lon=ll[1] if ll else None, coord_source='tei:geo' if ll else None,
                         auth=auth, note_type=kind_of_file if kind_of_file in ('country', 'region') else 'place',
                         licence=ctx.licence, tier=ctx.tier)
        out.append((ent, [ident] + [t[4:] for t in auth_toks if t.startswith('cnw:')]))   # CNW ids ('amsterdam.nl') are used as keys too
    return out


def _slug(s: str) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')


def _years(value):
    ys = [int(y) for y in re.findall(r'(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)', value or '')]
    ys = [y for y in ys if 1000 <= y <= 2100]
    return (min(ys), max(ys)) if ys else None


class Extractor(GenericTEI):
    # ---- files ------------------------------------------------------------------------------------------------------
    def iter_files(self, ctx, include_dups=False):
        recs = []
        for r in super().iter_files(ctx, include_dups):
            if r.path.startswith(_META + 'persons/') or r.path.startswith(_META + 'places/'):
                recs.append(r)
            elif r.path.startswith(_DATA) and r.path.endswith('.xml'):
                recs.append(r)
        root = C.raw_root() / _DATA
        if root.is_dir():
            for d in sorted(os.listdir(root)):
                for n in sorted(os.listdir(root / d)):
                    if n.endswith('.txt') and self._is_corpus(root / d / n):
                        recs.append(C.FileRecord(path=f'{_DATA}{d}/{n}', coll=C.raw_root().name))
        return sorted(recs, key=lambda r: r.path)

    @staticmethod
    def _is_corpus(p):
        try:
            with open(p, 'rb') as fh:
                head = fh.read(200000)
        except OSError:
            return False
        return b'\n### ' in head or head.startswith(b'### ') or b'\n### ' in head.replace(b'\r', b'')

    def is_register_file(self, rec, ctx):
        return rec.path.startswith(_META)

    def parse_register_file(self, rec, ctx, root):
        n = rec.name
        kind = 'org' if 'organizations' in n else 'country' if 'countries' in n else 'region' if 'regions' in n \
            else 'city' if '/places/' in rec.path else 'person'
        return _parse_register(root, ctx, rec.path, kind)

    def prepare(self, ctx):
        ents = super().prepare(ctx)
        ctx.options['_org_keys'] = {e.local_key for e in ents if e.kind == 'org'}
        # Text keys in the form '<name-slug>.<country>' that are not register ids ('venezia.ita', 'milano.ita': the
        # register has 'venetie.ita', 'milaan.ita') resolve through the place names of the register (primary name or
        # variant) when that name is unique within the country; real ids always win.
        cands: dict[str, set] = {}
        for e in ents:
            if e.kind != 'place' or e.note_type != 'place':
                continue
            cc = e.local_key.rsplit('.', 1)[-1]
            for nm in [e.name] + list(e.variants):
                cands.setdefault(f'{_slug(nm)}.{cc}', set()).add(e.local_key)
        n = 0
        for alias, keys in cands.items():
            if len(keys) == 1 and alias not in ctx.shared.index:
                ctx.shared.index[alias] = next(iter(keys))
                n += 1
        ctx.options['_alias_places'] = n
        # old-scheme codes: the letters of the code are the beginning of exactly one place name of the register
        stems: dict[str, set] = {}
        for e in ents:
            if e.kind != 'place' or e.note_type != 'place':
                continue
            for nm in [e.name]:
                f = re.sub(r'[^a-z]', '', unicodedata.normalize('NFKD', nm.casefold()).encode('ascii', 'ignore').decode())
                for k in range(3, min(len(f), 7) + 1):
                    stems.setdefault(f[:k], set()).add(e.local_key)
        by_key = {e.local_key: e for e in ents if e.kind == 'place'}
        code_stems = {}
        for s, keys in stems.items():
            if len({by_key[k].name.casefold() for k in keys}) == 1:      # one name; sub-locations (Clingendael, Den Haag) share it
                code_stems[s] = min(keys, key=lambda k: (k.count('.'), len(k), k))
        for stem, rid in _CODE_TABLE.items():          # Dutch stems whose register name is French / English / Danish
            k = C.make_key(ctx.slug, rid)
            if k in by_key:
                code_stems[stem] = k
        ctx.options['_code_stems'] = code_stems
        # letters in two corpora: same date + sender + recipient (the letter id is local to a volume)
        groups: dict[tuple, list] = {}
        for rec in ctx.all_files:
            if rec.path in ctx.options.get('_register_paths', ()):
                continue
            try:
                for doc_id, date, snd, rcp in _letters_of(rec):
                    if date and snd and snd != '?' and rcp and rcp != '?':
                        groups.setdefault((date, snd, rcp), []).append(doc_id)
            except Exception:      # noqa: BLE001  a file that does not parse is reported by extract()
                continue
        losers = set()
        for ids in groups.values():
            corpora = {i.split(':', 1)[0] for i in ids}
            if len(corpora) < 2:
                continue
            win = min(corpora, key=lambda c: _CORPUS_PRIORITY.index(c) if c in _CORPUS_PRIORITY else 99)
            losers.update(i for i in ids if i.split(':', 1)[0] != win)
        ctx.options['_dup_losers'] = losers
        return ents

    # ---- letters ------------------------------------------------------------------------------------------------------
    def extract(self, rec, ctx):
        self._ctx = ctx
        self._dup_losers = ctx.options.get('_dup_losers', set())
        if rec.path in ctx.options.get('_register_paths', ()):
            return [], []
        if rec.path.endswith('.txt'):
            return self._extract_txt(rec, ctx)
        root = C.parse_xml(rec.read())
        if root is None:
            raise ValueError('unparseable XML')
        return self.extract_root(root, rec, ctx)

    def _meta_mentions(self, meta, doc_id, rec, ctx, years, src, tagname):
        out = []
        for fld, role, kind in (('sender', 'sender', 'person'), ('recipient', 'addressee', 'person'),
                                ('senderloc', 'sent_from', 'place'), ('recipientloc', 'sent_to', 'place')):
            v = (meta.get(fld) or '').strip()
            if not v or v == '?':
                continue
            ent = ctx.shared.lookup(v)
            lk, inreg, surface, auth = None, None, v, C.empty_auth()
            if ent is not None:
                lk, inreg, surface, auth = ent.local_key, True, ent.name, C.merge_auth(C.empty_auth(), ent.auth)
                if ent.kind == 'org':
                    kind = 'org'
            elif _OLD.match(v):
                lk, inreg = C.make_key(SLUG, v), False
                ent = self._by_code(ctx, v) if kind == 'place' else None
                if ent is not None:
                    lk, inreg, surface, auth = ent.local_key, True, ent.name, C.merge_auth(C.empty_auth(), ent.auth)
            out.append(Mention(
                coll_slug=ctx.slug, renascor_id=ctx.renascor_id, source_path=rec.path, doc_id=doc_id,
                doc_year=best_year(*years) if years else None, doc_year_from=years[0] if years else None,
                doc_year_to=years[1] if years else None, date_source=src if years else None,
                in_period=period_overlap(*years) if years else None, kind=kind, tag=f'{tagname}@{fld}', role=role,
                zone='hdr', surface=surface, ref_raw=v, local_key=lk, in_register=inreg, auth=auth, context=None,
                lang=meta.get('language'), licence=ctx.licence, tier=ctx.tier,
                excluded_reason='duplicate_letter' if doc_id in ctx.options.get('_dup_losers', ()) else None))
        return out

    @staticmethod
    def _by_code(ctx, code):
        stem = re.sub(r'_?\d{3}$', '', code)
        k = ctx.options.get('_code_stems', {}).get(stem)
        return ctx.shared.entities.get(k) if k else None

    def _prep(self, containers, rec):
        """containers: list of (element, meta dict, vol_code) -> info dict id(el) -> (doc_id, years)."""
        info = {}
        code = rec.stem.split('_')[0] if rec.name.endswith('.xml') else None
        for el, meta, tag in containers:
            ys = _years(meta.get('date'))
            lid = (meta.get('id') or '').strip()
            corpus = rec.stem[:7] if code else rec.name.split('-')[0][:7]
            info[id(el)] = (f'{corpus}:{lid}' if lid else rec.stem, ys)
        return info

    def extract_root(self, root, rec, ctx):
        chapters = [d for d in root.iter('div') if d.get('type') == 'chapter']
        containers, spans = [], []
        for ch in chapters:
            if ch.get('subtype') == 'artifact':
                ig = C.child(ch, 'interpGrp')
                meta = {}
                if ig is not None:
                    for i in ig:
                        if C.ln(i.tag) == 'interp' and i.get('type') and i.get('type') not in meta:
                            meta[i.get('type')] = i.get('value') or ''
                containers.append((ch, meta, 'interp'))
                ys = _years(meta.get('date'))
                if ys:
                    spans.append(ys)
        info = self._prep(containers, rec)
        span = (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else None
        self._info, self._span = info, span
        self._orgs = ctx.options.get('_org_keys', set())
        self._stem = rec.stem
        hits = self.scan(root, rec, ctx, Counter())
        hits = [h for h in hits if not self._in_index(h)]
        ms = C.build_mentions(root, rec, ctx, date=C.DocDate(), hits=hits, adjust=self.adjust)
        for el, meta, tag in containers:
            doc_id, ys = info[id(el)]
            ms.extend(self._meta_mentions(meta, doc_id, rec, ctx, ys, 'letter', tag))
        return ms, []

    @staticmethod
    def _in_index(h):
        p = h.el.getparent()
        while p is not None:
            if C.ln(p.tag) == 'div' and p.get('type') == 'chapter':
                return p.get('subtype') == 'index'
            p = p.getparent()
        return False

    def adjust(self, m, hit):
        el = hit.el
        # --- date / doc id of the letter (chapter or txt chunk) ---
        p = el
        while p is not None and id(p) not in self._info:
            p = p.getparent()
        if p is not None:
            doc_id, ys = self._info[id(p)]
            m.doc_id = doc_id
            if ys:
                set_years(m, ys[0], ys[1], 'letter')
        elif self._span:
            set_years(m, self._span[0], self._span[1], 'collection')
        # --- keys ---
        key = (el.get('key') or '').strip()
        if m.local_key is None and key and _OLD.match(key):
            ent = self._by_code(self._ctx, key) if m.kind == 'place' and getattr(self, '_ctx', None) else None
            if ent is not None:
                m.local_key, m.in_register = ent.local_key, True
                m.auth = C.merge_auth(m.auth, ent.auth)
            else:
                m.local_key, m.in_register = C.make_key(SLUG, key), False
        if m.doc_id in self._dup_losers:
            m.excluded_reason = 'duplicate_letter'
        if m.local_key and m.local_key in self._orgs:
            m.kind = 'org'
        return m

    # ---- txt corpora --------------------------------------------------------------------------------------------------
    def _extract_txt(self, rec, ctx):
        raw = rec.read().decode('utf-8-sig', errors='replace').replace('\r\n', '\n')
        parts = re.split(r'(?m)^### (\S+)\s*$', raw)
        out: list[Mention] = []
        self._orgs = ctx.options.get('_org_keys', set())
        self._span = None
        for i in range(1, len(parts) - 1, 2):
            name, chunk = parts[i], parts[i + 1]
            root = C.parse_xml(chunk.encode('utf-8'))
            if root is None:
                continue
            metas: dict = {}
            for e in root.iter('meta'):
                t = e.get('type')
                if t and t not in metas:
                    metas[t] = e.get('value') or ''
            lid = metas.get('id') or name.rsplit('.', 1)[0]
            metas['id'] = lid
            corpus = rec.name.split('-')[0][:7] if rec.name.endswith('-corpus.txt') else (
                rec.path.split('/data/corpus/data/')[1].split('/')[0])
            ys = _years(metas.get('date'))
            self._info = {id(root): (f'{corpus}:{lid}', ys)}
            hits = self.scan(root, rec, ctx, Counter())
            ms = C.build_mentions(root, rec, ctx, date=C.DocDate(), hits=hits, adjust=self.adjust)
            ms.extend(self._meta_mentions(metas, f'{corpus}:{lid}', rec, ctx, ys, 'letter', 'meta'))
            out.extend(ms)
        return out, []


EXTRACTOR = Extractor()
