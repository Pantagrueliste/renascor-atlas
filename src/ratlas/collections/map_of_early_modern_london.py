"""Map of Early Modern London (MoEML, Univ. of Victoria), static release: 813 TEI files in files/original.

195 files are primary-source texts (catRef mdt:mdtPrimarySource*: Stow's Survey 1598/1633 sections, mayoral shows,
royal entries, drama extracts, proclamations); 618 are modern files (encyclopedia entries, personography PERS1,
organisations ORGS1, bibliography, blog, teaching): they supply the registers but are NOT read for mentions.
Mentions (primary texts only, running text):
  * persons: <name ref="mol:ID"> without type (kind person), <name type="org" ref="mol:ID"> (org);
  * places: <ref target="mol:ID"> (not type="bibl") -> kind place, tag label 'ref@place' (the scan counts only @ref,
    so the 21k place links are invisible in its numbers); <ref target="agas.htm"> and external urls are not places;
  * ids resolve against shared registers ('<slug>:<ID>'): PERS1.xml (persons, names from <reg>; project team
    members, persName type="cont", are dropped; type="lit" (literary / mythical figures) get note_type
    'literary_or_mythical'; ODNB / Wikipedia links of the note go to auth.other as 'odnb:<id>' /
    'wikipedia:en:<title>'), ORGS1.xml (orgs), and the place files ID.xml with a placeInfo div (name, GeoJSON <geo>
    in WGS84, [lon, lat]: a Point gives the point, LineString / Polygon / Multi* the mean of the vertices,
    coord_source 'tei:geo:geojson' / 'tei:geo:geojson-vertex-mean'). Only 536 place files were harvested, so about
    two thirds of the place ids found in the texts stay unresolved with an opaque key (in_register False);
  * if a ref resolves to an entity of another kind, the mention takes the entity's kind (untyped <name> may point at
    a place);
  * persons include medieval and ancient figures cited by Stow (Henry II, Fitzstephen): this is not
    'persons living 1450-1700';
  * dating: stow_<year>_ files by the year of the edition in the file name (1598 / 1633, source 'filename'), the others
    by the source imprint date (sourceDesc, date@when-custom), source 'header'; undated when none, or when the only
    date is after 1700 (imprint of a modern edition, e.g. SIRT1 'The book of Sir Thomas More', 1911);
  * header: names of the source's authors / printers in sourceDesc are mentions with zone 'hdr' (respStmt, i.e.
    MoEML team members, is never read; the staff and the University of Victoria named in the sourceDesc
    provenance paragraphs are dropped by ref, see MODERN_STAFF).
Licence: CC-BY-SA-4.0 (site footer; LICENSE_TIERS T1).

v2 (audit points 6 and 12, harvest of 2026-10-17):
  * place files: the 906 individual place files fetched from the official ``xml/original/`` directory
    (``map_of_early_modern_london-registers/``, Edition 7.0 of 2022-05-05; 1 request per second) are loaded after the 536 of the
    first harvest (first wins): only 4 ids (FINK1, GREE1, LIBE1, STAT2) stay without a file. Only 224 ids carry GeoJSON;
    London (LOND5, 4,218 mentions), the Tower, Southwark and England have none in MoEML itself;
  * the Wikipedia pages listed in a person's ``<list type="links">`` are context links (a list, a battle, a bishopric, a
    spouse's page), not identifiers: a page counts as the person's Wikipedia id only when its title matches the entry's
    name; ODNB ids are kept;
  * the date of a primary text: header dates are looked up first (as in v1); the 11 texts without one are dated by the
    ``resp/date`` of the author / printer / bookseller roles of the titleStmt (``when-custom``) or the date of the
    title page's imprint; a post-1700 imprint (modern edition) still leaves a text undated;
  * mentions inside editorial notes (``note type=editorial``, 819) are zone ``editorial_commentary``.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch0 as H0
from .. import helpers_v2 as V
from ..model import Entity, empty_auth
from .generic_tei import GenericTEI

SLUG = 'map_of_early_modern_london'
COLLS = ['renascor_2026-10/map-of-early-modern-london']
RENASCOR_IDS = ['map-of-early-modern-london']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'London persons, places (ref links) and companies in 195 primary texts, with GeoJSON place coordinates'
OPTIONS = {'opaque_keys': True}

_NUM = re.compile(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?')
_STOW = re.compile(r'(?:^|/)stow_(\d{4})_')


def is_primary(root) -> bool:
    return any(C.ln(e.tag) == 'catRef' and (e.get('target') or '').startswith('mdt:mdtPrimarySource')
               for e in root.iter())


def geojson_point(txt: str):
    """(lat, lon, source) from the GeoJSON fragment found in <geo> (coordinates are [lon, lat])."""
    m = re.search(r'"coordinates"\s*:\s*(.*)$', txt or '', re.S)
    if not m:
        return None
    nums = [float(x) for x in _NUM.findall(m.group(1))]
    pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    pairs = [(lo, la) for lo, la in pairs if -180 <= lo <= 180 and -90 <= la <= 90]
    if not pairs:
        return None
    if '"Point"' in txt and len(pairs) == 1:
        return pairs[0][1], pairs[0][0], 'tei:geo:geojson'
    lon = sum(p[0] for p in pairs) / len(pairs)
    lat = sum(p[1] for p in pairs) / len(pairs)
    return lat, lon, 'tei:geo:geojson-vertex-mean'


def _custom_date(el):
    if el is None:
        return None
    bits = [el.get(a) for a in ('when-custom', 'notBefore-custom', 'notAfter-custom', 'from-custom', 'to-custom',
                                'when', 'notBefore', 'notAfter') if el.get(a)]
    return '/'.join(dict.fromkeys(bits)) or None


_STOP = {'sir', 'lord', 'lady', 'king', 'queen', 'the', 'of', 'de', 'la', 'le', 'st', 'saint', 'dame', 'duke', 'earl',
         'bishop', 'archbishop', 'mr', 'mrs', 'master', 'alderman', 'knight'}


def _toks(s: str) -> set:
    return {t for t in H0.fold(re.sub(r'\([^)]*\)', ' ', s or '').replace('_', ' ')).split() if len(t) > 1 and t not in _STOP}


def wikipedia_matches(name_forms: list, wp: str) -> bool:
    """'wikipedia:en:Thomas_More' belongs to 'More, Thomas' (all the name's words are in the page title and the title adds
    at most one word); 'Peasants%27_Revolt' does not belong to 'Standish, Ralph'."""
    import urllib.parse
    title = urllib.parse.unquote(wp.split(':', 2)[-1])
    tt = _toks(title)
    if not tt:
        return False
    for nm in name_forms:
        nt = _toks(nm)
        if nt and nt <= tt and len(tt - nt) <= 1:
            return True
    return False


def _links_auth(note, name_forms: list = ()) -> dict:
    a = empty_auth()
    if note is None:
        return a
    for r in note.iter():
        t = r.get('target') if C.ln(r.tag) == 'ref' else None
        if not t:
            continue
        m = re.search(r'odnb-\d+-e-(\d+)', t)
        if m:
            x = 'odnb:' + m.group(1)
            if x not in a['other']:
                a['other'].append(x)
            continue
        s, i = C.classify_token(t)
        if s == 'other' and i.startswith('wikipedia:') and i not in a['other'] and wikipedia_matches(list(name_forms), i):
            a['other'].append(i)
    return a


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        out = []
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            stem = rec.stem
            if stem == 'PERS1':
                out += self._persons(root, rec, ctx)
            elif stem == 'ORGS1':
                out += self._orgs(root, rec, ctx)
            else:
                e = self._place(root, rec, ctx)
                if e is not None and e.local_key not in ctx.shared.entities:
                    ctx.shared.add(e, [e.local_key.split(':', 1)[1]])
                    out.append(e)
        out += self._new_places(ctx)
        return out

    def _new_places(self, ctx):
        """The individual place files fetched on 2026-10-17 (not in the scan; read by path)."""
        import os
        base = V.NEW + '/map_of_early_modern_london-registers/xml/original/'
        d = C.raw_root() / base
        if not V.use_new() or not d.is_dir():
            return []
        out = []
        for name in sorted(os.listdir(d)):
            if not name.endswith('.xml'):
                continue
            root = V.read_xml(base + name)
            if root is None:
                continue
            rec = C.FileRecord(path=base + name, coll='originals_2026-10-17/map_of_early_modern_london-registers')
            e = self._place(root, rec, ctx)
            if e is not None and e.local_key not in ctx.shared.entities:
                ctx.shared.add(e, [e.local_key.split(':', 1)[1]])
                out.append(e)
        return out

    def _persons(self, root, rec, ctx):
        out = []
        for p in root.iter():
            if C.ln(p.tag) != 'person' or p.get(C.XML_ID) is None:
                continue
            pn = C.child(p, 'persName')
            ptype = (pn.get('type') or '') if pn is not None else ''
            if ptype == 'cont':
                continue
            reg = C.child(pn, 'reg') if pn is not None else None
            name = C.surface_text(reg) if reg is not None else (C.name_text(pn) if pn is not None else p.get(C.XML_ID))
            ident = p.get(C.XML_ID)
            birth = _custom_date(C.child(p, 'birth')) or _custom_date(C.child(p, 'floruit'))
            death = _custom_date(C.child(p, 'death'))
            ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='person', name=name,
                         source_path=rec.path, renascor_id=ctx.renascor_id, birth=birth, death=death,
                         auth=_links_auth(C.child(p, 'note'), [name] + ([C.name_text(pn)] if pn is not None else [])),
                         note_type={'hist': 'person', 'lit': 'literary_or_mythical'}.get(ptype, ptype or None),
                         licence=ctx.licence, tier=ctx.tier)
            ctx.shared.add(ent, [ident])
            out.append(ent)
        return out

    def _orgs(self, root, rec, ctx):
        out = []
        for o in root.iter():
            if C.ln(o.tag) != 'org' or o.get(C.XML_ID) is None:
                continue
            on = C.child(o, 'orgName')
            name = C.norm(on.text) if on is not None and on.text else ''
            reg = C.child(on, 'reg') if on is not None else None
            variants = [C.surface_text(reg)] if reg is not None and C.surface_text(reg) else []
            ident = o.get(C.XML_ID)
            ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='org',
                         name=name or (variants[0] if variants else ident), variants=variants,
                         source_path=rec.path, renascor_id=ctx.renascor_id,
                         note_type=(o.get('type') or 'org').lower(), licence=ctx.licence, tier=ctx.tier)
            ctx.shared.add(ent, [ident])
            out.append(ent)
        return out

    def _place(self, root, rec, ctx):
        info = next((d for d in root.iter() if C.ln(d.tag) == 'div' and d.get('type') == 'placeInfo'), None)
        if info is None:
            return None
        pl = next((p for p in info.iter() if C.ln(p.tag) == 'place'), None)
        if pl is None:
            return None
        names = [C.surface_text(c) for c in pl if C.ln(c.tag) == 'placeName' and C.surface_text(c)]
        ident = root.get(C.XML_ID) or rec.stem
        lat = lon = csrc = None
        for g in pl.iter():
            if C.ln(g.tag) == 'geo':
                r = geojson_point(g.text or C.surface_text(g))
                if r:
                    lat, lon, csrc = r
                    break
        return Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='place',
                      name=names[0] if names else ident, variants=names[1:], lat=lat, lon=lon, coord_source=csrc,
                      source_path=rec.path, renascor_id=ctx.renascor_id, note_type='place',
                      licence=ctx.licence, tier=ctx.tier)

    def extract_root(self, root, rec, ctx):
        if not is_primary(root):
            return [], []
        text = next((e for e in root.iter() if C.ln(e.tag) == 'text'), None)
        if text is None:
            return [], []
        # make the editor's tagging visible to the generic scanner (type attributes), remembering the origin
        for el in root.iter():                 # header too (sourceDesc authors / printers); respStmt etc. are skipped
            n = C.ln(el.tag)
            if n == 'name' and (el.get('ref') or '').startswith('mol:') and not el.get('type'):
                el.set('type', 'person')
            elif n == 'ref' and (el.get('target') or '').startswith('mol:') and el.get('type') not in ('bibl', 'mol:bibl'):
                el.set('ref', el.get('target'))
                el.set('type', 'place')
                el.set('_orig', 'ref')
                el.tag = el.tag.replace('}ref', '}rs') if '}' in el.tag else 'rs'
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust(ctx)), []

    def doc_date(self, root, rec, ctx):
        m = _STOW.search(rec.path)
        if m:
            y = int(m.group(1))
            return C.DocDate(y, y, 'filename')
        d = C.header_dates(root)
        if d.year_from is None:
            d = self._respstmt_date(root)
        if d.year_from is not None and d.year_from > 1700:      # imprint of a modern edition (e.g. SIRT1, 1911)
            return C.DocDate()
        return d

    @staticmethod
    def _respstmt_date(root):
        """Date of the source: resp/date of the author / printer / bookseller (titleStmt), else the title page imprint."""
        ys = []
        for el in root.iter():
            if C.ln(el.tag) == 'date' and el.get('when-custom'):
                par = el.getparent()
                if par is not None and C.ln(par.tag) == 'resp' and (par.get('ref') or '') in (
                        'molresp:aut', 'molresp:prt', 'molresp:bsl', 'molresp:pub'):
                    y = C._attr_year(el.get('when-custom'))
                    if y:
                        ys.append(y)
        if not ys:
            for el in root.iter():
                if C.ln(el.tag) == 'docImprint':
                    for d in el.iter():
                        if C.ln(d.tag) == 'date' and (y := C._attr_year(d.get('when-custom') or d.get('when'))):
                            ys.append(y)
                    break
        return C.DocDate(min(ys), max(ys), 'header') if ys else C.DocDate()

    def adjust(self, ctx):
        def fn(m, hit):
            if m.zone == 'hdr' and (m.ref_raw or '').rsplit(':', 1)[-1] in MODERN_STAFF:
                return None                     # 'transcribed by ... at the University of Victoria' (sourceDesc/p)
            if hit.el.get('_orig') == 'ref':
                m.tag = 'ref@place'
            if m.local_key and m.local_key.rsplit(':', 1)[-1] in PLACEHOLDER_IDS:
                # LLLL1 is MoEML's "PLACEHOLDER LOCATION": the id of every place the project has not identified (303 different places
                # in the primary texts: Almory, Pimlico path, Dagger ...). It names no place, so the mention is keyed by its own
                # surface like any unregistered one (v2 audit, per-edition sample: 302 mentions counted as one place).
                m.local_key, m.in_register = None, None
                m.auth = empty_auth()
                return m
            if m.local_key and m.in_register:
                ent = ctx.shared.entities.get(m.local_key)
                if ent is not None and ent.kind != m.kind:
                    m.kind = ent.kind
            return m
        return fn

    def extract(self, rec, ctx):
        root = C.parse_xml(rec.read())
        if root is None:
            raise ValueError('unparseable XML')
        return self.extract_root(root, rec, ctx)


# Modern project staff / host institution named in the provenance paragraphs of sourceDesc (not respStmt, so the
# generic header skip does not see them): 1,305 header 'mentions' in the first full run.
# v2.2: POOL1 is "PLACE OUTSIDE OF LONDON" (84 mentions of Stratford, Hackney, Chelsea College ... that the project placed outside the City): the id of
# many places, the same case as LLLL1; the old build put all of them at one point (51.53 N, 0.0 E).
PLACEHOLDER_IDS = {'LLLL1', 'POOL1'}
MODERN_STAFF = {'HOLM3', 'UVIC3', 'JENS1', 'RAHT1', 'SCHA2', 'TEAM1', 'TAKE1', 'KAET1'}

EXTRACTOR = Extractor()
