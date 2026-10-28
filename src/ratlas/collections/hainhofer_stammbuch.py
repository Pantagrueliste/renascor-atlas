"""Philipp Hainhofer, Grosses Stammbuch (HAB Wolfenbuettel, stammbuch.hab.de), one TEI file of 268 entries.

<rs type="person" ref="psn:slug"> (1,486 tags). The 2,096 <term ref="GND-URL"> of the Verschlagwortung divs are subject
keywords (Bucheinband, Samt), not persons or places: not read. The modern editors in the header (respStmt / editor) are
skipped by the scanner.
Roles are taken from the entry sub-division that holds the tag: div type 'Name' (writer of the entry) -> 'author',
'Kuenstler' (artist) -> 'artist' (non-standard role, documented here), 'dargestellte_Personen' -> 'subject',
everything else (transcription, commentary) -> 'mention'.
Dating: per entry (each mention gets the years of the date element in its entry's 'Datierung' div, source 'header');
entries without a date (blank pages, frames) fall back to the stated range of the album, 1596-1633 ('collection').
Entry dates of the binding (1881-1931) or later restorations are real dates and make those mentions out of period.
Licence: LICENSE_TIERS CC-BY-SA-3.0-DE (T1); the TEI header says CC BY-SA 4.0 (target .../by-sa/4.0/): both BY-SA.

v2:
  * zones: 80% of the person tags sit in the modern commentary divs of each entry (weitere_Kontexte, Deutung,
    Bezug_Stammbuch, Besonderheiten, Bildbeschreibung, Anmerkungen): zone ``editorial_commentary``, excluded from the
    default counts. Only Name, Kuenstler, Transkription, Datierung and dargestellte_Personen are the album's own text;
  * registers (``hainhofer_stammbuch-registers/``, harvested from the public site API of stammbuch.hab.de, robots.txt allows
    it; the raw register files on git.hab.de are disallowed and were not fetched): 201 person pages (name, birth /
    death, GND, variant names) and the BEACON file slug -> GND, loaded as the shared register ('hainhofer_stammbuch:<slug>',
    in_register True); the place register (24 places with GeoNames ids) from the site's content API;
  * the ``Ort`` div of an entry (place of writing, plain text, 103 non-empty of 268) is read as a place mention with role
    ``written_at`` (non-standard, documented here): '?' at the end of the value means the editors are unsure (cert 'low');
    a value is resolved to a register place by its folded name or alternative designation.
"""
from __future__ import annotations

import glob
import json
import os
import re

from .. import common as C
from .. import helpers_batch0 as H0
from .. import helpers_batch1 as H1
from .. import helpers_v2 as V
from ..helpers_batch3 import enclosing, opaque_from_prefixed, set_years
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'hainhofer_stammbuch'
COLLS = ['originals_2026-10-15/oct-binary/oct__hainhofer-grosses-stammbuch']
RENASCOR_IDS = ['hainhofer-grosses-stammbuch']
LICENCE = 'CC-BY-SA-3.0-DE'
TIER = 'T1'
DESCRIPTION = "Hainhofer's Grosses Stammbuch 1596-1633: persons by psn: slug with the site register, place of writing"
_COMMENTARY = {'weitere_Kontexte': 'editorial_commentary', 'Deutung': 'editorial_commentary',
               'Bezug_Stammbuch': 'editorial_commentary', 'Besonderheiten': 'editorial_commentary',
               'Bildbeschreibung': 'editorial_commentary', 'Anmerkungen': 'editorial_commentary'}
OPTIONS = {'default_years': (1596, 1633), 'div_zone': _COMMENTARY}

_ROLE_BY_DIV = {'Name': 'author', 'Künstler': 'artist', 'dargestellte_Personen': 'subject'}
_REGDIR = V.NEW + '/hainhofer_stammbuch-registers/'


def _html_text(el) -> str:
    return C.norm(''.join(el.itertext()))


def _parse_person(slug_id: str, d: dict, ctx, gnd_beacon: dict):
    from lxml import html as LH
    title = C.norm(d.get('title') or '') or slug_id
    root = LH.fromstring('<div>' + (d.get('html') or '') + '</div>')
    gnd = gnd_beacon.get(slug_id)
    for a in root.iter('a'):
        if (a.get('href') or '').startswith('https://d-nb.info/gnd/'):
            gnd = gnd or a.get('href')
    auth = C.empty_auth()
    if gnd:
        C.parse_auth([gnd if gnd.startswith('http') else 'gnd:' + gnd], into=auth)
    birth = death = None
    for div in root.iter('div'):
        cls = div.get('class') or ''
        if cls == 'birthdeath':
            t = _html_text(div)
            m = re.search(r'\*\s*([^✝]*?)\s*(?:✝\s*(.*))?$', t)
            if m:
                birth, death = (m.group(1) or None), (m.group(2) or None)
    variants = [C.norm(''.join(p.itertext())) for p in root.iter('p') if p.get('class') == 'variantname']
    return Entity(local_key=C.make_key(ctx.slug, slug_id), coll_slug=ctx.slug, kind='person', name=title,
                  source_path=_REGDIR + f'site-api/personen/{slug_id}.json', renascor_id=ctx.renascor_id,
                  variants=[v for v in variants if v and v != title], birth=birth, death=death, auth=auth,
                  note_type='person', licence=ctx.licence, tier=ctx.tier)


def load_registers(ctx) -> list:
    base = C.raw_root() / _REGDIR
    if not base.is_dir():
        return []
    beacon = {}
    bp = base / 'cms' / 'stammbuch-beacon.txt'
    if bp.exists():
        for line in bp.read_text(encoding='utf-8-sig').splitlines():
            if line and not line.startswith('#') and '||' in line:
                gnd, slug = line.split('||', 1)
                beacon[slug.strip()] = gnd.strip()
    out = []
    for f in sorted(glob.glob(str(base / 'site-api' / 'personen' / '*.json'))):
        sid = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f, encoding='utf-8'))
        except ValueError:
            continue
        e = _parse_person(sid, d, ctx, beacon)
        ctx.shared.add(e, [f'psn:{sid}', sid])
        out.append(e)
    # place register: table of the Ortsregister page (name, alternative designations, GeoNames link)
    cp = base / 'site-api' / 'content.json'
    if cp.exists():
        from lxml import html as LH
        pages = json.load(open(cp, encoding='utf-8')).get('pages', [])
        for pg in pages:
            if pg.get('slug') == '/register/orte' and pg.get('content'):
                root = LH.fromstring('<div>' + pg['content'] + '</div>')
                for tr in root.iter('tr'):
                    tds = tr.findall('td')
                    if len(tds) < 5:
                        continue
                    name = _html_text(tds[0])
                    alts = [C.norm(x) for x in _html_text(tds[1]).split(',') if C.norm(x)]
                    local = _html_text(tds[2])
                    auth = C.empty_auth()
                    for a in tds[4].iter('a'):
                        C.parse_auth([a.get('href') or ''], into=auth)
                    if not name:
                        continue
                    key = C.make_key(ctx.slug, 'ort:' + H0.fold(name).replace(' ', '_'))
                    e = Entity(local_key=key, coll_slug=ctx.slug, kind='place', name=name,
                               source_path=_REGDIR + 'site-api/content.json', renascor_id=ctx.renascor_id,
                               variants=[v for v in [local, *alts] if v and v != name], auth=auth, note_type='place',
                               licence=ctx.licence, tier=ctx.tier)
                    ctx.shared.add(e, [])
                    out.append(e)
    return out


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        return load_registers(ctx)

    def doc_date(self, root, rec, ctx):
        return C.DocDate(1596, 1633, 'collection')       # per-entry dates are applied in adjust()

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        for h in hits:
            if h.zone == 'txt':
                p = h.el.getparent()
                while p is not None:
                    if C.ln(p.tag) == 'div' and p.get('type') in _ROLE_BY_DIV:
                        h.role = _ROLE_BY_DIV[p.get('type')]
                        break
                    p = p.getparent()
        hits += self.ort_hits(root, C.doc_language(root))
        return hits

    def ort_hits(self, root, lang):
        """The ``Ort`` div of each entry: place of writing as plain text."""
        out = []
        for div in root.iter():
            if C.ln(div.tag) != 'div' or div.get('type') != 'Ort':
                continue
            for p in div:
                if C.ln(p.tag) != 'p':
                    continue
                txt = C.norm((p.text or '') + ''.join(c.tail or '' for c in p))
                if not txt:
                    continue
                out.append(C.Hit(kind='place', tag='div@Ort', name='p', surface=txt, zone='txt', role='written_at',
                                 refs={}, lang=lang, context=None, el=p, ancestors=H1.ancestors_of(p)))
        return out

    def _entry_years(self, entry):
        cache = self.__dict__.setdefault('_cache', {})
        k = id(entry)
        if k in cache and cache[k][0] is not entry:
            del cache[k]                              # id reused by another proxy
        if k not in cache:
            rg = None
            for d in entry:
                if C.ln(d.tag) == 'div' and d.get('type') == 'Datierung':
                    rgs = [r for x in d.iter() if C.ln(x.tag) == 'date' and (r := C.date_range(x))]
                    rg = (min(r[0] for r in rgs), max(r[1] for r in rgs)) if rgs else None
                    break
            cache[k] = (entry, rg)
        return cache[k][1]

    def adjust(self, m, hit):
        if not m.local_key:
            opaque_from_prefixed(m, SLUG, ['psn'])
        if hit.tag == 'div@Ort':
            raw = m.surface
            if raw.endswith('?'):
                m.cert = 'low'
            val = raw.rstrip('?').strip()
            m.surface = val
            ent = self._place_by_name(val)
            if ent is not None:
                H0.attach(m, ent)
        entry = enclosing(hit.el, 'div', type='Eintrag')
        if entry is not None:
            rg = self._entry_years(entry)
            if rg:
                set_years(m, rg[0], rg[1], 'header')
        return m

    def _place_by_name(self, name):
        ctx = getattr(self, '_ctx', None)
        if ctx is None:
            return None
        return H0.unique_by_name(ctx.shared, name, 'place')

    def extract(self, rec, ctx):
        self._ctx = ctx
        return super().extract(rec, ctx)

    def extract_root(self, root, rec, ctx):
        self.__dict__['_cache'] = {}
        return super().extract_root(root, rec, ctx)


EXTRACTOR = Extractor()
