"""Controversia et Confessio (IEG Mainz / HAB / ULB Darmstadt): Reformation controversy texts (Interim, Flacius,
Osiander...), 138 TEI text files in 7 volumes (pa000009 ... pa000307) plus 8 wdbmeta.xml project files (not read).

Conventions of this edition:
  * <rs type="person|place" ref="#slug"> in the running text: only 7.5% of the person rs and 13% of the place rs carry
    a ref (all resolve); the rest are untied surface mentions (local_key None). rs type="bibl" (literature, sources,
    19,000 tags) is not an entity and is skipped; type typos (peron, perosn, peraon, peson, plac) are mapped,
    type="biblical" is mapped to person (tag label rs@biblical); rs whose @type holds an id are skipped;
  * each file has a standOff with listPerson / listPlace entries (xml:id slug, @corresp to ../register/register-*.xml
    that is NOT in the harvest): no names, authority ids or coordinates. Slugs are edition-wide, so entities are
    shared ('<slug>:<id>') and merged over all files (first wins); the entity name is the slug with '_' turned into
    a space and capitalised ('martin_luther' -> 'Martin Luther'), derived from the id only;
  * dating: no date in the TEI headers (only e-publication dates): the collection period 1548-1575 is applied to all
    texts (date_source 'collection'); the editorial introductions / prefaces (file name contains Einleitung or
    Vorwort) are modern commentary and are left undated (in_period None);
  * Latin and early modern German.
Licence: CC-BY-4.0 AND CC-BY-SA-4.0 (LICENSE_TIERS, T1; 131 files CC-BY-4.0, 7 CC-BY-SA-4.0).

v2 (audit points 8 and 12):
  * zones: 45% of the person and place tags sit in the modern German commentary, not in the 16th-century text. The editors'
    notes are typed ``note1`` (9,986), ``annotation`` (8,811) and ``footnote`` (2,389, with the typos footnotet / fotnote / foot):
    zone ``editorial_commentary``; ``app1`` (5,131) and ``crit_app`` (81) are the critical apparatus: zone ``apparatus``. The notes
    of the printed originals (``place="outer-margin"`` and ``margin-right``) stay in the text. All of these but the text are
    excluded from the default counts;
  * dating: no text is dated by the headers. Each text takes the latest year 1540-1585 of the first 1,200 characters of its own
    body (the title block / imprint of the print, date_source ``text``: 'Anno 1550', '1552.'); a text without such a year takes
    the span of the collection, 1548-1575, as a range (doc_year null: 27 years is too wide for a single year, in_period True
    since the range lies inside 1450-1700). The editorial introductions are still left undated;
  * registers: the TEI registers of the standOff (``register-person|place|bibl.xml``) return 404 on the eXist server. The printed
    Personenregister / Ortsregister of volumes 2, 3, 5, 6, 7 (``controversia_confessio-registers/``; volume 4 answers HTTP 500)
    give names and life dates: a standOff slug whose words are all in ONE register item of the volumes gets that name and
    those dates; nothing else is taken (no identifier, no coordinate). Biblical persons are not in the printed register
    (rs type=biblical stays class ``biblical``).
"""
from __future__ import annotations

import re
import unicodedata

from .. import common as C
from .. import helpers_v2 as V
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'controversia_confessio'
COLLS = ['originals_2026-10-15/web-editions-apis/Controversia_Confessio/']
RENASCOR_IDS = ['controversia-confessio']
LICENCE = 'CC-BY-4.0 AND CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Reformation controversy texts 1548-1575; sparse rs person/place tagging with a slug-only standOff register'
_NOTE_ZONE = {'note1': 'editorial_commentary', 'annotation': 'editorial_commentary', 'footnote': 'editorial_commentary',
              'footnotet': 'editorial_commentary', 'fotnote': 'editorial_commentary', 'foot': 'editorial_commentary',
              'notel': 'editorial_commentary', 'app1': 'apparatus', 'crit_app': 'apparatus'}
OPTIONS = {'default_years': (1548, 1575), 'note_zone': _NOTE_ZONE,
           'type_map': {'peron': 'person', 'perosn': 'person', 'peraon': 'person', 'peson': 'person',
                        'plac': 'place', 'biblical': 'person'}}

_INTRO = re.compile(r'einleitung|vorwort', re.I)
_YEAR = re.compile(r'(?<![\d,.])(15[4-8]\d)(?!\d)')
_REGDIR = V.NEW + '/controversia_confessio-registers/'
_REGFILES = ['pa000010/texts/201_-_Personenregister_C-C3.xml', 'pa000010/texts/202_-_Ortsregister_C-C_3.xml',
             'pa000286/texts/321_-_Personenregister_C-C5.xml', 'pa000286/texts/322_-_Ortsregister_C-C_5.xml',
             'pa000298/texts/321_-_Personenregister_C-C6.xml', 'pa000298/texts/322_-_Ortsregister_C-C_6.xml',
             'pa000307/texts/321_-_Personenregister_C_C7.xml', 'pa000307/texts/322_-_Ortsregister_C_C_7.xml',
             'pa000009/texts/232_-_Ortsregister.xml']
_STOPW = {'von', 'van', 'de', 'der', 'den', 'dem', 'zu', 'zur', 'und', 'des', 'the', 'der'}


def _fold(s: str) -> list[str]:
    s = unicodedata.normalize('NFKD', (s or '').casefold())
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return [w for w in re.sub(r'[^a-z0-9]+', ' ', s).split() if w not in _STOPW and len(w) > 1]


def _register_items() -> list[tuple[str, list[str], str]]:
    """(printed name, folded words, dates) of every item of the printed person / place registers of volumes 2-7."""
    out = []
    for f in _REGFILES:
        root = V.read_xml(_REGDIR + f)
        if root is None:
            continue
        for it in root.iter():
            if C.ln(it.tag) != 'item':
                continue
            txt = C.surface_text(it)
            m = re.match(r'^(.*?)(?:\s+(?=\d)|$)', txt)
            head = (m.group(1) if m else txt).strip(' ,;')
            dm = re.search(r'\(([^()]*\d[^()]*)\)', head)
            name = re.sub(r'\s*\([^()]*\)', '', head).strip(' ,;')
            if name:
                out.append((name, _fold(name), dm.group(1) if dm else ''))
    return out


def _lifespan(d: str):
    ys = re.findall(r'\d{1,4}', d or '')
    if len(ys) >= 2:
        return ys[0], ys[1]
    return None, None


def _humanize(ident: str) -> str:
    return ' '.join(w[:1].upper() + w[1:] for w in re.split(r'[_\s]+', ident.strip()) if w)


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups)
                if r.path.lower().endswith('.xml') and not r.path.endswith('wdbmeta.xml')]

    def prepare(self, ctx):
        out = []
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for el in root.iter():
                n = C.ln(el.tag)
                if n not in ('person', 'place', 'org') or el.get(C.XML_ID) is None:
                    continue
                if not any(C.ln(a.tag) in ('standOff', 'listPerson', 'listPlace') for a in el.iterancestors()):
                    continue
                ident = el.get(C.XML_ID)
                key = C.make_key(ctx.slug, ident)
                if key in ctx.shared.entities:
                    continue
                ent = Entity(local_key=key, coll_slug=ctx.slug, kind=n, name=_humanize(ident),
                             source_path=rec.path, renascor_id=ctx.renascor_id, licence=ctx.licence, tier=ctx.tier,
                             note_type='standoff_slug')
                ctx.shared.add(ent, [ident])
                out.append(ent)
        self._names_from_printed_register(out)
        return out

    @staticmethod
    def _names_from_printed_register(ents):
        items = _register_items()
        if not items:
            return
        for e in ents:
            if e.kind not in ('person', 'place'):
                continue
            slug_words = _fold(e.name)
            if not slug_words:
                continue
            cand = [it for it in items if set(slug_words) <= set(it[1]) and (e.kind == 'place' or len(slug_words) > 1)]
            if len({c[0] for c in cand}) != 1:
                continue
            name, _w, dates = cand[0]
            e.variants = [e.name] if e.name != name else []
            e.name = name
            if e.kind == 'person':
                e.birth, e.death = _lifespan(dates)
            e.note_type = 'printed_register'

    def extract_root(self, root, rec, ctx):
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root))

    def doc_date(self, root, rec, ctx):
        if _INTRO.search(rec.name):
            return C.DocDate()
        text = next((e for e in root.iter() if C.ln(e.tag) == 'text'), None)
        if text is not None:
            head = C.norm(''.join(text.itertext()))[:1200]
            ys = [int(y) for y in _YEAR.findall(head)]
            if ys:
                y = max(ys)                      # the imprint year is the latest year of the title block
                return C.DocDate(y, y, 'text')
        return C.DocDate(ctx.default_years[0], ctx.default_years[1], 'collection')


EXTRACTOR = Extractor()
