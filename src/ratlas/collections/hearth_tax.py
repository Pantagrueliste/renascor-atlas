"""Hearth Tax Digital (Roehampton / Univ. Graz, GAMS): 61 TEI transcriptions of hearth tax returns 1662-1674.

What is tagged: every <persName> is a taxpayer row of a list (name + <measure unit="hearths">), every <placeName>
is a heading of the list (county, hundred, parish, township, ward, street...). They are not mentions in running
text, so co-occurrence of two persons only means 'same list'.

Handling:
  * kind: persName -> person; placeName -> place. The place type (@type: county, hundred, parish, township...) is
    appended to the tag as 'placeName@parish' (documented extension of the 'rs@type' convention) because it gives
    the administrative level.
  * place surface: the regularised reading (<reg>, e.g. "Ward of St Michael") replaces the diplomatic Latin
    abbreviation (<orig>) when the heading has both; the header title place ('Bristol') is a normal mention.
  * roles (non-standard, counted as nonstandard_role): 'taxpayer' (ana h_tp / h:tp), 'owner' (ana ...owner),
    'tenant' (ana ...tenant); persons with other ana codes (h_ad, h_adp) and anything else stay 'mention'.
  * refs: 1,586 persName and 25 placeName carry a ref of the form https://gams.uni-graz.at/o:htx.persons#ID,
    .../o:htx.places#Name or a bare '#ID'. The GAMS registers are not in the harvest, so the ids are opaque
    edition keys ('hearth_tax:person#ID', 'hearth_tax:place#Name'; in_register False). They are NOT put in
    auth.other (they are not authority ids). No GND/Wikidata/GeoNames/coordinates exist in the data.
  * places in msIdentifier (archive: London, TNA) get role 'repository'.
  * dating: header dates (the taxation date is in the title, <date ana="h_taxationDate" when="1670">); fall back to
    the edition period 1662-1674 (date_source 'collection') if a file has none.
v2 (audit point 11, issues 162-167):
  * identity: a taxpayer row has no id, so the bare name is not an identity ('john smith' was ONE person with 363 mentions across
    counties): every untied row gets its own opaque key 'hearth_tax:row#<file>#<n>' (n = reading order), so rows are never merged
    into homonym piles; rows with a GAMS ref keep their ref key;
  * rows without a letter ('...', empty) are dropped; vacant houses ('A vacant', 'in vacua domo') are kept with
    ``excluded_reason`` 'vacant'; editorial text inside the tag (<note>, <add>, <supplied>: 'Supplied from exemption certificate...')
    is not part of the surface;
  * dating: when a file has no header date (the Norfolk exemption certificates E179/335-338 got the default 1668) the years of its
    ``h_taxationDate`` elements are used (1671-1673), date_source 'text';
  * the National Archives address of the msIdentifier is the repository (zone ``repository``, excluded from the default counts).
v2.4 (check of v2.3): a heading name is scoped to the county of its file. 'Kirkby' of the Lancashire returns and 'Kirkby' of the North Riding return were ONE entity (13 entities and
  37 mentions in all), placed once, on the Kirkby of Merseyside, 150 km from the North Riding one. The county of the file (``regions.doc_region`` on the county headings, the header title and
  the city headings, as the geocoder reads it) is part of the key of every undetermined heading: 'surf:kirkby @lancashire'. Headings that carry a ref keep their ref key; the county
  headings themselves are not split.
Licence: CC-BY-NC-4.0 (TEI header licence element and LICENSE_TIERS), T2. The publisher's robots.txt forbids
scripted access: nothing is fetched, only the harvested files are read.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch1 as H
from .. import regions as RGN

SLUG = 'hearth_tax'
COLLS = ['renascor_2026-10/hearth-tax-digital']
RENASCOR_IDS = ['hearth-tax-digital']
LICENCE = 'CC-BY-NC-4.0'
TIER = 'T2'
DESCRIPTION = 'English and Welsh hearth tax returns 1662-1674: taxpayer rows and administrative place headings'
OPTIONS = {'default_years': (1662, 1674)}

_ID_RX = re.compile(r'o:htx\.(persons|places)#(.+)$')
_VACANT = re.compile(r'^(?:an?\s+|the\s+)?vacan[tc]\w*(?:\s+(?:house|howse|hearth|tenement)s?)?\.?$|^in\s+vacu[ao]\s+domo|^void\b|^empty\b', re.I)
_EDITORIAL = ('note', 'add', 'supplied')


def _own_surface(el) -> str:
    """Surface of a taxpayer without the editors' <note> / <add> / <supplied> text (diplomatic <orig> kept, <reg> dropped)."""
    out = [el.text or '']
    for c in el:
        if isinstance(c.tag, str) and C.ln(c.tag) not in _EDITORIAL:
            out.append(C.surface_text(c))
        out.append(c.tail or '')
    return C.norm(''.join(out))


def _slug(s) -> str:
    import unicodedata
    t = unicodedata.normalize('NFKD', str(s or '').casefold())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', t).split())


def file_county(hits) -> str:
    """Label of the county (or riding, or city) of a file from its own headings (``regions.doc_region``): '' when no heading names one."""
    cty = [h.surface for h in hits if h.kind == 'place' and h.tag == 'placeName@county' and h.surface]
    hdr = [h.surface for h in hits if h.kind == 'place' and h.zone in ('hdr', 'header') and h.tag in ('placeName', 'placeName@') and h.surface]
    oth = [h.surface for h in hits if h.kind == 'place' and h.tag in ('placeName@city', 'placeName@borough') and h.surface]
    r = RGN.doc_region(cty, hdr, oth)
    return r[0] if r else ''


def place_key(surface: str, county: str) -> str:
    """The entity key of a heading: the surface scoped to the county of its file ('surf:kirkby @lancashire'). It starts with 'surf:' like every surface key."""
    return f'surf:{_slug(surface)} @{_slug(county)}'


class Extractor(H.BaseTEI):
    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        for h in hits:
            if h.kind == 'place' and h.name == 'placeName':
                t = (h.el.get('type') or '').strip().lower()
                if t:
                    h.tag = 'placeName@' + t
                # regularised reading for headings that hold <choice><orig/></choice><reg/>
                reg = [c for c in h.el.iter() if C.ln(c.tag) == 'reg']
                if reg:
                    s = C.norm(' '.join(C.surface_text(r) for r in reg))
                    if s:
                        h.surface = s
            elif h.kind == 'person' and h.name == 'persName' and h.zone == 'txt':
                if any(C.ln(c.tag) in _EDITORIAL for c in h.el.iter() if c is not h.el):
                    h.surface = _own_surface(h.el)
                ana = (h.el.get('ana') or '').lower()
                if 'owner' in ana:
                    h.role = 'owner'
                elif 'tenant' in ana:
                    h.role = 'tenant'
                elif re.search(r'\bh[_:]tp\b', ana):
                    h.role = 'taxpayer'
        self._county = file_county(hits)
        return hits

    def doc_date(self, root, rec, ctx):
        d = super().doc_date(root, rec, ctx)
        if d.source == 'collection':
            ys = []
            for el in root.iter():
                if C.ln(el.tag) == 'date' and 'taxationdate' in (el.get('ana') or '').lower():
                    y = C._attr_year(el.get('when'))
                    if y and 1600 <= y <= 1700:
                        ys.append(y)
            if ys:
                return C.DocDate(min(ys), max(ys), 'text')
        return d

    def extract_root(self, root, rec, ctx):
        self._row = 0
        return super().extract_root(root, rec, ctx)

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is None:
            return None
        if m.kind == 'person' and m.tag == 'persName':
            if _VACANT.match(m.surface or ''):
                m.excluded_reason = 'vacant'
            self._row = getattr(self, '_row', 0) + 1
        ref = m.ref_raw
        if ref and m.local_key is None:
            tok = ref.split()[0]
            mt = _ID_RX.search(tok)
            if mt:
                ident = f'{"person" if mt.group(1) == "persons" else "place"}#{mt.group(2)}'
            elif tok.startswith('#') and len(tok) > 1:
                ident = f'{m.kind}#{tok[1:]}'
            else:
                ident = None
            if ident:
                H.set_opaque(m, SLUG, ident)
                H.drop_url_other(m, 'http://gams.uni-graz.at/o:htx')
                H.drop_url_other(m, 'https://gams.uni-graz.at/o:htx')
        if m.kind == 'place' and m.local_key is None and getattr(self, '_county', '') and m.role == 'mention' and m.zone != 'repository' and m.surface \
                and str(m.tag or '').startswith('placeName@') and m.tag != 'placeName@county':
            m.local_key = place_key(m.surface, self._county)
        if m.kind == 'person' and m.local_key is None and m.tag == 'persName' and m.surface:
            H.set_opaque(m, SLUG, f'row#{m.doc_id}#{self._row}')           # one row = one taxpayer, never a name pile
        return m


EXTRACTOR = Extractor()
