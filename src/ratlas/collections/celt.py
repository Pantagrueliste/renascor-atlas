"""CELT, Corpus of Electronic Texts (UCC, Cork): TEI P4 / SGML-derived (<TEI.2>, ISO-8859-1, named entities).

437 distinct files (549 minus 112 duplicates; a few are empty shells). The editorial declaration says names are not
tagged in most files; where they are (about 87 files, 60% of the tags in Keating's Foras Feasa G100054 and G100070)
<name type="..."> marks them in the running text. There is no @ref/@key, no register: mentions have no local_key.
Parsing: helpers_batch2.sgml_to_xml_bytes (DOCTYPE dropped, SGML entities mapped to characters) before lxml.
Kinds from the free-text @type (the part before ':' or '/' decides, case-insensitive):
  person   person, personn, pn, personified, and the supernatural beings deity, angel, demon, nomen sacrum (God,
           Christ...), mythological, kept as person mentions: the tag label keeps the type ('name@deity',
           'name@nomen sacrum', 'name@person:saint') so that later stages can filter them;
  place    place*, river, residence, country;
  org      person:collective (peoples, kindreds, groups of persons);
  not read book, manuscript, journal, animal, halidom, weapon, constellation, hymn, kenning... (objects and works).
Dating: profileDesc/creation, the dateRange@from/@to (or its text) else 4-digit years in creation; no guess when the
header has no creation date (undated, in_period None). Annals that span 550-1616 overlap the period although most of
their tags concern earlier centuries: the date is the document's, not the passage's.
v2 (issues 59-62): (a) a date range wider than 20 years has no doc_year and is in_period True only when it lies inside 1450-1700 (the Annals of
the Four Masters 550-1616 and Keating-like ranges straddling 1450 are None, not 'in period'); (b) deity / nomen sacrum / angel / demon /
personified / mythological names are person_class mythological (743 mentions) and stay visible; (c) the scribes named in the header
(``name type=person:scribe``: 18th-19th c. copyists such as Micheál Óg Ó Longáin, 60 in G402563) are not dated by the poem: undated;
(d) there is no register or @ref: persons are still surface strings (given names, spelling variants): do not draw person-level counts
from this edition.
Licence: custom CELT terms (LICENSE_TIERS CELT, T3, redistribution not permitted; internal use only: no context).
"""
from __future__ import annotations

import re

from .. import common as C
from ..helpers_batch2 import sgml_to_xml_bytes
from .generic_tei import GenericTEI

SLUG = 'celt'
COLLS = ['celt']
RENASCOR_IDS = ['celt']
LICENCE = 'LicenseRef-CELT-custom'
TIER = 'T3'
DESCRIPTION = 'CELT Irish and Latin texts: editor-tagged names (only a minority of the files)'
OPTIONS = {}

_PERSON = {'person', 'personn', 'pn', 'personified', 'deity', 'angel', 'demon', 'nomen sacrum', 'nomen saacrum',
           'mythological'}
_PLACE = {'place', 'river', 'residence', 'country'}


def classify_type(typ: str):
    t = re.sub(r'\s+', ' ', (typ or '').strip().lower())
    if not t:
        return None
    if t.startswith('person') and 'collective' in t:
        return 'org'
    head = re.split(r'[:/]', t)[0].strip()
    if head in _PERSON:
        return 'person'
    if head in _PLACE:
        return 'place'
    return None


_RANGE_NUM = re.compile(r'(?<!\d)(\d{3,4})(?!\d)')


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def extract(self, rec, ctx):
        data = rec.read()
        if not data.strip():
            return [], []
        root = C.parse_xml(sgml_to_xml_bytes(data))
        if root is None or len(root) == 0:
            raise ValueError('unparseable XML')
        for el in root.iter('name'):
            t = el.get('type')
            if t:
                k = classify_type(t)
                el.set('x-orig', re.sub(r'\s+', ' ', t.strip().lower()))
                el.set('type', k or 'x-other')
        ms, _ = self.extract_root(root, rec, ctx)
        return ms, []             # the <person> elements of header listBibl are bibliography, not a register

    def doc_date(self, root, rec, ctx):
        for el in root.iter('creation'):
            ys = []
            for dr in el.iter('dateRange'):
                for a in ('from', 'to'):
                    v = dr.get(a)
                    if v and v.strip().lstrip('-').isdigit():
                        ys.append(int(v))
                if not ys:
                    ys += [int(x) for x in _RANGE_NUM.findall(C.surface_text(dr))]
            if not ys:
                ys = [int(x) for x in _RANGE_NUM.findall(C.surface_text(el))]
            ys = [y for y in ys if 100 <= y <= 2100]
            if ys:
                return C.DocDate(min(ys), max(ys), 'header')
        return C.DocDate()

    def adjust(self, m, hit):
        o = hit.el.get('x-orig')
        if o and hit.name == 'name':
            m.tag = f'name@{o}'
        if m.zone == 'hdr':
            m.context = None
            if o and 'scribe' in o:
                from ..helpers_v2 import set_range
                set_range(m, None, None, None)          # a later copyist of the manuscript, not a contemporary of the text
        return m


EXTRACTOR = Extractor()
