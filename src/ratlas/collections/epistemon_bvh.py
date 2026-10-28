"""Epistemon / Bibliotheques Virtuelles Humanistes (CESR Tours): 53 annotated texts (Rabelais, Diodorus 1554, letters),
tei-fr/Projects/BVH/Corpus.

Mention-level tagging in the text: <persName key="#pers_x">, untyped <name> (29,949: ancient authors, gods, characters),
<name type="allegorie">, <placeName type="ville|pays|region|batiment|rue|...">, <geogName> and <orgName type="peuple">
(peoples). There is no register for the keys (9 person elements in the whole corpus) and no authority id or coordinate.
Handling:
  * untyped <name> in the text is read as a person (type="person" set on the in-memory tree; tag 'name@person');
    type 'allegorie' is person ('name@allegorie'); header names stay untouched;
  * keys '#pers_x', '#persh_x', '#persf_x', '#persm_x', '#loc_x', '#locf_x', '#loch_x', '#locm_x' are stable ids across
    the corpus: opaque local keys '<slug>:<id>' (in_register False). The prefix carries the edition's own class
    (pers = historical/author, persh = historical, persf = fictional, persm = mythical, loc/loch = place, locf =
    fictional place, locm = mythical place); there is no register entity to hold it, so downstream code must read it
    from local_key. Other keys (document-local ids such as 'antoine_loisel', 'pers1') are scoped to the file
    ('<slug>:<doc_id>#<id>') and header keys of a file-level register resolve to that register;
  * v2 (issues 263-264): the class of a mention comes from the key prefix when there is one (persf = fictional -> person_class
    'character', persm = mythical -> 'mythological') and, for the untyped <name> without a key (88% of the person mentions), from a
    short lexicon of the divine, mythical and fictional names of the corpus (Dieu, Jésus, Vénus, Amour...; Pantagruel, Panurge,
    Gargantua...); <orgName type=peuple> (4,679 of the 4,724 org mentions) is a people: person_class 'people'. None of them is a historical
    person; all stay visible, the class tells them apart;
  * dating: the imprint date of the original (monogr/imprint/date) or, for manuscripts and letters, the msDesc history
    date (1529-1616); the generic header reader would mix in the digital edition's dates and the modern bibliography;
  * surfaces: a name broken over a line end keeps 'Antio- pe' in the text; hyphen + space before a lower-case letter and
    bracketed page numbers ('[81]') inside a tagged name are removed.
Licence: the 53 headers say CC BY-NC-SA 4.0 (LICENSE_TIERS Epistemon / BVH_Epistemon_Print agree, T2; FreEM headers of
other texts say cc-by-nc-nd, none in this folder).
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'epistemon_bvh'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/tei-fr/Projects/BVH']
RENASCOR_IDS = ['epistemon']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'BVH Epistemon: renaissance French texts with person/place/people mentions, partial #pers_/#loc_ keys'
OPTIONS = {'opaque_keys': True, 'type_map': {'allegorie': 'person'}}

_GLOBAL = re.compile(r'^(pers[hfm]?|loc[hfm]?)_')
_PREFIX_CLASS = {'persf': 'character', 'persm': 'mythological'}
_BIBLICAL = {'dieu', 'jesus', 'jesus christ', 'christ', 'jesu christ', 'saint esprit', 'adam', 'eve', 'moyse', 'moise', 'abraham', 'noe',
             'david', 'salomon', 'marie', 'sainct paul', 'saint paul', 'sainct pierre', 'satan', 'lucifer'}
_MYTHICAL = {'venus', 'jupiter', 'jupin', 'mars', 'apollon', 'minerve', 'junon', 'diane', 'bacchus', 'amour', 'cupidon', 'mercure',
             'neptune', 'pluton', 'saturne', 'hercule', 'les muses', 'muses', 'mercurius', 'jove', 'phebus', 'pallas', 'vulcan', 'vulcain',
             'cerberus', 'cerbere', 'proserpine', 'junon', 'aeole', 'eole', 'pan', 'priape', 'hymenee', 'fortune'}
_FICTIONAL = {'pantagruel', 'panurge', 'gargantua', 'grandgousier', 'frere jean', 'picrochole', 'gargamelle', 'epistemon', 'carpalim',
              'eusthenes', 'ponocrates', 'gymnaste', 'badebec', 'pantagruelion', 'xenomanes', 'basche', 'homenaz', 'hippothadee'}


def _fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize('NFKD', (s or '').casefold())
    return re.sub(r'[^a-z ]+', '', ''.join(ch for ch in s if not unicodedata.combining(ch))).strip()


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        """Year(s) of the original imprint (sourceDesc/msDesc/.../monogr/imprint/date). The header also holds the digital
        edition's own dates and modern bibliography (1976, 1999 ...) which the generic reader would mix in."""
        ys = []
        for imp in root.iter('{*}imprint'):
            for d in imp.iter('{*}date'):
                rg = C.date_range(d, 1400, 1800)
                if rg:
                    ys.append(rg)
        if not ys:                      # letters / manuscripts: dates under msDesc history or origin
            for el in root.iter():
                if C.ln(el.tag) in ('origDate', 'creation') or (C.ln(el.tag) == 'date' and el.get('when')
                                                                  and any(C.ln(a.tag) == 'history' for a in el.iterancestors())):
                    rg = C.date_range(el, 1400, 1800)
                    if rg:
                        ys.append(rg)
        if ys:
            return C.DocDate(min(a for a, _ in ys), max(b for _, b in ys), 'header')
        d = C.header_dates(root)
        if d.known and d.year_to <= 1800:
            return d
        return C.DocDate(1525, 1625, 'collection')

    def scan(self, root, rec, ctx, counters):
        for el in root.iter('{*}name'):
            if el.get('type') is None and not any(C.ln(a.tag) == 'teiHeader' for a in el.iterancestors()):
                el.set('type', 'person')
        return super().scan(root, rec, ctx, counters)

    def adjust(self, m, hit):
        if m.kind == 'org' and hit.name == 'orgName':
            m.person_class = 'people'
        elif m.kind == 'person' and not m.person_class:
            for tok in (m.ref_raw or '').split():
                mm = re.match(r'^#?(pers[hfm]?)_', tok)
                if mm and mm.group(1) in _PREFIX_CLASS:
                    m.person_class = _PREFIX_CLASS[mm.group(1)]
            if not m.person_class:
                f = _fold(m.surface)
                if f in _FICTIONAL:
                    m.person_class = 'character'
                elif f in _MYTHICAL:
                    m.person_class = 'mythological'
                elif f in _BIBLICAL:
                    m.person_class = 'biblical'
        if m.surface:
            m.surface = re.sub(r'\s*\[\d+\]\s*', '', m.surface)
            m.surface = re.sub(r'(?<=\w)-\s+(?=[a-zà-ÿ])', '', m.surface)
        if m.local_key and m.in_register is False:                  # opaque key produced by resolve_refs
            ident = m.local_key.split(':', 1)[1]
            if m.zone == 'hdr':
                m.local_key, m.in_register = None, None                # file-local header ids: not stable
            elif not _GLOBAL.match(ident):
                m.local_key = C.make_key(SLUG, ident, m.doc_id)         # document-scoped id
        return m


EXTRACTOR = Extractor()
