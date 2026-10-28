"""Dasypodius, Dictionarium Latinogermanicum (Strasbourg 1535), HAB edition (ed. J. West), 5 part files.

Conventions of this edition:
  * <rs type="person|place|corporate|animal" ref="#slug"> marks the dictionary headwords that are proper names
    (ancient persons, mythological figures, places, peoples); the entities are the ancient world as glossed by a
    1535 dictionary, not early modern contemporaries. 'corporate' (peoples, tribes, communities) is mapped to org;
    'animal' (6 tags) is not a person/place/org and is skipped; typos ('peson', 'plac') are mapped;
  * the slug in @ref has no register in the harvest (the TEI header is an external entity that is not harvested):
    OPTIONS opaque_keys gives 'slug:<id>' with in_register False (stable inside the edition, not merged);
  * the authority comes from the <w lemmaRef="http://en.wikipedia.org/wiki/..."> inside the tag (English
    Wikipedia, Perseus, Livius): stored as 'wikipedia:en:<title>' / 'url:...' in auth.other (no coordinates,
    no Wikidata in the source; a later stage can map Wikipedia titles to Wikidata/Pleiades);
  * the entity tagging stops after the second file: the A-B entries and the front matter are untagged (0 tags).
  * dating: single 1535 print: header/imprint year, else default_years (1535).
Licence: CC-BY-SA-3.0 (LICENSE_TIERS, T1; HAB diglib rights).
"""
from __future__ import annotations

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'dasypodius_dictionarium_latinogermanicum'
COLLS = ['renascor_2026-10/dasypodius-dictionarium-latinogermanicum']
RENASCOR_IDS = ['dasypodius-dictionarium-latinogermanicum']
LICENCE = 'CC-BY-SA-3.0'
TIER = 'T1'
DESCRIPTION = 'Proper-name headwords (ancient persons, myths, places, peoples) of a 1535 Latin-German dictionary'
OPTIONS = {'default_years': (1535, 1535), 'opaque_keys': True,
           'type_map': {'peson': 'person', 'peron': 'person', 'perosn': 'person', 'plac': 'place',
                        'corporate': 'org'}}


class Extractor(GenericTEI):
    def adjust(self, m, hit):
        vals = [w.get('lemmaRef') for w in hit.el.iter() if C.ln(w.tag) == 'w' and w.get('lemmaRef')]
        if vals:
            m.auth = C.merge_auth(m.auth, C.parse_auth(vals))
            m.ref_raw = ' '.join(x for x in [m.ref_raw] + vals[:1] if x) or None
        return m


EXTRACTOR = Extractor()
