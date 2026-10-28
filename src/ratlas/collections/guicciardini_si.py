"""Guicciardini, Storia d'Italia: semi-diplomatic transcription of the 8 drafts of the exordium (ENS Lyon / Liege), 1 file.

Thematic annotation of the argument: <rs type="concept|place|person|org|population" ref="label"> (1,087 tags: 894
concept, 97 place, 35 person, 34 org, 25 population). Only person / place / org are read; concept (a thematic term
index) and population (peoples) are not entities. @ref is a normalised label ('Italia', 'Lorenzo dei Medici', 'Impero
romano'), not an identifier and not a register id: the label is used as an opaque key
('guicciardini_si:Lorenzo_dei_Medici', in_register False), spaces turned into underscores, auth left empty (the
generic splitter would otherwise read 'Lorenzo dei Medici' as three tokens). 'io' (the narrator) is tagged person by the
editors and kept. No register, no coordinates, no entities. msIdentifier/settlement|country (archive location of the
manuscript) in the header are not mentions and are dropped.
Dating: the drafts are 1537-1540 (catalogue entry); the 14 sections carry no date: default_years (1537, 1540),
date_source 'collection'.
Licence: CC BY-NC-SA 4.0 (TEI availability and catalogue); LICENSE_TIERS Renascor_guicciardini-storia-italia T2.
"""
from __future__ import annotations

import re

from .. import common as C
from ..model import empty_auth
from .generic_tei import GenericTEI

SLUG = 'guicciardini_si'
COLLS = ['originals_2026-10-15/oct-binary/oct__guicciardini-storia-italia']
RENASCOR_IDS = ['guicciardini-storia-italia']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Guicciardini Storia d Italia, drafts of the exordium: places, persons, institutions as label keys'
OPTIONS = {'default_years': (1537, 1540)}


class Extractor(GenericTEI):
    def doc_date(self, root, rec, ctx):         # header dates are citations (e.g. 1919 editions), not the drafts
        return C.DocDate(1537, 1540, 'collection')

    def adjust(self, m, hit):
        if 'msIdentifier' in hit.ancestors:
            return None
        label = C.norm(hit.refs.get('ref') or '')
        m.auth = empty_auth()
        if label:
            m.local_key = C.make_key(SLUG, re.sub(r'\s+', '_', label))
            m.in_register = False
        else:
            m.local_key, m.in_register = None, None
        return m


EXTRACTOR = Extractor()
