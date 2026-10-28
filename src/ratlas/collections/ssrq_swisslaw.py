"""SSRQ - Sammlung Schweizerischer Rechtsquellen (Collection of Swiss Law Sources), editio-data (de_dh/ssrq.zip).

5,596 TEI files, one per document or witness (volumes FR_I_2_8, ZH_NF_*, SG_III_4, NE, VD), 1300s-1800s, German,
French and Latin. persName / placeName / orgName mark the persons, places and bodies of the transcribed running text
and carry @ref with an SSRQ database id (per010051a.01, loc001060.04, org004722.02). The SSRQ register is not in the
harvest (it lives in the SSRQ database), so the ids are stable local ids without register: opaque keys
'ssrq_swisslaw:<id>' (in_register False), no authority ids and no coordinates. Header persNames (editors, seal
holders, respStmt) are modern or editorial and are not read.

Dating: header origDate (history/origin). About 15% of the files lie outside 1450-1700 (flagged in_period False,
kept); 20 intro files have no date (in_period None).
v2 (audit points 11 and 12):
  * entity key: the SSRQ id suffix ``.NN`` is a name-form variant of the same entity (loc007693 Estavayer has 14 variant ids:
    Estavayer, Staffis, Staviacum...): the key is the id WITHOUT it ('ssrq_swisslaw:loc007693'; 10,697 person keys become
    8,689, 5,556 place keys 3,283); the full written id stays in ``ref_raw``, the form in ``surface``;
  * dating: a file holds several ``origDate`` (the original and later witnesses or copies). The document is dated by the principal
    one: ``msItem/filiation[@type=original]/origDate``, else those of ``history/origin``; the dates of copies are not merged
    in (SSRQ-SG-III_4-16-1: original 1395 and a copy of 1619 gave 1395-1619, doc_year 1507). A range wider than 20 years has no
    doc_year;
  * the SSRQ register (persons per..., places loc..., organisations org...) lives in the SSRQ database and was NOT harvested (owner
    decision of 2026-10-16: leave as is, no request to the editors): names, GND / HLS ids and coordinates stay unavailable.
Licence: CC-BY-NC-SA-4.0 (licence element of the TEI headers, LICENSE_TIERS SSRQ_SwissLaw), T2.
The catalogue entry (1500-1644, 7 texts) understates this folder; renascor id 'ssrq-swisslaw'.
"""
SLUG = 'ssrq_swisslaw'
COLLS = ['de_dh/ssrq.zip']
RENASCOR_IDS = ['ssrq-swisslaw']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Swiss legal sources: tagged persons, places and organisations with SSRQ register ids'
OPTIONS = {'opaque_keys': True}

import re  # noqa: E402

from .. import common as C  # noqa: E402
from .. import helpers_batch1 as H  # noqa: E402

_VARIANT = re.compile(r'\.\d+$')


def principal_dates(root):
    """DocDate of the document itself: original filiation, else history/origin, else None."""
    orig, origin = [], []
    for el in root.iter():
        if C.ln(el.tag) != 'origDate':
            continue
        rg = C.date_range(el, 1000, 2100)
        if not rg:
            continue
        anc = [(C.ln(a.tag), a.get('type')) for a in el.iterancestors()]
        if ('filiation', 'original') in anc:
            orig.append(rg)
        elif any(n == 'origin' for n, _t in anc):
            origin.append(rg)
    for grp in (orig, origin):
        if grp:
            return C.DocDate(min(r[0] for r in grp), max(r[1] for r in grp), 'header')
    return None


class Extractor(H.BaseTEI):
    def doc_date(self, root, rec, ctx):
        d = principal_dates(root)
        return d if d is not None else super().doc_date(root, rec, ctx)

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is not None and m.local_key and not m.in_register:
            m.local_key = _VARIANT.sub('', m.local_key)
        return m


EXTRACTOR = Extractor()
