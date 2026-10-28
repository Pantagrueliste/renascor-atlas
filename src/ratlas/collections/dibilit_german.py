"""DiBiLit-Korpus (BBAW / Deutsches Textarchiv, texts of the 'Digitale Bibliothek' / Zeno.org), DTABf TEI.

Only the 47 texts of the Renascor entry (1517-1696, 2.14M words) are read: the early modern originals retained by
the harvest (drama 28, roman 12, erzaehlungen 3, fabel 2, prosa 2). The remaining ~2,030 files are 18th-20th c.
literature (and 2,098 metadata duplicates) and, as the harvest does, modernised versions are not counted.
What the TEI tags:
  * no entity tags in the running text; the header holds the author as persName/@ref=GND (titleStmt/author, role
    'author', zone hdr); the same author repeated in sourceDesc/biblFull is dropped; modern editors are never read;
  * one listPerson entry per text (the author, with birth/death/GND): a file-local entity; the author mention is
    linked to it through the GND id;
  * dating: the DTA first-publication date is 19th-20th c. edition data; the harvest's own year of the text (author's
    period, '~1600') is used -> date_source 'collection' (table SELECTION_YEARS from RQ/rebuild/
    remaining_missing_source_selection_2026-06-13.tsv, live_year);
  * dramatis personae are plain <castItem> strings, not tagged persons: not extracted.
v2 (issues 152-153): the document date is the first-publication year of the header (``date type=firstPublication subtype=clariah``,
else the untyped firstPublication, date_source 'header'); the author-period table above is only the fallback (25 of the 47 texts differed
by 5 years or more: Brant's Tugent Spyl 1518, not 1554). Header-zone rows carry no context (it held the modern DTA / CLARIAH editors'
names); the author is keyed by GND, so a repeated author is one entity.
Licence: CC-BY-SA-4.0 (LICENSE.md of the corpus; LICENSE_TIERS DiBiLit_German, T1; TextGrid metadata CC-BY-3.0).
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'dibilit_german'
COLLS = ['dibilit/data']
RENASCOR_IDS = ['dibilit-german']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Authors (GND) of 47 early modern German literary texts in the DiBiLit corpus'
OPTIONS = {}

SELECTION_YEARS = {
    'dibilit/data/drama/ayrer_sidea_1865.txt.xml': 1600,
    'dibilit/data/drama/ayrer_theseus_1865.txt.xml': 1600,
    'dibilit/data/drama/ayrer_wie-einem-weib-jhr-eygner-mann-verkuppelt-wird_1865.txt.xml': 1600,
    'dibilit/data/roman/beer_das-narrenspital_1957.txt.xml': 1682,
    'dibilit/data/roman/beer_die-kurzweiligen-sommer-taege_1963.txt.xml': 1682,
    'dibilit/data/roman/beer_teutsche-winter-naechte_1963.txt.xml': 1682,
    'dibilit/data/drama/brant_tugent-spyl-hercules-am-scheideweg_1968.txt.xml': 1554,
    'dibilit/data/roman/fischart_geschichtklitterung_1963.txt.xml': 1575,
    'dibilit/data/drama/frischlin_fraw-wendelgard_1908.txt.xml': 1579,
    'dibilit/data/drama/gengenbach_der-gauchmatt_1856.txt.xml': 1517,
    'dibilit/data/drama/gengenbach_die-zehn-alter-dieser-welt_1856.txt.xml': 1517,
    'dibilit/data/erzaehlungen/grimmelshausen_der-seltzame-springinsfeld_1921.txt.xml': 1670,
    'dibilit/data/prosa/harsdoerffer_der-grosse-schauplatz-jaemmerlicher-mordgeschichte_1656.txt.xml': 1650,
    'dibilit/data/prosa/harsdoerffer_der-grosse-schauplatz-lust--und-lehrreicher-geschichte_1664.txt.xml': 1650,
    'dibilit/data/drama/opitz_judith_1933.txt.xml': 1624,
    'dibilit/data/drama/rebhun_susanna_1967.txt.xml': 1535,
    'dibilit/data/roman/reuter_schelmuffskys-kuriose-und-sehr-gefaehrliche-reisebeschreibung-zu-wasser-und-lande_1979.txt.xml': 1696,
    'dibilit/data/drama/rist_das-friedejauchtzende-teutschland_1972.txt.xml': 1645,
    'dibilit/data/drama/rist_das-friedewuenschende-teutschland_1972.txt.xml': 1645,
    'dibilit/data/drama/rist_irenaromachia_1972.txt.xml': 1645,
    'dibilit/data/drama/rist_perseus_1972.txt.xml': 1645,
    'dibilit/data/erzaehlungen/rollenhagen_froschmeuseler01_1876.txt.xml': 1595,
    'dibilit/data/erzaehlungen/rollenhagen_froschmeuseler02_1876.txt.xml': 1595,
    'dibilit/data/drama/rollenhagen_vom-reichen-mann-und-armen-lazaro_1929.txt.xml': 1595,
    'dibilit/data/drama/sachs_das-heyss-eysen_1875.txt.xml': 1553,
    'dibilit/data/drama/sachs_das-narren-schneyden_1870.txt.xml': 1553,
    'dibilit/data/drama/sachs_der-farendt-schuler-mit-dem-teuffel-pannen_1875.txt.xml': 1553,
    'dibilit/data/drama/sachs_der-hoernen-sewfriedt_1880.txt.xml': 1553,
    'dibilit/data/drama/sachs_der-kraemerskorb_1888.txt.xml': 1553,
    'dibilit/data/drama/sachs_der-schwanger-pawer_1892.txt.xml': 1553,
    'dibilit/data/drama/sachs_die-alt-verschlagen-kuplerin-mit-dem-thumbherrn_1888.txt.xml': 1553,
    'dibilit/data/drama/sachs_die-gedultig-und-gehorsam-marggraefin-griselda_1870.txt.xml': 1553,
    'dibilit/data/drama/sachs_lucretia_1879.txt.xml': 1553,
    'dibilit/data/drama/sachs_von-der-strengen-lieb-herrn-tristrant-mit-der-schoenen-koenigin-isalden_1879.txt.xml': 1553,
    'dibilit/data/fabel/waldis_esopus01_1882.txt.xml': 1548,
    'dibilit/data/fabel/waldis_esopus02_1882.txt.xml': 1548,
    'dibilit/data/drama/weise_der-niederlaendische-bauer_1969.txt.xml': 1685,
    'dibilit/data/roman/weise_die-drei-aergsten-erznarren-in-der-ganzen-welt_1878.txt.xml': 1685,
    'dibilit/data/drama/weise_koenig-wentzel_1971.txt.xml': 1685,
    'dibilit/data/roman/weise_kurtzer-bericht-vom-politischen-naescher_1680.txt.xml': 1685,
    'dibilit/data/drama/weise_vom-verfolgten-lateiner_1970.txt.xml': 1685,
    'dibilit/data/drama/wickram_das-narrengiessen_1903.txt.xml': 1555,
    'dibilit/data/roman/wickram_der-goldtfaden_1903.txt.xml': 1555,
    'dibilit/data/roman/wickram_der-jungen-knaben-spiegel_1903.txt.xml': 1555,
    'dibilit/data/roman/wickram_gabriotto-und-reinhart_1903.txt.xml': 1555,
    'dibilit/data/roman/wickram_von-guten-und-boesen-nachbarn_1903.txt.xml': 1555,
    'dibilit/data/roman/zesen_simson_1970.txt.xml': 1679,
}


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.path in SELECTION_YEARS]

    def doc_date(self, root, rec, ctx):
        fp = {}
        for el in root.iter():
            if C.ln(el.tag) == 'date' and el.get('type') == 'firstPublication':
                y = C._attr_year(el.get('when')) or (int(C.norm(C.surface_text(el))) if re.fullmatch(r'\d{4}', C.norm(C.surface_text(el))) else None)
                if y:
                    fp.setdefault(el.get('subtype') or '', y)
        y = fp.get('clariah') or fp.get('') or next(iter(fp.values()), None)
        if y and 1450 <= y <= 1750:
            return C.DocDate(y, y, 'header')
        y = SELECTION_YEARS.get(rec.path)
        return C.DocDate(y, y, 'collection') if y else C.DocDate()

    def adjust(self, m, hit):
        if 'sourceDesc' in hit.ancestors:       # the author is repeated in biblFull (source of the digital text)
            return None
        m.context = None                        # header rows: the neighbouring text is the modern editors' names
        return m

    def extract_root(self, root, rec, ctx):
        mentions, ents = super().extract_root(root, rec, ctx)
        by_gnd = {e.auth.get('gnd'): e for e in ents if e.auth.get('gnd')}
        for m in mentions:
            e = by_gnd.get(m.auth.get('gnd')) if m.auth.get('gnd') else None
            if e is not None and e.kind == m.kind and not m.local_key:
                m.local_key, m.in_register = e.local_key, True
        return mentions, ents


EXTRACTOR = Extractor()
