"""Interpretation rules of the document table (stage 'build' of docs_table.py).

Input: the raw header metadata of docs_extract.py (one JSON per document), the keys row (place mentions, years,
licence, tier) and the Renascor catalogue entry. Output: one row of documents.parquet.

Everything collection-specific is in this file:
  DOC_TYPE / doc_type_for   document type per collection (+ refinements by structure and title)
  URLS                      URL builders per collection. Every pattern was checked on 1-3 real documents of the
                            edition (log: docs/DOCUMENT_URLS.md); a collection without a verified pattern gets no
                            `url` (the catalogue's `edition_url` is always filled)
  synth_title               readable titles where the header only has an identifier (CASE1000, COSUIZA-0001...)

URL verification log (Oct 2026, user agent Renascor-Atlas/0.1, robots.txt read first, 1 request per second, no
crawling: 1-3 documents per edition). 200 = the page exists; "pattern" = built from the edition's own links.
  bullinger          header idno https://www.bullinger-digital.ch/letter/1 (200, redirects to /file1)
  betamasaheft_geez  https://betamasaheft.eu/LIT4969EdictGala (200, redirects to /works/LIT4969EdictGala/main)
  biblioteca_italiana http://www.bibliotecaitaliana.it/testo/bibit000316 (200)
  casebooks          live site answers 404 on every page (checked CASE1000 and the home page); the Internet Archive
                     holds /cases/CASE1000 (snapshot 2021-04-18); other cases not checked (archive.org rate limit)
  celt               https://celt.ucc.ie/published/E900006/ (200)
  cidtc              https://editions.al.uw.edu.pl/exist/apps/dantiscus/letters/IDL0001.xml (200; TEI Publisher)
  dipko / qhod / hearth_tax / urfehde  https://gams.uni-graz.at/<PID of the header> (200 on 4 documents)
  dta_1600 / sandrart header idno https://www.deutschestextarchiv.de/<id> (200, redirects to /book/show/<id>)
  ferdinand_i        https://ferdinand-korrespondenz.acdh.oeaw.ac.at/A001.html (200)
  kaiserin           https://kaiserin-eleonora.oeaw.ac.at/<file stem>.html (200 on two folders)
  koenigsfelden      https://koenigsfelden.sources-online.org/data/docs/<stem>.xml (200; a made-up id gives 404)
  st_galler_missiven https://missiven.stadtarchiv.ch/missiven/<id> (200; pattern from the app's own collection API)
  map_of_early_modern_london https://mapoflondon.uvic.ca/AMWE1.htm and stow_1598_FARR2.htm (200)
  old_bailey         sessions papers https://www.oldbaileyonline.org/record/16790827 (Internet Archive snapshot of
                     2025-09-24; the site itself answers 403 to automated requests). Ordinary's Accounts
                     (OAyyyymmdd): no snapshot found, pattern not verified, so no url
  postscriptum       http://teitok.clul.ul.pt/postscriptum/index.php?action=file&cid=PS4012 and PSCR8317 (200, title
                     of the page = title of the letter)
  h15corpus / letradas / cosuiza  TEITOK index.php?action=file&cid=<file stem> (200; page titles checked)
  reed               https://ereed.org/records/<xml:id>/ (200 on 2 records)
  religionsfrieden   https://tueditions.ulb.tu-darmstadt.de/v/pa000001-0025 (200)
  ssrq_swisslaw      https://editio.ssrq-online.ch/api/v1/documents/<id>/xml (200; the HTML view route was not found)
  HAB edoc (tagebuecher ed000228, fruchtbringende ed000213, luene ed000248, rubens ed000083, karlstadt ed000216 via
                     texts/NNN/NNN_transcript.xml (METS), paesi ed000145, dasypodius ed000008)  200 on one file each
  epistemon / freem_bvh  header bibl/ref http://www.bvh.univ-tours.fr/Consult/index.asp?numfiche=71 (200, notice)
  GitHub blobs (disco, croala, corpus17, ethica, lettres_calvin, poliziano, rup, suriano)  200 on one file each
  no url: abacus, bdpn, camena_*, cavriana (repo file names changed), ckcc (single page app, letter id format not
          verifiable), controversia (tueditions id format unknown), corpus_corporum, elan_crenum, emmo_folger (page
          route not found), freem_other, germanc, gongora_* (OBVIL page 404), grand_tour, hainhofer, leibniz,
          selbstzeugnisse, sozinianische_briefwechsel (Angular app, guid unknown)
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional

CATALOGUE_BASE = 'https://pantagrueliste.github.io/renascor/#'
PERIOD = (1450, 1700)


# ======================================================================================================================
# small helpers
# ======================================================================================================================
_MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def human_date(d: Optional[str]) -> str:
    """'1596-12-22' -> '22 Dec 1596'; '1596-12' -> 'Dec 1596'; 'a/b' -> 'a - b' humanised; anything else as is."""
    if not d:
        return ''
    if '/' in d:
        a, b = d.split('/', 1)
        if b == '..':
            return f'after {human_date(a)}'
        if a == '..':
            return f'before {human_date(b)}'
        ha, hb = human_date(a), human_date(b)
        return f'{ha} - {hb}' if ha != hb else ha
    m = re.fullmatch(r'(-?\d{4})(?:-(\d{2}))?(?:-(\d{2}))?', d)
    if not m:
        return d
    y, mo, da = m.groups()
    if mo and 1 <= int(mo) <= 12:
        return (f'{int(da)} ' if da else '') + f'{_MON[int(mo) - 1]} {y}'
    return y


def fold(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', (s or '').lower()) if unicodedata.category(c) != 'Mn')


def trim(s: str, n: int = 160) -> str:
    s = re.sub(r'\s+', ' ', s or '').strip()
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    sp = cut.rfind(' ')
    if sp > n * 0.6:
        cut = cut[:sp]
    return cut.rstrip(' ,;:-') + '…'


def _dedupe_commas(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    out = []
    for p in [x.strip() for x in s.split(',')]:
        if p and p not in out:
            out.append(p)
    return ', '.join(out)


def _int(v) -> Optional[int]:
    try:
        if v is None or v != v:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


_ISO2 = {'deu': 'de', 'ger': 'de', 'eng': 'en', 'lat': 'la', 'ita': 'it', 'fra': 'fr', 'fre': 'fr', 'spa': 'es',
         'nld': 'nl', 'dut': 'nl', 'por': 'pt', 'ces': 'cs', 'cze': 'cs', 'pol': 'pl', 'hrv': 'hr', 'hun': 'hu',
         'swe': 'sv', 'dan': 'da', 'ell': 'el', 'heb': 'he', 'gle': 'ga', 'cym': 'cy', 'cat': 'ca', 'rus': 'ru',
         'tur': 'tr', 'ara': 'ar', 'slv': 'sl', 'srp': 'sr', 'eus': 'eu'}


_LANG_NAMES = {'Latin': 'la', 'German': 'de', 'English': 'en', 'French': 'fr', 'Spanish': 'es', 'Italian': 'it',
               'Dutch': 'nl', 'Hungarian': 'hu', 'Greek': 'el', 'Portuguese': 'pt', 'Polish': 'pl', 'Hebrew': 'he',
               'Czech': 'cs', 'Swedish': 'sv', 'Danish': 'da', 'Irish': 'ga', 'Catalan': 'ca', 'Russian': 'ru'}


_LANG_WORDS = {'german': 'de', 'deutsch': 'de', 'latein': 'la', 'latin': 'la', 'english': 'en', 'french': 'fr',
               'italian': 'it', 'spanish': 'es', 'dutch': 'nl', 'niederländisch': 'nl', 'niederlaendisch': 'nl',
               'französisch': 'fr', 'italienisch': 'it', 'englisch': 'en', 'spanisch': 'es', 'griechisch': 'el',
               'frühneuhochdeutsch': 'de', 'mittelhochdeutsch': 'gmh', 'mittelniederdeutsch': 'gml'}

# Collections whose header has no language and whose text is mostly in one language although the place mentions
# fall in passages in another (the diaries of Christian II of Anhalt: German with French and Latin passages).
_DEFAULT_LANG = {'tagebuecher_christian_ii_anhalt': 'de'}


def norm_lang(code: Optional[str]) -> Optional[str]:
    if not code or not isinstance(code, str):
        return None
    c = code.strip().lower().replace('_', '-')
    if c in ('?', 'und', 'mul', 'zxx', ''):
        return None
    base = c.split('-')[0]
    if len(base) == 2:
        return base
    return _ISO2.get(base) or _LANG_WORDS.get(base) or base


_TECH_TITLE = re.compile(r'^(title|untitled|transkription \w+|[a-z]{2,10}[-_ ]?\d+[\w.\-]*|case ?\d+|letter \d+\.\d+.*|'
                         r'stadtasg missive, nr\. \d+|urkunde \d+|hs\. ?\d+)$', re.I)


def technical(t: str, doc_id: str = '') -> bool:
    """True for a title that is only an identifier."""
    t = (t or '').strip()
    return (not t) or t == doc_id or bool(_TECH_TITLE.match(t))


_SUFFIX = re.compile(r'[,.;:]?\s*(digitale ausgabe|edici[oó]n electr[oó]nica\.?|versio electronica|edition [eé]lectronique|'
                     r'an electronic edition|machine-readable text|edition numérique)\s*\.?$', re.I)


def clean_title(t: str) -> str:
    t = re.sub(r'\s+', ' ', t or '').strip()
    t = _SUFFIX.sub('', t).strip()
    # a title cut inside a parenthesis by the markup ("... ernannt (28." with the date as a child element)
    m = re.search(r'\s*\(([^()]{0,24})$', t)
    if m and t.count('(') > t.count(')'):
        t = t[: m.start()]
    return t.strip(' ,;')


# ======================================================================================================================
# titles
# ======================================================================================================================
_SKIP_TYPES = {'sub', 'subtitle', 'gmd', 'short', 'num', 'volume', 'alt', 'abbr', 'incipit', 'series', 's', 'j',
               'supplementary', 'part', 'titre_court', 'temizorig'}
_BOILER_SUB = re.compile(r'e-ditiones edition|machine-readable|digital edition|edition numérique', re.I)
_MAIN_TYPES = ('main', 'uniform', 'file', 'a', 'm', 'titre_long')


def header_title(info: dict, doc_lang: Optional[str] = None) -> tuple[str, str]:
    """(title, where) from the header: titleStmt first (main titles, language-matched), then head / msItem /
    analytic / monogr / bibl titles. ('', '') if the header has none."""
    tl = info.get('titles') or []
    for where in ('titleStmt', 'head', 'msItem', 'analytic', 'monogr', 'bibl'):
        c = [t for t in tl if t[0] == where and t[3] and (t[1] or '').lower() not in _SKIP_TYPES]
        if where == 'head':
            c = [t for t in c if (t[1] or '').lower() not in ('abstract', 'correspondence')]
        if not c:
            continue
        if len({t[2] for t in c if t[2]}) > 1 and doc_lang:
            m = [t for t in c if norm_lang(t[2]) == doc_lang]
            if m:
                c = m
        main = [t for t in c if (t[1] or '').lower() in _MAIN_TYPES] or c
        title = clean_title(main[0][3])
        subs = [t for t in tl if t[0] == where and (t[1] or '').lower() in ('sub', 'subtitle') and t[3]
                and not _BOILER_SUB.search(t[3])]
        if subs and (technical(title) or re.fullmatch(r'[A-Z]?\d+\.?', title)):
            title = clean_title(subs[0][3])
        elif subs and len(title) < 60:
            title = f'{title}: {clean_title(subs[0][3])}'
        if title:
            return title, where
    return '', ''


# ======================================================================================================================
# dates
# ======================================================================================================================
_CLASS_RANK = {'sent': 0, 'creation': 1, 'title': 2, 'source': 3, 'text': 4, 'other': 5, 'received': 6}
_PREC_RANK = {'day': 0, 'month': 1, 'year': 2, 'range': 3, '': 4}


def choose_date(info: dict, yf: Optional[int], yt: Optional[int]):
    """(date, precision, year_from, year_to, source, conflict) from the header candidates, kept consistent with the
    mention-level years (yf, yt) of the v1 tables when those exist (a candidate that does not overlap them is a
    publication or edition date and is dropped)."""
    cands = list(info.get('dates') or []) + list(info.get('text_dates') or [])
    ok = []
    conflict = False
    for c in cands:
        cls, iso, prec, txt, a, b = c
        if a is None and b is None:
            continue
        ya = a if a is not None else b
        yb = b if b is not None else a
        if yf is not None:
            lo, hi = yf - 1, (yt if yt is not None else yf) + 1
            if yb < lo or ya > hi:
                conflict = True
                continue
        ok.append(c)
    if ok:
        main = [c for c in ok if c[0] in ('sent', 'creation', 'title')]
        if main:      # the document's own date: the most precise of the header's own dating elements
            same = sorted(main, key=lambda c: (_PREC_RANK.get(c[2], 9), _CLASS_RANK.get(c[0], 9)))
        else:
            best = min(_CLASS_RANK.get(c[0], 9) for c in ok)
            same = sorted((c for c in ok if _CLASS_RANK.get(c[0], 9) == best), key=lambda c: _PREC_RANK.get(c[2], 9))
        cls, iso, prec, txt, a, b = same[0]
        ya = a if a is not None else b
        yb = b if b is not None else a
        if iso:
            return iso, prec, ya, yb, cls, False
        if ya == yb:
            return str(ya), 'year', ya, yb, cls, False
        return f'{ya}/{yb}', 'range', ya, yb, cls, False
    if yf is not None:
        if yt is None or yf == yt:
            return str(yf), 'year', yf, yf, 'mentions', conflict
        return f'{yf}/{yt}', 'range', yf, yt, 'mentions', conflict
    return None, None, None, None, None, False


def _iso_title(title: str):
    from .docs_extract import iso_from_text
    iso, prec = iso_from_text(title)
    return (iso, prec) if iso and prec in ('day', 'month') else (None, None)


# ======================================================================================================================
# persons and places
# ======================================================================================================================
def _clean_name(n: str) -> str:
    n = re.sub(r'\s+', ' ', n or '').strip(' ,;')
    n = re.sub(r'^\[|\]$', '', n)
    n = re.sub(r'\bAn[oó]nim[oa]\d+\b', 'Anónimo', n)
    return n


_PLACEHOLDER = re.compile(r'^\[?\s*(s\.\s?[nl]\.?|o\.\s?o\.?|n\.\s?n\.?|nn|\?|¿\?|sine (nomine|loco)|ohne (ort|name))\s*\]?$', re.I)


def names(lst) -> Optional[str]:
    out, seen = [], set()
    for n in lst or []:
        n = _clean_name(n)
        n = re.sub(r'¿\?$', '', n).strip()
        if not n or n.lower() in ('unknown', 'unknown recipient', 'unbekannt', '?', 'anonymous') or n in seen \
                or _PLACEHOLDER.match(n):
            continue
        seen.add(n)
        out.append(n)
    if not out:
        return None
    return ('; '.join(out[:6]) + (' et al.' if len(out) > 6 else ''))[:300]


_DE_MONTHS = 'Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember'
_DE_DAYS = 'Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag|Sonnabend'
_DATEWORD = re.compile(r'^(?:\d{1,2}\.?|\[?\d{4}\]?|und|bis|oder|[–\-/]|s\.\s?d\.?|Anfang|Mitte|Ende|vor|nach|um|ca\.?|zwischen|'
                       + _DE_MONTHS + '|' + _DE_DAYS + r'|\s)+$', re.I)


def _strip_date_tail(a: str):
    """('Heinrich Bullinger', None) from 'Heinrich Bullinger, 4. Oktober' and ('Bullinger', 'Chur') from
    'Bullinger, Chur, Montag, 23. April 1548': trailing date / weekday segments are not part of the addressee; the place
    that stands before a weekday is where the letter was written."""
    parts = [p.strip() for p in a.split(',')]
    weekday = False
    while len(parts) > 1 and _DATEWORD.match(parts[-1]):
        weekday = weekday or bool(re.search(_DE_DAYS, parts[-1], re.I))
        parts.pop()
    place = None
    if weekday and len(parts) > 1:
        place = parts.pop()
    return ', '.join(parts), place


def split_letter_title(slug: str, title: str):
    """(sender, sender place, addressee, addressee place) read from a letter title of an edition whose header has no
    correspDesc. Only for editions whose title format was checked on samples; None if the title does not fit."""
    t = re.sub(r'\s+', ' ', title or '').strip()
    strip = lambda x: re.sub(r'[\[\]]', '', x or '').strip() or None   # noqa: E731
    if slug == 'bullinger':
        tail_place = None
        mt = re.search(r',?\s*\[([^\]]*)\]\s*$', t)
        if mt:
            first = mt.group(1).split(',')[0].strip()
            if ',' in mt.group(1) and first and not re.search(r'\d', first):
                tail_place = first
            t = t[: mt.start()].strip()
        m = re.match(r'^(?P<s>.+?)\s+an\s+(?P<a>.+)$', t)
        if not m:
            return None
        s, a = m.group('s'), m.group('a')
        wd_place = None
        if ' / ' not in a:           # "A an B, 24. Februar 1570" / "A an B, Chur, Montag, 23. April 1548": no date, weekday or place in the addressee
            a, wd_place = _strip_date_tail(re.sub(r'[\[\]]', '', a))
        sp = ap = None
        if ' / ' in s:
            s, sp = [x.strip() for x in s.split(' / ', 1)][:2]
        if ' / ' in a:
            a, ap = [x.strip() for x in a.split(' / ', 1)][:2]
            ap = re.sub(r',.*$', '', ap)
        return strip(s), strip(sp or tail_place or wd_place), strip(a), strip(ap)
    if slug == 'sozinianische_briefwechsel':
        m = re.match(r'^(?P<s>.+?)(?: \((?P<sp>[^)]*)\))? an (?P<a>.+?)(?: \((?P<ap>[^)]*)\))?, [^,]+$', t)
        if m:
            return (m.group('s'), (m.group('sp') or '').rstrip('?') or None, m.group('a'),
                    (m.group('ap') or '').rstrip('?') or None)
        return None
    if slug == 'fruchtbringende_briefe':
        m = re.match(r'^(?P<s>.+?) an (?P<a>.+)$', t)
        return (m.group('s'), None, m.group('a'), None) if m else None
    if slug == 'cidtc':
        m = re.match(r'^(?P<s>.+?) to (?P<a>.+)$', t)
        return (m.group('s'), None, m.group('a'), None) if m else None
    if slug == 'emmo_folger':
        m = re.match(r'^Letter from (?P<s>.+?),\s*(?P<sp>[^,]+?),\s*to (?P<a>.+?),\s*(?P<ap>[^,]+?),\s*[^,]*\d{4}', t)
        if m:
            return m.group('s'), m.group('sp'), m.group('a'), m.group('ap')
        m = re.match(r'^Letter from (?P<s>.+?)\s+to (?P<a>[^,:]+)', t)
        return (m.group('s'), None, m.group('a'), None) if m else None
    return None


# ======================================================================================================================
# document types
# ======================================================================================================================
DOC_TYPE = {
    'abacus': 'book', 'bdpn': 'book', 'betamasaheft_geez': 'book', 'biblioteca_italiana': 'book',
    'bullinger': 'letter', 'camena_cera': 'book', 'camena_poemata': 'poem', 'camena_thesaurus': 'treatise',
    'casebooks': 'record', 'cavriana': 'letter', 'celt': 'other', 'cidtc': 'letter', 'ckcc_epistolarium': 'letter',
    'controversia_confessio': 'treatise', 'corpus17_french': 'book', 'corpus_corporum': 'other',
    'cosuiza': 'record', 'croala': 'poem', 'dasypodius_dictionarium_latinogermanicum': 'dictionary',
    'dipko_schwarzenhorn_1649': 'letter', 'disco_sonnets': 'poem', 'dta_1600': 'book', 'elan_crenum': 'book',
    'emmo_folger': 'letter', 'epistemon_bvh': 'book', 'ethica_complementoria': 'poem', 'ferdinand_i': 'letter',
    'freem_bvh': 'book', 'freem_other': 'book', 'fruchtbringende_briefe': 'letter', 'germanc': 'book',
    'gongora_epistolario': 'letter', 'gongora_soledades': 'poem', 'grand_tour': 'travelogue',
    'guicciardini_si': 'book', 'h15corpus': 'letter', 'hainhofer_reisen': 'travelogue', 'hearth_tax': 'record',
    'kaiserin_eleonora_magdalena': 'letter', 'karlstadt_kritische_gesamtausgabe': 'treatise',
    'koenigsfelden': 'act', 'leibniz_reihe_viii': 'treatise', 'letradas': 'record', 'lettres_calvin': 'letter',
    'luene_nonnenbriefe': 'letter', 'map_of_early_modern_london': 'record', 'old_bailey': 'record',
    'paesi_novamente': 'book', 'poliziano_sappho': 'annotation', 'postscriptum': 'letter', 'qhod_sema': 'letter',
    'reed': 'record', 'rejestr_poznan': 'record', 'religionsfrieden_quellenedition': 'act', 'rubens_online': 'act',
    'rup': 'poem', 'sandrart': 'book', 'selbstzeugnisse_hab_diarium': 'diary', 'selbstzeugnisse_hab_sz2': 'diary',
    'sozinianische_briefwechsel': 'letter', 'ssrq_swisslaw': 'act', 'st_galler_missiven': 'letter',
    'suriano': 'letter', 'tagebuecher_christian_ii_anhalt': 'diary', 'urfehdebuecher_basel_x': 'record',
}
_SERMON_RX = re.compile(r'predigt|sermon|homil', re.I)
_GERMANC = {'DRAM': 'play', 'HUMA': 'treatise', 'LEGA': 'act', 'NARR': 'book', 'NEWS': 'periodical',
            'SCIE': 'treatise', 'SERM': 'sermon'}
_LETTER_BY_SLUG = {
    'karlstadt_kritische_gesamtausgabe': re.compile(r'\b(Brief|Schreiben)\b|\ban\s+[A-ZÄÖÜ]'),
    'rubens_online': re.compile(r'\b(Brief|Schreiben|schreibt|schrieb)\b|\bwendet sich an\b'),
    'corpus_corporum': re.compile(r'^Ad\s|[Ee]pistol|[Ee]pistul'),
    'croala': re.compile(r'^Ad\s|[Ee]pistol|[Ee]pistul'),
    'cosuiza': re.compile(r'^(Carta|Billete|Epístola)\b'),
    'h15corpus': re.compile(r'^(Carta|Billete|Epístola)\b'),
    'letradas': re.compile(r'^(Carta|Billete|Epístola)\b'),
    'epistemon_bvh': re.compile(r'\blettres?\b|épître|epistol', re.I),
    'freem_bvh': re.compile(r'\blettres?\b|épître|epistol', re.I),
}
_MIXED_LETTER = set(_LETTER_BY_SLUG)


def doc_type_for(slug: str, info: dict, title: str, doc_id: str, stem: str) -> str:
    base = DOC_TYPE.get(slug, 'other')
    st = info.get('struct') or {}
    if slug == 'germanc':
        m = re.match(r'([A-Z]{4})_', stem)
        return _GERMANC.get(m.group(1), base) if m else base
    if slug == 'koenigsfelden':
        return 'charter' if doc_id.startswith('U') else 'act'
    if slug == 'ckcc_epistolarium' and info.get('ckcc_volume'):
        return 'book'
    if slug in ('epistemon_bvh', 'freem_bvh') and re.match(r'\[?Arr[êe]t', title or ''):
        return 'act'
    if slug in ('qhod_sema', 'dipko_schwarzenhorn_1649') and re.search(r'relation|report|bericht', title or '', re.I):
        return 'report'
    play = st.get('sp', 0) >= 10 or st.get('cast', 0) > 0
    verse = st.get('l', 0) > 3 * max(st.get('p', 0), 1)
    if slug in ('corpus17_french', 'dta_1600', 'sandrart', 'corpus_corporum', 'croala') and play:
        return 'play'
    if slug == 'dta_1600' and _SERMON_RX.search(title or ''):
        return 'sermon'
    if slug in _MIXED_LETTER:
        if info.get('senders') or info.get('addressees'):
            return 'letter'
        rx = _LETTER_BY_SLUG[slug]
        if (rx.search(title or '') or (slug in ('croala', 'corpus_corporum') and re.search(r'epist', doc_id, re.I))) \
                and not verse:
            return 'letter'
        if verse and slug in ('corpus_corporum', 'croala'):
            return 'poem'
    return base


# ======================================================================================================================
# URLs
# ======================================================================================================================
class Ctx:
    """What a URL builder sees of one document."""

    def __init__(self, slug, source_path, doc_id, info):
        self.slug, self.source_path, self.doc_id, self.info = slug, source_path, doc_id, info
        self.member = source_path.split('!', 1)[1] if '!' in source_path else source_path
        self.fname = self.member.rsplit('/', 1)[-1]
        self.stem = re.sub(r'\.(xml|tei|txt)$', '', self.fname)
        self.urls = [u for _, u in (info.get('urls') or [])]
        self.idnos = info.get('idnos') or []

    def hdr(self, rx: str) -> Optional[str]:
        r = re.compile(rx)
        return next((u for u in self.urls if r.search(u)), None)

    def idno(self, typ: str) -> Optional[str]:
        return next((v for _, t, v in self.idnos if (t or '').lower() == typ.lower()), None)

    def after(self, marker: str) -> str:
        """Path inside the repository: the member path after ``marker``."""
        i = self.source_path.find(marker)
        return self.source_path[i + len(marker):] if i >= 0 else self.fname


def _gh(repo: str, branch: str, path: str) -> str:
    from urllib.parse import quote
    return f'https://github.com/{repo}/blob/{branch}/' + quote(path, safe='/')


def _hab(ed: str, rel: str):
    return f'http://diglib.hab.de/edoc/{ed}/{rel}', 'source-xml'


def _teitok(base: str, cid: str):
    return f'{base}index.php?action=file&cid={cid}', 'edition'


def _gams(c: Ctx):
    pid = c.idno('PID')
    return (f'https://gams.uni-graz.at/{pid}', 'edition') if pid and pid.startswith('o:') else (None, None)


def _bullinger(c: Ctx):
    u = c.hdr(r'^https://www\.bullinger-digital\.ch/letter/\d+$')
    return (u, 'edition') if u else (None, None)


def _dta(c: Ctx):
    u = c.hdr(r'^https://www\.deutschestextarchiv\.de/[^/]+$')
    return (u, 'edition') if u else (None, None)


def _bvh(c: Ctx):
    u = c.hdr(r'bvh\.univ-tours\.fr/Consult/index\.asp\?numfiche=\d+')
    return (u, 'catalogue-record') if u else (None, None)


def _old_bailey(c: Ctx):
    if '/sessionsPapers/' in c.source_path and re.fullmatch(r'\d{8}', c.doc_id):
        return f'https://www.oldbaileyonline.org/record/{c.doc_id}', 'edition'
    return None, None   # ordinary's accounts: pattern not verifiable (the site refuses automated requests)


def _casebooks(c: Ctx):
    # The live site (casebooks.lib.cam.ac.uk) answers 404 for every page in October 2026. The case pages are in the
    # Internet Archive (pattern /cases/CASEnnnn; snapshot of 2021-04-18 checked on CASE1000).
    return f'https://web.archive.org/web/2021/https://casebooks.lib.cam.ac.uk/cases/{c.doc_id}', 'web-archive'


def _rubens(c: Ctx):
    rel = c.source_path.split('files/', 1)[-1]
    return _hab('ed000083', rel)


URLS: dict[str, Callable[[Ctx], tuple[Optional[str], Optional[str]]]] = {
    'betamasaheft_geez': lambda c: (f'https://betamasaheft.eu/{c.doc_id}', 'edition'),
    'biblioteca_italiana': lambda c: (f'http://www.bibliotecaitaliana.it/testo/{c.doc_id}', 'edition'),
    'bullinger': _bullinger,
    'casebooks': _casebooks,
    'celt': lambda c: (f'https://celt.ucc.ie/published/{c.doc_id}/', 'edition'),
    'cidtc': lambda c: (f'https://editions.al.uw.edu.pl/exist/apps/dantiscus/letters/{c.doc_id}.xml', 'edition'),
    'dipko_schwarzenhorn_1649': _gams, 'qhod_sema': _gams, 'hearth_tax': _gams, 'urfehdebuecher_basel_x': _gams,
    'dta_1600': _dta, 'sandrart': _dta,
    'disco_sonnets': lambda c: (_gh('pruizf/disco', 'main', 'tei/' + c.after('disco-main/tei/')), 'source-file'),
    'croala': lambda c: (_gh('nevenjovanovic/croatiae-auctores-latini-textus', 'master', f'txts/{c.fname}'),
                         'source-file'),
    'corpus17_french': lambda c: (_gh('e-ditiones/CORPUS17', 'master', f'data/{c.fname}'), 'source-file'),
    'ethica_complementoria': lambda c: (_gh('arockenberger/EthicaComplementoria', 'master',
                                            'transcriptions/' + c.after('transcriptions/')), 'source-file'),
    'lettres_calvin': lambda c: (_gh('pippaJeanne/lettres-calvin', 'main', f'src/lib/xml/{c.fname}'), 'source-file'),
    'poliziano_sappho': lambda c: (_gh('AndreaLaVeglia/Postille_Poliziano_Sappho_EVT', 'main',
                                       f'DigitalEditionEVT/data/text/{c.fname}'), 'source-file'),
    'rup': lambda c: (_gh('jedmondes/Rare-or-Unique-Poems-from-manuscripts-1500-1660-FOLGER', 'main',
                          f'poems/{c.fname}'), 'source-file'),
    'suriano': lambda c: (_gh('HuygensING/suriano', 'main', 'datasource/tei/' + c.after('datasource/tei/')),
                          'source-file'),
    'epistemon_bvh': _bvh, 'freem_bvh': _bvh,
    'ferdinand_i': lambda c: (f'https://ferdinand-korrespondenz.acdh.oeaw.ac.at/{c.stem}.html', 'edition'),
    'kaiserin_eleonora_magdalena': lambda c: (f'https://kaiserin-eleonora.oeaw.ac.at/{c.stem}.html', 'edition'),
    'koenigsfelden': lambda c: (f'https://koenigsfelden.sources-online.org/data/docs/{c.stem}.xml', 'edition'),
    'st_galler_missiven': lambda c: (f'https://missiven.stadtarchiv.ch/missiven/{c.doc_id}', 'edition'),
    'map_of_early_modern_london': lambda c: (f'https://mapoflondon.uvic.ca/{c.stem}.htm', 'edition'),
    'old_bailey': _old_bailey,
    'postscriptum': lambda c: _teitok('http://teitok.clul.ul.pt/postscriptum/', re.sub(r'_TEIP5$', '', c.stem)),
    'h15corpus': lambda c: _teitok('https://historia15.us.es/H15corpus/', c.stem),
    'letradas': lambda c: _teitok('https://corpora.uah.es/letradas/', c.stem),
    'cosuiza': lambda c: _teitok('https://grafila.unil.ch/cosuiza/', c.stem),
    'reed': lambda c: (f'https://ereed.org/records/{c.doc_id}/', 'edition'),
    'religionsfrieden_quellenedition': lambda c: (f'https://tueditions.ulb.tu-darmstadt.de/v/{c.stem}', 'edition'),
    'ssrq_swisslaw': lambda c: (f'https://editio.ssrq-online.ch/api/v1/documents/{c.stem}/xml', 'source-xml'),
    'tagebuecher_christian_ii_anhalt': lambda c: _hab('ed000228', c.fname),
    'fruchtbringende_briefe': lambda c: _hab('ed000213', f'briefe/{c.fname}'),
    'luene_nonnenbriefe': lambda c: _hab('ed000248', f'texts/{c.fname}'),
    'rubens_online': _rubens,
    'karlstadt_kritische_gesamtausgabe': lambda c: _hab('ed000216', f'texts/{c.fname.split("_")[0]}/{c.fname}'),
    'paesi_novamente': lambda c: _hab('ed000145', c.fname),
    'dasypodius_dictionarium_latinogermanicum': lambda c: _hab('ed000008', c.fname),
    'guicciardini_si': lambda c: (c.hdr(r'guicciardini-storia-italia\.huma-num\.fr/storia\.html'), 'edition'),
    'rejestr_poznan': lambda c: ('https://exist.lhdb.kul.pl/exist/apps/1561/index.html', 'edition'),
}


# ======================================================================================================================
# synthetic titles
# ======================================================================================================================
_ALWAYS_SYNTH = {'koenigsfelden', 'germanc', 'hainhofer_reisen', 'dasypodius_dictionarium_latinogermanicum',
                 'leibniz_reihe_viii', 'elan_crenum', 'old_bailey', 'camena_poemata', 'camena_thesaurus',
                 'ssrq_swisslaw', 'casebooks', 'cosuiza', 'ckcc_epistolarium'}


def _parts(*xs) -> str:
    return ', '.join(x for x in xs if x)


def synth_title(slug: str, c: Ctx, info: dict, row: dict) -> Optional[str]:
    """A readable title where the header only has an identifier; None to keep the header title."""
    date = human_date(row.get('date')) if row.get('date') else ''
    snd, adr, plc = row.get('sender'), row.get('addressee'), row.get('place_written')
    if slug == 'casebooks':
        cb = info.get('casebooks') or {}
        t = f'{c.doc_id}: consultation'
        if cb.get('astrologer'):
            t += f' by {cb["astrologer"]}'
        q = cb.get('querent') or cb.get('subject')
        if q:
            t += f' for {q}'
        return _parts(t, date)
    if slug == 'cosuiza':
        s = info.get('summary') if row.get('tier') != 'T3' else None    # editors' abstracts of T3 texts are not copied
        return trim(f'{c.doc_id}: {s}') if s else _parts(c.doc_id, plc, date)
    if slug in ('st_galler_missiven', 'suriano', 'qhod_sema', 'ckcc_epistolarium'):
        if snd or adr:
            n = ''
            t0 = (info.get('titles') or [[0, 0, 0, '']])[0][3] or ''
            m = re.search(r'Nr\. (\d+)', t0) if slug == 'st_galler_missiven' else (
                re.search(r'Letter (\d+\.\d+)', t0) if slug == 'suriano' else None)
            if m:
                n = ('Missive ' if slug == 'st_galler_missiven' else 'Letter ') + m.group(1) + ': '
            return _parts(n + f'{snd or "Unknown sender"} to {adr or "unknown recipient"}', plc, date)
        return None
    if slug == 'koenigsfelden':
        return _parts(row.get('title') or c.doc_id, plc, date)
    if slug == 'old_bailey':
        if c.doc_id.startswith('OA'):
            return _parts("Ordinary's Account of Newgate", date)
        return _parts('Proceedings of the Old Bailey', date)
    if slug == 'germanc':
        m = re.search(r'_(\d{4})(?:-\d{2})?_([a-z]+)\d*$', c.stem)
        place = m.group(2).capitalize() if m else None
        return _parts(row.get('title') or '', place, date or (m.group(1) if m else ''))
    if slug == 'hainhofer_reisen':
        return f'Hainhofer, travel report: {row.get("title")}' if row.get('title') else None
    if slug == 'dasypodius_dictionarium_latinogermanicum':
        return trim(f'Dasypodius, Dictionarium Latinogermanicum: {info.get("head") or c.stem}')
    if slug == 'leibniz_reihe_viii':
        h = info.get('head')
        return trim(f'Leibniz, Reihe VIII: {h}') if h else None
    if slug == 'elan_crenum':
        return f'Cretin, Chronique française: {row.get("title")}' if row.get('title') else None
    if slug in ('camena_poemata', 'camena_thesaurus'):
        h, t = info.get('head'), row.get('title')
        if h and t and len(h) < 60 and h.lower() not in t.lower():
            return f'{t} [{h}]'
        return None
    if slug == 'ssrq_swisslaw':
        sig = info.get('msid') or ''
        sig = sig.split(',', 1)[-1].strip() if sig else ''
        return trim(f'{c.stem}: ' + _parts(sig, date))
    return None


_APPEND_PLACE_DATE = {'lettres_calvin', 'cidtc', 'emmo_folger', 'cavriana', 'dipko_schwarzenhorn_1649',
                      'kaiserin_eleonora_magdalena', 'ferdinand_i', 'postscriptum', 'fruchtbringende_briefe',
                      'bullinger', 'sozinianische_briefwechsel', 'h15corpus', 'luene_nonnenbriefe'}


# ======================================================================================================================
# row assembly
# ======================================================================================================================
_ENT: dict = {}


def set_ckcc_entities(m: dict) -> None:
    _ENT.clear()
    _ENT.update(m)


def _ckcc_person(key: Optional[str]) -> Optional[str]:
    key = (key or '').strip()
    if not key or key == '?':
        return None
    if key.startswith('@'):
        return re.sub(r'\s*\([^)]*\)\s*$', '', key[1:]).strip() or None
    n = _ENT.get('ckcc_epistolarium:' + key)
    if n:
        return n
    parts = key.split('.')
    if len(parts) >= 3 and re.fullmatch(r'[\dfl\-]+', parts[-1]):
        return ' '.join(w.capitalize() for w in (parts[1] + ' ' + parts[0]).replace('-', ' ').split())
    return None


def _ckcc_place(key: Optional[str]) -> Optional[str]:
    key = (key or '').strip()
    return _ENT.get('ckcc_epistolarium:' + key) if key and key != '?' else None


def make_row(slug: str, source_path: str, doc_id: str, info: dict, k, cat: dict) -> dict:
    from .docs_table import doc_key
    c = Ctx(slug, source_path, doc_id, info)
    row: dict = dict(doc_key=doc_key(slug, source_path, doc_id), coll_slug=slug, source_path=source_path, doc_id=doc_id)
    rids = [x for x in (k['renascor_id'] or '').split(';') if x]
    first = next((x for x in rids if x in cat), rids[0] if rids else None)
    row['renascor_id'] = k['renascor_id']
    row['catalogue_url'] = (CATALOGUE_BASE + first) if first else None
    row['edition_url'] = (cat.get(first) or {}).get('url') if first else None
    row['licence'], row['tier'], row['n_place_mentions'] = k['licence'], k['tier'], int(k['n_place_mentions'])
    yf, yt = _int(k['year_from']), _int(k['year_to'])

    # language
    langs = sorted(info.get('langs') or [], key=lambda x: -x[1])
    lang = norm_lang(langs[0][0]) if langs else None
    if (info.get('ckcc') or {}).get('language'):
        lang = norm_lang(info['ckcc']['language'])
    if slug == 'betamasaheft_geez' and any(norm_lang(x[0]) == 'gez' for x in langs):
        lang = 'gez'
    lsrc = 'header' if lang else None
    if not lang:
        lang = norm_lang((info.get('root_attrs') or {}).get('lang'))
        lsrc = 'header' if lang else None
    if not lang and info.get('textlang_words'):
        lang = norm_lang(info['textlang_words'][0])
        lsrc = 'header' if lang else None
    if not lang:
        lang = norm_lang(k.get('lang_m'))
        lsrc = 'mentions' if lang else None
    if not lang:
        cl = (cat.get(first) or {}).get('languages') or []
        if len(cl) == 1 and cl[0] in _LANG_NAMES:
            lang, lsrc = _LANG_NAMES[cl[0]], 'catalogue'
    if slug in _DEFAULT_LANG and lsrc != 'header':
        lang, lsrc = _DEFAULT_LANG[slug], 'collection'
    row['language'], row['language_source'] = lang, lsrc

    # people and places
    snd, adr = list(info.get('senders') or []), list(info.get('addressees') or [])
    pf, pt = list(info.get('place_from') or []), list(info.get('place_to') or [])
    ck = info.get('ckcc')
    if ck:
        s, a = _ckcc_person(ck.get('sender')), _ckcc_person(ck.get('recipient'))
        snd, adr = ([s] if s else []), ([a] if a else [])
        p1, p2 = _ckcc_place(ck.get('senderloc')), _ckcc_place(ck.get('recipientloc'))
        pf, pt = ([p1] if p1 else []), ([p2] if p2 else [])
    title0, tsrc = header_title(info, lang)
    if not (snd or adr) and slug in ('bullinger', 'sozinianische_briefwechsel', 'fruchtbringende_briefe', 'cidtc',
                                     'emmo_folger'):
        tp = split_letter_title(slug, title0)
        if tp:
            s, sp, a, ap = tp
            snd, adr = ([s] if s else []), ([a] if a else [])
            if sp and not pf:
                pf = [sp]
            if ap and not pt:
                pt = [ap]
    row['sender'], row['addressee'] = names(snd), names(adr)
    auth = (names([a for w, a in (info.get('authors') or []) if w == 'titleStmt'])
            or names([a for w, a in (info.get('authors') or []) if w in ('msItem', 'monogr', 'bibl')]))
    row['author'] = row['sender'] or auth
    if slug == 'casebooks':
        row['author'] = (info.get('casebooks') or {}).get('astrologer') or None
    row['place_written'] = names(pf) or names(info.get('origin_place') or [])
    row['place_received'] = names(pt)
    row['repository'] = _dedupe_commas(info.get('msid')) or None

    # date
    date, prec, a, b, dsrc, conflict = choose_date(info, yf, yt)
    row['date_conflict'] = bool(conflict)
    if slug == 'fruchtbringende_briefe' and (not date or prec in ('year', None)):
        m = re.fullmatch(r'(\d{2})(\d{2})(\d{2})', c.stem)
        if m and yf is not None and 1600 + int(m.group(1)) in range(yf, (yt or yf) + 1):
            yy, mm, dd = 1600 + int(m.group(1)), int(m.group(2)), int(m.group(3))
            date, prec = (f'{yy}-{mm:02d}-{dd:02d}', 'day') if mm and dd else ((f'{yy}-{mm:02d}', 'month') if mm else (str(yy), 'year'))
            dsrc = 'filename'
    dsm = k.get('date_source')
    if (not date or prec in ('year', None) or dsrc == 'mentions') and title0 and dsm != 'collection':
        ti, tp = _iso_title(title0)
        if ti and (yf is None or yf - 1 <= int(ti[:4]) <= (yt if yt is not None else yf) + 1):
            date, prec, dsrc = ti, tp, 'title'
            a = b = int(ti[:4])
    if dsm == 'collection' and dsrc == 'mentions' and not (yf is not None and yf == yt):
        date, prec = None, None          # the mention years are only the collection-wide default period
    if yf is None and a is None:        # undated everywhere: a year written in the title, as a last resort
        ys = re.findall(r'(?<!\d)(1[4-7]\d{2})(?:/(\d{2}))?(?!\d)', title0 or '')
        if ys:
            y1 = int(ys[0][0])
            y2 = int(ys[-1][0][:2] + ys[-1][1]) if ys[-1][1] else int(ys[-1][0])
            a, b, dsrc = y1, max(y1, y2), 'title'
            date, prec = (str(y1), 'year') if a == b else (f'{y1}/{b}', 'range')
    if slug == 'germanc' and (not date or prec in ('year', None)):
        for pdte in info.get('pubdates') or []:        # the issue date is in publicationStmt for these newspapers
            y0 = int(pdte[:4])
            if len(pdte) >= 7 and yf is not None and yf <= y0 <= (yt if yt is not None else yf):
                date, prec, dsrc = pdte, ('day' if len(pdte) == 10 else 'month'), 'header'
                break
    row['date'], row['date_precision'] = date, prec
    if yf is None:
        row['year_source'] = dsrc if a is not None else None
    else:
        row['year_source'] = {'collection': 'collection-default', 'filename': 'filename'}.get(dsm, 'header') if dsm else 'mentions'
    row['year_from'] = yf if yf is not None else a
    row['year_to'] = yt if yt is not None else b
    ip = k['in_period']
    if ip is None or ip != ip:
        yfr, yto = row['year_from'], row['year_to']
        ip = None if yfr is None else (yfr <= PERIOD[1] and (yto if yto is not None else yfr) >= PERIOD[0])
    row['in_period'] = None if ip is None else bool(ip)

    # type
    row['doc_type'] = doc_type_for(slug, info, title0 or info.get('summary') or '', doc_id, c.stem)   # type only: nothing is copied

    # title
    title, tsrc2 = title0, tsrc
    if not title and info.get('head'):
        title, tsrc2 = info['head'], 'head'
    if not title and info.get('summary') and row['tier'] != 'T3':
        title, tsrc2 = info['summary'], 'summary'
    if slug == 'emmo_folger':
        title = re.sub(r':\s*(autograph|holograph|scribal|copy|contemporary|manuscript)[^:]*$', '', title or '')
    if slug == 'postscriptum':
        title = re.sub(r'^\[?\d{4}\]?\.\s+', '', title or '')
    full_title = clean_title(title) if title else ''
    row['title'] = trim(full_title) if title else None
    row['title_source'] = tsrc2 or None
    if slug == 'ssrq_swisslaw' and info.get('msheads') and (not title or technical(title, doc_id) or len(title) > 100):
        # the heading of the edited document (msDesc/head), in the language of the document when it has two
        hs = info['msheads']
        pick = [h for h in hs if norm_lang(h[0]) == lang] or [h for h in hs if norm_lang(h[0]) == 'de'] or hs
        if pick[0][1]:
            full_title = clean_title(pick[0][1])
            row['title'], row['title_source'] = trim(full_title), 'head'
    if slug == 'koenigsfelden' and info.get('summary') and row['tier'] != 'T3':
        full_title = clean_title(info['summary'])      # the editors' one-sentence regest is the only descriptive title
        row['title'], row['title_source'] = trim(full_title), 'summary'
    st = synth_title(slug, c, info, row)
    if row['title_source'] in ('head', 'summary') and slug in ('ssrq_swisslaw', 'koenigsfelden'):
        st = None
        d = human_date(row['date']) if row.get('date') else ''
        if d and not re.search(r'\d{4}', row['title']):
            row['title'] = trim(full_title, 160 - len(d) - 2) + ', ' + d
    if st and (not row['title'] or technical(row['title'], doc_id) or slug in _ALWAYS_SYNTH):
        row['title'], row['title_source'] = trim(st), 'synthetic'
    if row['title'] and row['title_source'] != 'synthetic' and technical(row['title'], doc_id) \
            and slug not in _ALWAYS_SYNTH:
        alt = [t for t in (info.get('titles') or []) if t[0] in ('monogr', 'analytic', 'msItem', 'bibl') and t[3]
               and not technical(t[3])]
        alt_t = clean_title(alt[0][3]) if alt else (info.get('head') or '')
        if alt_t:
            row['title'], row['title_source'] = trim(alt_t), 'fallback'
    if not row['title']:
        row['title'], row['title_source'] = trim(_parts(doc_id, row['place_written'], human_date(row['date']))), 'doc_id'
    if row['doc_type'] == 'letter' and slug in _APPEND_PLACE_DATE and row['title_source'] in ('titleStmt', 'head', 'msItem'):
        t = full_title
        if not re.search(r'\d{4}', t):
            extra = []
            if row['place_written'] and fold(row['place_written'].split(';')[0]) not in fold(t):
                extra.append(row['place_written'].split(';')[0])
            if row['date']:
                extra.append(human_date(row['date']))
            if extra:
                sfx = ', ' + ', '.join(extra)
                row['title'] = trim(t, 160 - len(sfx)) + sfx      # shorten the title, never the place and date

    # url
    fn = URLS.get(slug)
    url, kind = fn(c) if fn else (None, None)
    row['url'], row['url_kind'] = (url, kind) if url else (None, None)
    return row
