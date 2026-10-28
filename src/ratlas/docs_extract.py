"""Document metadata extraction for the Atlas cards (stage 1 of the document table).

For every document that has at least one place mention (see docs_table.py), read the TEI header (and, where the
header has nothing, the first lines of the text) and collect, without interpretation:

  titles (typed), authors, senders / addressees, places of writing / reception, date candidates (class, ISO
  string, text, years), languages, URL candidates (idno / ptr / ref / xml:base / facsimile with their header path),
  identifiers, repository + shelfmark.

Interpretation (which title, which date, which URL, the document type) is done in docs_table.py, so that the rules
can be changed without re-reading the 100k files. Only well-formed local parsing (common.parse_xml): nothing from
the harvest is executed, no network.

Multi-document files: CKCC (one pseudo-TEI chunk per letter, or one <div type=chapter subtype=artifact> per letter in
the DBNL volumes) and REED (one <text type=record> per record) are split here; every other collection has one
document per file.
"""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional

from lxml import etree

from . import common as C

XML_BASE = '{http://www.w3.org/XML/1998/namespace}base'

# URL hosts that are never a link to the document (licences, authority files, schemas, funders...)
_URL_NOISE = re.compile(
    r'(creativecommons\.org|tei-c\.org|orcid\.org|d-nb\.info|viaf\.org|wikidata\.org|geonames\.org|isni\.org|'
    r'id\.loc\.gov|lobid\.org|w3\.org|schema\.org|purl\.org|doi\.org/10\.(?!5281)|www\.hab\.de/?$|'
    r'gerda-henkel|volkswagenstiftung|klosterkammer|dfg\.de|transkribus|claude\.ai|spacy\.io|rhymetagger|'
    r'fedora/context|diglib\.hab\.de/\?link=|xmlns|isil|/rules/|\.xsd|\.odd|\.rng|\.sch$)', re.I)

_HDR_SKIP = {'editionStmt', 'revisionDesc', 'encodingDesc', 'respStmt', 'change', 'funder', 'sponsor', 'principal',
             'availability', 'licence', 'editor', 'tagsDecl', 'listPrefixDef', 'namespace'}
_TITLE_SUB_TYPES = {'sub', 'subtitle', 'gmd', 'short', 'num', 'volume', 'alt', 'abbr', 'incipit', 'series', 's'}
_ISO = re.compile(r'^-?\d{4}(-\d{2}(-\d{2})?)?$')
_ISO_LOOSE = re.compile(r'^(-?\d{4})(?:-(\d{2}))?(?:-(\d{2}))?')


_TEXTLANG_WORDS = {'frühneuhochdeutsch': 'de', 'fruehneuhochdeutsch': 'de', 'deutsch': 'de', 'latein': 'la',
                   'lateinisch': 'la', 'mittelhochdeutsch': 'gmh', 'mittelniederdeutsch': 'gml', 'französisch': 'fr',
                   'italienisch': 'it', 'niederländisch': 'nl', 'spanisch': 'es', 'englisch': 'en'}


def lname(e) -> str:
    return C.ln(e.tag) or ''


def clean(s: Optional[str], n: int = 0) -> str:
    s = C.norm(s)
    s = re.sub(r'\s+([,.;:])', r'\1', s)
    return s[:n] if n else s


def text_no(el, drop=('note', 'date', 'ref', 'ptr', 'bibl', 'respStmt', 'listBibl'), sep: bool = False,
            pad: tuple = ()) -> str:
    """Text of an element without the given children (names with dates, titles with notes). ``sep``: put a space at
    element boundaries (names written as <surname>X</surname><forename>Y</forename> without whitespace). ``pad``: the
    same, but only around children with these names (titles such as ...<placeName>Torgau</placeName><date>1516</date>)."""
    out = []

    def rec(e):
        if e.text:
            out.append(e.text)
        prev, gap = None, False
        for c in e:
            if isinstance(c.tag, str) and lname(c) not in drop:
                sp = sep or lname(c) in pad
                if lname(c) == 'date' and prev == 'placeName' and not gap:
                    out.append(',')       # ...<placeName>Torgau</placeName><date>1516</date>
                if sp:
                    out.append(' ')
                rec(c)
                if sp:
                    out.append(' ')
            if c.tail:
                out.append(c.tail)
            if isinstance(c.tag, str):
                prev, gap = lname(c), bool((c.tail or '').strip())

    rec(el)
    return clean(''.join(out))



_MONTHS_RAW = {
    1: 'january jan januar janvier janv enero ene gennaio gen genn ianuarii ianuarius januarii januarius ianuario janeiro januari',
    2: 'february feb februar fevrier fev febrero febrer febbraio febbr februarii februarius ianuarii-no fevereiro februari',
    3: 'march mar marz maerz mars marzo marc martii martius martio marco maart',
    4: 'april apr avril abril aprile aprilis aprilii aprili',
    5: 'may mai mayo maggio maii maius maio mei',
    6: 'june jun juni juin junio giugno iunii iunius junii junius iunio junho',
    7: 'july jul juli juillet julio luglio iulii iulius julii julius iulio julho',
    8: 'august aug august aout agosto ago augusti augustus agosto',
    9: 'september sep sept septembre septiembre setiembre settembre septembris septembri setembro',
    10: 'october oct okt oktober octobre octubre ottobre octobris octobri outubro',
    11: 'november nov novembre noviembre novembris novembri novembro',
    12: 'december dec dez dezember decembre diciembre dic dicembre decembris decembri dezembro',
}
_MONTH = {}
for _m, _names in _MONTHS_RAW.items():
    for _n in _names.split():
        _MONTH[_n] = _m


def _fold(s: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')


_MONTH_RX = '|'.join(sorted(_MONTH, key=len, reverse=True))
_MR = r'(?<![a-z])(' + _MONTH_RX + r')(?![a-z])'
_P_DMY = re.compile(r'(?<!\d)(\d{1,2})(?:st|nd|rd|th|er|\.|o|º)?\s*(?:de |of |di |d\')?\s*' + _MR + r'\.?\s*(?:de |of |anno |a\. ?)?\[?(\d{4})')
_P_MDY = re.compile(_MR + r'\.?\s*(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})')
_P_YMD = re.compile(r'(\d{4})\s*,?\s*' + _MR + r'\.?\s*(\d{1,2})(?!\d)')
_P_MY = re.compile(_MR + r'\.?\s*,?\s*(?:de |of |anno )?(\d{4})')


def iso_from_text(text: str):
    """(ISO date, precision) read from a written date ('12 novembre 1688', '1688 noviembre 12', 'March 3, 1590',
    'Oktober 1623'); (None, '') if the text has no day or month with a year."""
    t = _fold(text or '')
    m = re.fullmatch(r'\s*(\d{4})-(\d{2})(?:-(\d{2}))?\s*', t)
    if m and 1 <= int(m.group(2)) <= 12:
        return (t.strip(), 'day' if m.group(3) else 'month')
    m = _P_DMY.search(t)
    if m and 1 <= int(m.group(1)) <= 31:
        return f'{int(m.group(3)):04d}-{_MONTH[m.group(2)]:02d}-{int(m.group(1)):02d}', 'day'
    m = _P_YMD.search(t)
    if m and 1 <= int(m.group(3)) <= 31:
        return f'{int(m.group(1)):04d}-{_MONTH[m.group(2)]:02d}-{int(m.group(3)):02d}', 'day'
    m = _P_MDY.search(t)
    if m and 1 <= int(m.group(2)) <= 31:
        return f'{int(m.group(3)):04d}-{_MONTH[m.group(1)]:02d}-{int(m.group(2)):02d}', 'day'
    m = _P_MY.search(t)
    if m:
        return f'{int(m.group(2)):04d}-{_MONTH[m.group(1)]:02d}', 'month'
    return None, ''


_STRUCT = {'name', 'forename', 'surname', 'gender', 'event', 'socecStatus', 'occupation', 'birth', 'death', 'note',
           'state', 'trait', 'residence', 'nationality', 'age', 'sex', 'floruit', 'affiliation', 'location', 'geo',
           'settlement', 'country', 'region', 'addName', 'roleName', 'nameLink', 'genName', 'idno', 'bibl'}


def own_name(el, maxlen: int = 140) -> str:
    """Name text of a persName / placeName / name element; structured children (name, forename...) are read, the
    biographical ones (event, gender, note...) are never part of the name."""
    kids = [c for c in el if isinstance(c.tag, str)]
    if any(lname(c) in _STRUCT for c in kids):
        parts = [c for c in kids if lname(c) in ('name', 'forename', 'surname', 'addName', 'roleName', 'nameLink', 'genName')]
        if parts:
            if any(lname(c) == 'name' for c in parts):
                parts = [c for c in parts if lname(c) == 'name'][:1]
            return clean(' '.join(text_no(c, drop=('note', 'date', 'ref', 'ptr')) for c in parts))[:maxlen]
        loc = [c for c in kids if lname(c) in ('settlement', 'location', 'country', 'region')]
        if loc:
            return clean(', '.join(text_no(c, drop=('geo', 'note')) for c in loc))[:maxlen]
        return clean((el.text or '') + ' '.join((c.tail or '') for c in kids))[:maxlen]
    drop = ('note', 'date', 'ref', 'ptr', 'geo', 'event')
    if lname(el) in ('persName', 'name', 'orgName'):       # <persName><persName>X</persName> <placeName>Y</placeName></persName>
        drop += ('placeName', 'settlement')
    return text_no(el, drop=drop, sep=True)[:maxlen]


def iso_of(el) -> tuple[Optional[str], str]:
    """(ISO string, precision) of a date-like element from its attributes; (None, '') if no attribute helps."""
    g = lambda *k: next((el.get(x) for x in k if el.get(x)), None)   # noqa: E731
    when = g('when', 'when-iso', 'when-custom')
    if when:
        w = when.strip()
        m = _ISO_LOOSE.match(w)
        if m:
            iso = m.group(1) + (f'-{m.group(2)}' if m.group(2) else '') + (f'-{m.group(3)}' if m.group(3) else '')
            return iso, ('day' if m.group(3) else 'month' if m.group(2) else 'year')
    a = g('notBefore', 'notBefore-iso', 'notBefore-custom', 'from', 'from-iso')
    b = g('notAfter', 'notAfter-iso', 'notAfter-custom', 'to', 'to-iso')
    ma = _ISO_LOOSE.match(a.strip()) if a else None
    mb = _ISO_LOOSE.match(b.strip()) if b else None
    fa = ma.group(0) if ma else None
    fb = mb.group(0) if mb else None
    if fa and fb:
        return (fa if fa == fb else f'{fa}/{fb}'), ('day' if fa == fb and len(fa) == 10 else 'range')
    if fa:
        return f'{fa}/..', 'range'
    if fb:
        return f'../{fb}', 'range'
    return None, ''


def _walk(h):
    """Yield (el, local name, stack of ancestor names including el) for a header, skipping non-element nodes."""
    stack: list[str] = []
    for ev, el in etree.iterwalk(h, events=('start', 'end')):
        n = C.ln(el.tag) if isinstance(el.tag, str) else None
        if n is None:
            continue
        if ev == 'end':
            stack.pop()
            continue
        stack.append(n)
        yield el, n, stack


def _date_class(stack: list[str], el) -> Optional[str]:
    s = set(stack)
    if s & _HDR_SKIP and 'correspAction' not in s:
        return None
    if s & {'author', 'persName', 'birth', 'death', 'floruit', 'person', 'respStmt', 'bibl_edition'} and \
            not (s & {'correspAction'}):
        return None
    if 'correspAction' in s:
        return 'sent' if C._corresp_type(el) == 'sent' else 'received'
    if 'publicationStmt' in s and 'sourceDesc' not in s:
        return None
    if s & {'creation', 'origDate', 'docDate', 'settingDesc', 'origin'}:
        return 'creation'
    if 'titleStmt' in s:
        return 'title' if 'title' in s else None
    if 'sourceDesc' in s:
        return 'source'
    if 'seriesStmt' in s or 'notesStmt' in s:
        return None
    return 'other'


def header_of(root):
    for el in root.iter():
        if lname(el) == 'teiHeader':
            return el
    return None


def parse_header(root, hdr=None) -> dict:
    """Raw metadata of one TEI document (root = the document element or the whole tree)."""
    h = hdr if hdr is not None else header_of(root)
    info: dict = dict(titles=[], authors=[], senders=[], addressees=[], place_from=[], place_to=[], dates=[],
                      langs=[], urls=[], idnos=[], msid='', origin_place=[], keywords=[])
    if h is None:
        return info
    for el, n, stack in _walk(h):
        st = set(stack)
        par = stack[-2] if len(stack) > 1 else ''
        if n == 'title':
            if st & {'respStmt', 'funder', 'sponsor', 'principal', 'listBibl', 'taxonomy', 'tagsDecl', 'encodingDesc',
                     'editionStmt', 'revisionDesc'}:
                continue
            where = ('titleStmt' if 'titleStmt' in st else 'msItem' if 'msItem' in st else 'analytic' if 'analytic' in st
                     else 'monogr' if 'monogr' in st else 'bibl' if 'bibl' in st else 'head' if 'head' in st
                     else 'series' if 'seriesStmt' in st else 'other')
            t = text_no(el, drop=('note', 'ref', 'ptr'), pad=('placeName', 'date'))
            if t:
                info['titles'].append((where, el.get('type') or el.get('level') or '', el.get(C.XML_LANG) or el.get('lang') or '', t))
        elif n == 'head' and 'msDesc' in st and 'sourceDesc' in st:
            t = text_no(el, drop=('note', 'ref', 'ptr'), pad=('placeName', 'date'))
            if t:
                info.setdefault('msheads', []).append((el.get(C.XML_LANG) or el.get('lang') or '', t[:300]))
        elif n in ('author',) and not (st & {'respStmt', 'editionStmt', 'revisionDesc', 'encodingDesc'}):
            t = text_no(el, drop=('note', 'date', 'ref', 'ptr'), sep=True)
            where = 'titleStmt' if 'titleStmt' in st else 'msItem' if 'msItem' in st else 'monogr' if 'monogr' in st else 'bibl'
            if t and 'listBibl' not in st:
                info['authors'].append((where, t))
        elif n in ('persName', 'name', 'orgName', 'rs') and 'correspAction' in st \
                and not ({'persName', 'name', 'orgName'} & set(stack[:-1])):
            t = own_name(el)
            ct = C._corresp_type(el)
            tgt = 'senders' if ct == 'sent' else 'addressees'
            if t and n != 'rs':
                info[tgt].append(t)
        elif n in ('persName',) and (el.get('role') in ('sender', 'addressee')) and 'correspAction' not in st:
            t = text_no(el, drop=('note', 'date'))
            if t:
                info['senders' if el.get('role') == 'sender' else 'addressees'].append(t)
        elif n in ('placeName', 'settlement', 'geogName', 'region') and 'correspAction' in st \
                and not ({'placeName', 'settlement', 'geogName'} & set(stack[:-1])):
            t = own_name(el)
            if t:
                info['place_from' if C._corresp_type(el) == 'sent' else 'place_to'].append(t)
        elif n == 'placeName' and ('origPlace' in st or 'creation' in st or 'origin' in st):
            t = own_name(el)
            if t:
                info['origin_place'].append(t)
        elif n == 'origPlace' and not el.findall('.//{*}placeName'):
            t = own_name(el)
            if t:
                info['origin_place'].append(t)
        elif n in ('date', 'origDate', 'docDate', 'creation'):
            if n == 'date' and 'publicationStmt' in st and 'sourceDesc' not in st \
                    and re.fullmatch(r'\d{4}-\d{2}(-\d{2})?', el.get('when') or ''):
                info.setdefault('pubdates', []).append(el.get('when'))
            cls = _date_class(stack, el)
            if cls is None:
                continue
            if n == 'creation' and el.findall('.//{*}date'):
                continue
            iso, prec = iso_of(el)
            txt = clean(''.join(el.itertext()))[:80]
            rg = C.date_range(el)
            if iso is None and rg is None:
                continue
            if prec in ('', 'year', 'month') and txt:
                ti, tp = iso_from_text(txt)
                if ti and (iso is None or (ti.startswith(iso[:4]) and len(ti) > len(iso))):
                    iso, prec = ti, tp
            info['dates'].append((cls, iso, prec, txt, rg[0] if rg else None, rg[1] if rg else None))
        elif n == 'language':
            ident = el.get('ident') or el.get('id') or ''
            if ident and 'langUsage' in st:
                info['langs'].append((ident, int(el.get('usage')) if (el.get('usage') or '').isdigit() else 0))
        elif n == 'textLang':
            if el.get('mainLang'):
                info['langs'].append((el.get('mainLang'), 1000))
            else:                      # language written in words: 'Frühneuhochdeutsch', 'Latein'
                code = _TEXTLANG_WORDS.get(clean(''.join(el.itertext())).lower())
                if code:                # only a fallback (rules): a root xml:lang is better than textLang in Bullinger
                    info.setdefault('textlang_words', []).append(code)
        elif n == 'summary' and not info.get('summary'):
            t = text_no(el, drop=('note', 'ref', 'ptr'))
            if t:
                info['summary'] = t[:300]
        elif n == 'term' and 'keywords' in st and len(info['keywords']) < 6:
            t = clean(''.join(el.itertext()))
            if t:
                info['keywords'].append(t[:40])
        elif n == 'idno':
            t = clean(''.join(el.itertext()))
            if t:
                info['idnos'].append((par, el.get('type') or el.get('subtype') or '', t[:200]))
                if re.match(r'https?://', t) and not _URL_NOISE.search(t):
                    for u in re.findall(r'https?://\S+', t):
                        info['urls'].append((f'{par}/idno[{el.get("type") or el.get("subtype") or ""}]', u.rstrip('.,;)')))
        # URL-bearing attributes (ptr, ref, graphic, edition@base, include...)
        if n in ('ptr', 'ref', 'graphic', 'edition', 'authority', 'link', 'bibl', 'idno', 'include'):
            for a in ('target', 'url', 'base', 'href', 'cRef', 'source', XML_BASE):
                v = el.get(a)
                if v and re.match(r'https?://', v) and not _URL_NOISE.search(v):
                    info['urls'].append((f'{par}/{n}@{C.ln(a)}[{el.get("type") or ""}]', v.strip()))
        # msIdentifier: repository + shelfmark
        if n == 'msIdentifier' and not info['msid']:
            rep = el.find('.//{*}repository')
            idns = [clean(''.join(x.itertext())) for x in el.iterfind('.//{*}idno')]
            r = clean(''.join(rep.itertext())) if rep is not None else ''
            i = max(idns, key=len) if idns else ''
            info['msid'] = ', '.join(x for x in (r, i) if x)[:160]
    # root attributes
    b = root.get(XML_BASE)
    if b and b.startswith('http'):
        info['urls'].append(('TEI@xml:base', b))
    info['root_attrs'] = {C.ln(k): v for k, v in root.attrib.items() if C.ln(k) in ('id', 'type', 'lang', 'n', 'subtype')}
    # de-duplicate lists
    for k in ('senders', 'addressees', 'place_from', 'place_to', 'origin_place', 'keywords'):
        seen, out = set(), []
        for x in info[k]:
            if x not in seen:
                seen.add(x)
                out.append(x)
        info[k] = out
    info['urls'] = list(dict.fromkeys(info['urls']))
    return info


def text_dates(root, limit: int = 60000) -> list:
    """First date of an opener/dateline/docDate in the text (the generic last resort)."""
    out = []
    stack: list[str] = []
    seen = 0
    for ev, el in etree.iterwalk(root, events=('start', 'end')):
        n = C.ln(el.tag) if isinstance(el.tag, str) else None
        if n is None:
            continue
        if ev == 'end':
            stack.pop()
            continue
        stack.append(n)
        seen += 1
        if seen > limit:
            break
        if n == 'teiHeader':
            continue
        if n in ('date', 'docDate') and ('dateline' in stack or 'opener' in stack or 'closer' in stack
                                         or 'docImprint' in stack or n == 'docDate') and 'teiHeader' not in stack:
            iso, prec = iso_of(el)
            rg = C.date_range(el)
            txt = clean(''.join(el.itertext()))[:80]
            if prec in ('', 'year', 'month') and txt:
                ti, tp = iso_from_text(txt)
                if ti and (iso is None or (ti.startswith(iso[:4]) and len(ti) > len(iso))):
                    iso, prec = ti, tp
            if iso or rg:
                out.append(('text', iso, prec, txt, rg[0] if rg else None, rg[1] if rg else None))
                break
    return out


def first_head(root) -> str:
    """First <head> / <docTitle> of the text, for documents whose header has no usable title."""
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if lname(el) in ('head', 'titlePart') and 'teiHeader' not in [lname(a) for a in el.iterancestors()]:
            t = text_no(el, drop=('note', 'ref', 'ptr', 'pb', 'lb'))
            if len(t) > 2:
                return t[:200]
    return ''


# ----------------------------------------------------------------------------------------------------------------------
# multi-document files
# ----------------------------------------------------------------------------------------------------------------------
def _years_of(v: str):
    ys = [int(y) for y in re.findall(r'(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)', v or '')]
    return (min(ys), max(ys)) if ys else None


def _ckcc_docs(rec_path: str, data: bytes, want: set) -> Iterator[tuple[str, dict]]:
    """Yield (doc_id, info) for the letters of a CKCC file (same doc_id rule as collections/ckcc_epistolarium.py)."""
    stem = rec_path.rsplit('/', 1)[-1]
    for ext in ('.xml', '.txt'):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    if rec_path.endswith('.txt'):
        raw = data.decode('utf-8-sig', errors='replace').replace('\r\n', '\n')
        parts = re.split(r'(?m)^### (\S+)\s*$', raw)
        corpus = rec_path.rsplit('/', 1)[-1].split('-')[0][:7] if rec_path.endswith('-corpus.txt') else (
            rec_path.split('/data/corpus/data/')[1].split('/')[0])
        for i in range(1, len(parts) - 1, 2):
            name, chunk = parts[i], parts[i + 1]
            lid_guess = name.rsplit('.', 1)[0]
            if f'{corpus}:{lid_guess}' not in want and not any(w.startswith(f'{corpus}:') for w in want):
                continue
            root = C.parse_xml(chunk.encode('utf-8'))
            if root is None:
                continue
            meta = {}
            for e in root.iter('meta'):
                if e.get('type') and e.get('type') not in meta:
                    meta[e.get('type')] = e.get('value') or ''
            lid = meta.get('id') or lid_guess
            did = f'{corpus}:{lid}'
            if did in want:
                yield did, _ckcc_info(meta, root)
    else:
        root = C.parse_xml(data)
        if root is None:
            return
        code = stem.split('_')[0]
        corpus = stem[:7]
        found: set = set()
        for ch in root.iter('div'):
            if ch.get('type') != 'chapter' or ch.get('subtype') != 'artifact':
                continue
            meta = {}
            ig = C.child(ch, 'interpGrp')
            if ig is not None:
                for i in ig:
                    if C.ln(i.tag) == 'interp' and i.get('type') and i.get('type') not in meta:
                        meta[i.get('type')] = i.get('value') or ''
            lid = (meta.get('id') or '').strip()
            did = f'{corpus}:{lid}' if lid else stem
            if did in want:
                found.add(did)
                yield did, _ckcc_info(meta, ch)
        if stem in want and stem not in found:   # front matter / indexes of the volume (doc_id = file stem)
            info = parse_header(root)
            info['ckcc_volume'] = True
            yield stem, info


def _ckcc_info(meta: dict, el) -> dict:
    info = parse_header(el, hdr=etree.Element('x'))
    info['ckcc'] = {k: v for k, v in meta.items() if v}
    d = meta.get('date') or ''
    rg = _years_of(d)
    if d:
        info['dates'].append(('sent', d if _ISO.match(d) else None, 'day' if len(d) == 10 else 'month' if len(d) == 7 else 'year',
                              d, rg[0] if rg else None, rg[1] if rg else None))
    if meta.get('language'):
        info['langs'].append((meta['language'], 1000))
    # first lines of the letter, for the title fallback
    return info


def _reed_docs(data: bytes, want: set) -> Iterator[tuple[str, dict]]:
    root = C.parse_xml(data)
    if root is None:
        return
    base = parse_header(root)
    for text in root.iter('{*}text'):
        if text.get('type') != 'record':
            continue
        did = text.get(C.XML_ID)
        if did not in want:
            continue
        info = dict(base)
        info = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
        info['dates'] = []
        info['titles'] = []
        head = None
        for el in text.iter('{*}head'):
            head = el
            break
        if head is not None:
            info['titles'].append(('head', '', '', text_no(head, drop=('note', 'ref', 'ptr', 'pb'))[:400]))
            for d in head.iter('{*}date'):
                iso, prec = iso_of(d)
                rg = C.date_range(d)
                if iso or rg:
                    info['dates'].append(('creation', iso, prec, clean(''.join(d.itertext()))[:80],
                                          rg[0] if rg else None, rg[1] if rg else None))
        info['reed'] = {'type': text.get('type'), 'n': text.get('n') or ''}
        yield did, info



_STRUCT_TAGS = {'sp': 'sp', 'castList': 'cast', 'l': 'l', 'lg': 'lg', 'opener': 'opener', 'closer': 'closer',
                'salute': 'salute', 'signed': 'signed', 'dateline': 'dateline', 'p': 'p', 'head': 'head', 'entry': 'entry',
                'item': 'item', 'row': 'row', 'cell': 'cell', 'div': 'div', 'text': 'text', 'trial': 'trial'}


def struct_counts(root, cap: int = 400000) -> dict:
    """Counts of a few structural tags in the text (play: sp/castList; verse: l/lg; letter: opener/closer...)."""
    out: dict = {}
    n = 0
    for el in root.iter():
        t = el.tag
        if not isinstance(t, str):
            continue
        k = _STRUCT_TAGS.get(t.rsplit('}', 1)[-1])
        if k:
            out[k] = out.get(k, 0) + 1
        n += 1
        if n > cap:
            out['capped'] = 1
            break
    return out


def _casebooks_extras(root) -> dict:
    """Astrologer and querent of a Casebooks consultation (settingDesc/cb:consultation + listPerson)."""
    people = {}
    for p in root.iter('{*}person'):
        pid = p.get(C.XML_ID)
        pn = p.find('{*}persName')
        if pid and pn is not None:
            parts = [clean(''.join(c.itertext())) for c in pn if lname(c) in ('forename', 'surname', 'addName')]
            people[pid] = clean(' '.join(x for x in parts if x)) or clean(''.join(pn.itertext()))
    out: dict = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        n = lname(el)
        ref = (el.get('ref') or '').lstrip('#')
        if n == 'astrologer' and 'astrologer' not in out:
            out['astrologer'] = people.get(ref) or ref
        elif n == 'querent' and 'querent' not in out:
            out['querent'] = people.get(ref) or ref
        elif n == 'subject' and 'subject' not in out:
            out['subject'] = people.get(ref) or ref
    return out


def _old_bailey(root, info: dict) -> None:
    """Old Bailey files have no teiHeader: title / date from the first division and its <interp> elements."""
    for d in root.iter():
        if isinstance(d.tag, str) and lname(d) in ('div0', 'div1') and d.get('type') in ('ordinarysAccount', 'sessionsPaper'):
            info['ob'] = {'type': d.get('type'), 'id': d.get('id')}
            for i in d.iter('interp'):
                if i.get('type') == 'date' and i.get('value'):
                    v = i.get('value')
                    if re.fullmatch(r'\d{8}', v):
                        info['dates'].append(('creation', f'{v[:4]}-{v[4:6]}-{v[6:]}', 'day', v, int(v[:4]), int(v[:4])))
                    break
            p = d.find('p')
            if p is not None:
                info['head'] = text_no(p, drop=('note',))[:200]
            break


def extract_file(slug: str, path: str, data: bytes, want_doc_ids: set) -> Iterator[tuple[str, dict]]:
    """Yield (doc_id, info) for the documents of one file that appear in ``want_doc_ids``."""
    if slug == 'ckcc_epistolarium':
        yield from _ckcc_docs(path, data, want_doc_ids)
        return
    if slug == 'reed':
        yield from _reed_docs(data, want_doc_ids)
        return
    root = C.parse_xml(data)
    if root is None:
        raise ValueError('unparseable XML')
    info = parse_header(root)
    if not info['dates']:
        info['text_dates'] = text_dates(root)
    info['head'] = first_head(root)
    info['struct'] = struct_counts(root)
    if slug == 'casebooks':
        info['casebooks'] = _casebooks_extras(root)
    elif slug == 'old_bailey':
        _old_bailey(root, info)
    for did in want_doc_ids:
        yield did, info


def jsonable(info: dict) -> str:
    return json.dumps(info, ensure_ascii=False, default=list)
