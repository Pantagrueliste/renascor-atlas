"""NER layer, extra documents: readers for the formats the first extraction (ner2_layer_extract.py) did not read.

Pure functions (text in, text out), no I/O, so that they can be tested. The driver is ``ner2_layer_extra_extract.py``.
Each reader returns the plain text of the primary text of one document, or raises ``Skip(status, why)`` when the document must not go to the model
(translation, site page, place markup of the editors...).

Formats:
  ReF (CorA-XML, diplomatic tokens)      ref_text
  TML (HTML pages of the Thesaurus Musicarum Latinarum)  tml_text
  Pepys diary (Gutenberg, Wheatley)      pepys_text
  Bartoli letters (markdown, one file per letter, grouped by year)  bartoli_md_text
  Wikisource Bible pages (KJV 1769 page, Diodati 1607 page, Luther 1522 page)  bible_page_text
  EMOTHE (UTF-16 TEI, with translations)  emothe_info
  generic TEI (recover mode, NUL bytes, UTF-16)  parse_xml_loose
"""
from __future__ import annotations

import html
import re
import unicodedata


class Skip(Exception):
    """The document is read but must not go to the model. ``status`` is the inventory status."""

    def __init__(self, status: str, why: str = ''):
        super().__init__(f'{status}: {why}')
        self.status, self.why = status, why


# ---------------------------------------------------------------------------------------------------------------------
# bytes -> text, XML
# ---------------------------------------------------------------------------------------------------------------------
def decode_any(b: bytes) -> str:
    """UTF-16 (BOM), UTF-8, then cp1252; NUL bytes dropped (some files carry them in the middle of UTF-8 text)."""
    if b[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return b.decode('utf-16', 'replace')
    if b[:3] == b'\xef\xbb\xbf':
        b = b[3:]
    b = b.replace(b'\x00', b'')
    for enc in ('utf-8', 'cp1252'):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode('latin-1', 'replace')


def parse_xml_loose(b: bytes):
    """lxml root from bytes: decoded by ``decode_any``, XML declaration removed (the encoding is known), recover mode (duplicate xml:id, stray NULs)."""
    from lxml import etree
    s = decode_any(b).lstrip('﻿')
    s = re.sub(r'^\s*<\?xml[^>]*\?>', '', s)
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True, recover=True)
    try:
        root = etree.fromstring(s.encode('utf-8'), parser)
    except etree.XMLSyntaxError as e:
        raise Skip('parse_error', str(e)[:80])
    if root is None:
        raise Skip('parse_error', 'empty document')
    return root


def ltag(el) -> str:
    t = el.tag
    return t.rsplit('}', 1)[-1] if isinstance(t, str) else ''


def top_level_bodies(root) -> list:
    """Every ``body`` that is not inside another body (TEI files with several ``text`` elements, e.g. a collection of letters in one file)."""
    out = []
    for el in root.iter():
        if ltag(el) == 'body':
            p = el.getparent()
            nested = False
            while p is not None:
                if ltag(p) == 'body':
                    nested = True
                    break
                p = p.getparent()
            if not nested:
                out.append(el)
    return out


PLACE_TAGS = frozenset(('placeName', 'settlement', 'geogName', 'country', 'region', 'district', 'bloc'))
RS_PLACE_TYPE = re.compile(r'^(place|location|loc|settlement|region|country|geog\w*|toponym)$', re.I)


def has_place_markup(root) -> bool:
    """Place elements anywhere in the text of the file (front, body, back, groups), not in the TEI header: the atlas reads the whole text, so a place tag in the
    front matter of a play is already an editors' tag of that file."""
    for t in root.iter():
        if ltag(t) != 'text':
            continue
        for el in t.iter():
            n = ltag(el)
            if n in PLACE_TAGS:
                return True
            if n in ('rs', 'name') and RS_PLACE_TYPE.match(el.get('type') or el.get('role') or ''):
                return True
    return False


def itertext_clean(el) -> str:
    return re.sub(r'\s+', ' ', ''.join(el.itertext())).strip()


# ---------------------------------------------------------------------------------------------------------------------
# ReF
# ---------------------------------------------------------------------------------------------------------------------
REF_TOK = re.compile(r'<tok_dipl\b[^>]*\butf="([^"]*)"')


def ref_sigle(head: str, fallback: str) -> str:
    m = re.search(r'sigle="([^"]*)"', head)
    return m.group(1) if m else fallback


def ref_text(xml: str) -> str:
    """Diplomatic tokens (``tok_dipl utf``) that contain a letter, joined by spaces (same rule as the catalogue's harvest)."""
    toks = [html.unescape(t) for t in REF_TOK.findall(xml)]
    return ' '.join(t for t in toks if any(c.isalpha() for c in t))


# ---------------------------------------------------------------------------------------------------------------------
# TML
# ---------------------------------------------------------------------------------------------------------------------
def tml_text(raw: str) -> str:
    raw = re.sub(r'<head\b.*?</head>', ' ', raw, flags=re.S | re.I)
    raw = re.sub(r'<(script|style|nav|header|footer)\b.*?</\1>', ' ', raw, flags=re.S | re.I)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    t = re.sub(r'\s+', ' ', html.unescape(raw)).strip()
    t = re.sub(r'\b(?:Previous|Next) part\b( -->| <--)?', ' ', t)
    t = re.sub(r'(?:-->\s*)?The Center for the History of Music Theory and Literature is a research center.*$', ' ', t, flags=re.S)
    return re.sub(r'\s+', ' ', t).strip()


TML_SITE_PAGES = re.compile(r'^(index|404|search|about|contact|credits|links|help|abbreviations|bibliography)$', re.I)


# ---------------------------------------------------------------------------------------------------------------------
# Pepys
# ---------------------------------------------------------------------------------------------------------------------
def pepys_text(raw: str) -> str:
    """The diary proper of the Gutenberg Wheatley edition: from the first entry (January 1659-60) to 'END OF THE DIARY', without the editor's notes in square brackets
    (they are by Wheatley, 1893-99), without the preface, the appendices and the licence."""
    i = raw.find('JANUARY 1659-1660')
    j = raw.find('END OF THE DIARY')
    if i < 0 or j < 0 or j <= i:
        raise Skip('parse_error', 'Pepys landmarks not found')
    t = raw[i:j]
    t = re.sub(r'\[[^\[\]]{0,6000}\]', ' ', t)         # editor's notes (non-nested brackets)
    t = re.sub(r'\[[^\[\]]{0,6000}\]', ' ', t)         # a second pass for notes nested one level
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t


# ---------------------------------------------------------------------------------------------------------------------
# Bartoli
# ---------------------------------------------------------------------------------------------------------------------
def bartoli_docdate_year(md: str) -> int | None:
    m = re.search(r'docdate:[^\n]*?(1[4-7]\d{2})', md)
    return int(m.group(1)) if m else None


def bartoli_md_text(md: str) -> str:
    md = re.sub(r'^---.*?---', '', md, count=1, flags=re.S)         # front matter (shelfmark, place, sender, recipient, date)
    md = re.sub(r'\[\d+[rv]?\]', ' ', md)                           # folio marks [001r]
    md = re.sub(r'(?m)^\s*---\s*$', ' ', md)                        # separators between the pieces of a letter
    md = re.sub(r'[#*_>`]', ' ', md)
    return re.sub(r'[ \t]+', ' ', md).strip()      # words broken at a line end ("Pa  dova") are left broken: the hyphen is not in the keying, a join would be a guess


# ---------------------------------------------------------------------------------------------------------------------
# Wikisource pages of the plain-text tarballs (it, es)
# ---------------------------------------------------------------------------------------------------------------------
def wikisource_plain(t: str) -> str:
    """Page header of the exports ('Questo testo è riletto e controllato . <dc:title> ... </dc:title> ... //it.wikisource.org/w/index.php?...') and licence tail."""
    t = html.unescape(t)
    t = re.sub(r'<dc:([a-z]+)\b[^>]*>.*?</dc:\1>', ' ', t, flags=re.S)
    t = re.sub(r'<dc:[^>]*/?>', ' ', t)
    t = re.sub(r'^\s*Questo testo è [^.]{0,40}\.', ' ', t)
    t = re.sub(r'//(?:it|es)\.wikisource\.org/\S+', ' ', t)
    t = re.sub(r'\b\d{14}\b', ' ', t)                                   # revision stamp
    t = re.sub(r'Esta obra se encuentra en dominio público.*$', ' ', t, flags=re.S)       # licence tail of the es pages
    t = re.sub(r'(?:Public domain\s*)+(?:false\s*)*$', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


# ---------------------------------------------------------------------------------------------------------------------
# Wikisource Bible pages
# ---------------------------------------------------------------------------------------------------------------------
def bible_page_text(raw: str, kind: str) -> str:
    """kind: 'kjv' (en_v2 pages: chrome up to 'Layout n'), 'diodati' (it_bibles_v2: dc: header and chapter navigation), 'luther' (luther_v4 pages: header, page marks,
    footnotes of the editors after the arrow)."""
    t = raw
    if kind == 'kjv':
        m = re.search(r'Layout \d+ (?:&#8203;|​)', t)
        if m:
            t = t[m.end():]
        t = re.sub(r'&#32;|&#8203;', ' ', t)
    elif kind == 'diodati':
        t = html.unescape(t)
        m = re.search(r'►', t)
        if m:
            t = t[m.end():]
        else:
            m = re.search(r'Informazioni sulla fonte del testo', t)
            if m:
                t = t[m.end():]
    elif kind == 'luther':
        t = html.unescape(t)
        m = re.search(r'\(in eckigen Klammern\)\.', t)
        if m:
            t = t[m.end():]
        k = t.find('↑')
        if k > 0.3 * len(t):                       # footnote list of the transcribers
            t = t[:k]
        t = re.sub(r'\[\s*\d+\s*\]|\[\s*WA\.DB[^\]]*\]', ' ', t)
    else:
        raise ValueError(kind)
    t = html.unescape(t)
    return re.sub(r'\s+', ' ', t).strip()


ARCHAIC_IT = re.compile(r"perciocch|figliuol|\bIddio\b|\bgiuso\b|\bappo\b|\bavea\b|conciossiach|altresì|eziandio|\bOR\b")


def diodati_check(t: str) -> str:
    """The Italian Bible pages of the catalogue are labelled Diodati 1607 but some are another text: an index page (a few lines of chapter links) or a modern translation
    (CEI, 'Sia la luce!'). Diodati's chapters carry archaic forms (perciocchè, figliuoli, Iddio, avea, OR...) at about one per 100 words or more (smallest seen 0.9)."""
    n = len(t.split())
    if n < 150:
        raise Skip('site_page', f'index or empty page ({n} words)')
    per100 = 100 * len(ARCHAIC_IT.findall(t)) / n
    if per100 < 0.5:
        raise Skip('translation', f'modern Italian text, not the 1607 translation (archaic forms {per100:.2f} per 100 words)')
    return t


def fold_name(s: str) -> str:
    """Name key for matching file names of the catalogue (non-ASCII letters already lost there: 'Matth_usevangelium') with real names ('Matthäusevangelium')."""
    s = unicodedata.normalize('NFC', s)
    s = ''.join(c for c in s if c.isascii())
    return re.sub(r'[^a-z0-9]', '', s.lower())


# ---------------------------------------------------------------------------------------------------------------------
# EMOTHE
# ---------------------------------------------------------------------------------------------------------------------
def emothe_info(root) -> dict:
    """Header facts of an EMOTHE TEI file: is it a translation, who translated it, which printed or edited source it is based on (with a year)."""
    info = {'translation': False, 'translators': [], 'source': '', 'base_year': None, 'editor': ''}
    for x in root.iter():
        t = ltag(x)
        if t == 'title' and (x.get('type') or '') == 'traduccion':
            info['translation'] = True
        if t == 'editor':
            role = (x.get('role') or '').lower()
            if role in ('traductor', 'translator'):
                info['translation'] = True
                nm = itertext_clean(x)
                if nm and nm not in info['translators']:
                    info['translators'].append(nm)
        if t == 'sourceDesc' and not info['source']:
            info['source'] = itertext_clean(x)
    years = [int(y) for y in re.findall(r'\b(1[3-9]\d{2}|20[0-2]\d)\b', info['source'])]
    info['base_year'] = years[-1] if years else None
    return info


def emothe_decision(info: dict) -> tuple[bool, str]:
    """(keep, why). A translation is kept only when its base text is a print of 1450-1700 (the translator's own century); all the others are modern translations,
    which are not early modern texts. An original-language play in a modern critical edition (same language, normalised spelling) is kept."""
    if not info['translation']:
        return True, 'original language, modern edition'
    y = info['base_year']
    if y is not None and 1450 <= y <= 1700:
        return True, f'translation printed {y}'
    return False, f'translation, base text {y}' if y else 'translation, base text undated'
