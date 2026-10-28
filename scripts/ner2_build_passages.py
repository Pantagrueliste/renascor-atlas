"""Build the candidate passages of reference set v2 (gold2): 30 seed passages (unchanged) + ~220 new ones, stratified by language.

Sources (all read from copies extracted by the caller into ``--root``, from RQ/release/text_archive_v1.0, which is never modified):
  en  EEBO-TCP (eebotcp_fill, eebo_phase2: TEI, no place markup, checked per document), Lampeter tracts, Hakluyt (Principal Navigations)
  la  EEBO-TCP Latin (latineebo), Latin Wikisource
  it  Italian Wikisource (Renaissance), Galileo (Favaro edition on Wikisource), LiberLiber (curated early modern prose)
  es  Estilometria / Siglo de Oro (novels, entremeses), Spanish Wikisource
  fr  ABU, Gallicorpora/CREMMA HTR (diplomatic), French Wikisource (Renaissance), Mazarinades (Antonomaz)
  de  German Wikisource (Renaissance), TextGrid (Digitale Bibliothek)

Rules: dated 1450-1700; not in the seed; not in a v1 train/dev/test edition (document paths of data_v1 are compared with the archive file names, and
``scripts/ner2_leak_check.py`` compares 8-grams on Hypatia); at most 2 passages per document (1 where possible); at most 15 % of a language from one
author; 120-200 words, >= 8 capitalised words, <= 10 % digits; diplomatic reading kept (long s ...); end-of-line hyphens rejoined; detached drop capitals
repaired. No model output is used anywhere.

    python scripts/ner2_build_passages.py --root SCRATCH/ta --seed-file gold_seed_v1.jsonl --out passages_candidates.jsonl
"""
from __future__ import annotations

import argparse
import glob
import gzip
import html
import json
import math
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.ner2_passages import doc_vocab, draw_passage, fingerprints, rejoin_eol_hyphens, repair_dropcaps, tokenize  # noqa: E402

TEI = '{http://www.tei-c.org/ns/1.0}'
YEAR_LO, YEAR_HI = 1450, 1700
PLACE_TAGS = ('placeName', 'geogName', 'settlement', 'persName', 'region', 'country', 'district', 'rs', 'name')


@dataclass
class Doc:
    lang: str
    collection: str
    doc: str
    title: str
    author: str
    year: int | None
    date: str
    source_url: str
    licence_note: str
    text: str
    pool: str
    date_precision: str = 'work'
    min_stop: float = 0.12
    extra: dict = field(default_factory=dict)
    vocab: object = None
    max_oov: float | None = None


def lic(lic_name: str, coll: str, tier: str = 'T1') -> str:
    return f'{lic_name} ({coll}, text archive {tier})'


def year_of(s) -> int | None:
    m = re.search(r'\b(1[4-7]\d\d)\b', str(s or ''))
    return int(m.group(1)) if m else None


def clean_generic(t: str) -> str:
    t = html.unescape(t)
    t = re.sub(r'\[§\]', ' ', t)
    t = re.sub(r'\(M\d+\)', ' ', t)                      # Hakluyt marginal markers
    t = re.sub(r'(?m)^\s*\[?\s*\d{1,4}\s*\]?\s*$', '', t)  # lone page / line numbers
    t = rejoin_eol_hyphens(t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


# ---------------------------------------------------------------- TEI readers (EEBO, Mazarinades)

def tei_body_text(root):
    body = root.find(f'.//{TEI}body')
    if body is None:
        body = root.find('.//body')
    if body is None:
        return ''
    buf: list[str] = []

    def walk(el):
        tag = etree.QName(el).localname if isinstance(el.tag, str) else None
        if tag in ('note', 'figure', 'pb', 'fw', 'gap', 'teiHeader', 'del', 'corr', 'reg', 'expan', 'supplied', 'rdg', 'imprimatur'):
            return
        if tag == 'lb':
            if (el.get('rend') == '¬' or el.get('break') == 'no'):
                buf.append('\u0001')                       # joined with the next piece
            else:
                buf.append('\n')
            return
        if tag == 'g' and (el.get('ref') or '').endswith('EOLhyphen'):
            return
        if tag in ('p', 'l', 'lg', 'head', 'item', 'salute', 'dateline', 'closer', 'signed', 'trailer', 'label', 'speaker', 'stage', 'div') and buf and not buf[-1].endswith('\n'):
            buf.append('\n')
        if el.text:
            buf.append(el.text)
        for c in el:
            walk(c)
            if c.tail:
                buf.append(c.tail)
        if tag in ('p', 'lg', 'head', 'item'):
            buf.append('\n')

    walk(body)
    txt = ''.join(buf).replace('∣', '')
    txt = re.sub(r'\u0001\s*', '', txt)
    txt = re.sub(r'[ \t\r\f\v]+', ' ', txt)
    txt = re.sub(r'\n\s*', '\n', txt)
    return txt.strip()


def has_place_markup(root) -> bool:
    body = root.find(f'.//{TEI}body')
    if body is None:
        body = root.find('.//body')
    if body is None:
        return False
    for el in body.iter():
        if isinstance(el.tag, str) and etree.QName(el).localname in PLACE_TAGS:
            return True
    return False


def load_eebo(root: Path, sub: str, lang: str, pool: str, coll: str, base_url: str, licnote: str, exclude: set[str]):
    out = []
    for f in sorted(glob.glob(str(root / sub / '*.xml'))):
        doc = Path(f).stem
        if doc in exclude:
            continue
        try:
            tree = etree.parse(f)
        except Exception:
            continue
        r = tree.getroot()
        xp = lambda q: ''.join(r.xpath(q)).strip()
        ident = xp('string(//*[local-name()="langUsage"]/*[local-name()="language"]/@ident)')
        if ident != {'en': 'eng', 'la': 'lat'}[lang]:
            continue
        if has_place_markup(r):
            continue
        title = re.sub(r'\s+', ' ', xp('string(//*[local-name()="titleStmt"]/*[local-name()="title"])'))
        author = re.sub(r'\s+', ' ', xp('string(//*[local-name()="titleStmt"]/*[local-name()="author"])')).strip(' .,') or f'anon:{doc}'
        yr = year_of(xp('string(//*[local-name()="editionStmt"]//*[local-name()="date"])')) or year_of(
            xp('string(//*[local-name()="sourceDesc"]//*[local-name()="publicationStmt"]/*[local-name()="date"])'))
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        txt = clean_generic(tei_body_text(r))
        if len(txt.split()) < 600:
            continue
        out.append(Doc(lang, coll, doc, title[:160], author, yr, str(yr), f'{base_url}/{doc}.0001.001', licnote, txt, pool, 'edition'))
    return out


def load_mazarinades(root: Path):
    out = []
    for f in sorted(glob.glob(str(root / 'mazarinades' / '*.xml'))):
        doc = Path(f).stem
        r = etree.parse(f).getroot()
        if has_place_markup(r):
            continue
        xp = lambda q: re.sub(r'\s+', ' ', ''.join(r.xpath(q)).strip())
        title = xp('string(//*[local-name()="titleStmt"]/*[local-name()="title"][@type="main"])')
        author = xp('string(//*[local-name()="sourceDesc"]//*[local-name()="author"])') or f'anon:{doc}'
        yr = year_of(xp('string(//*[local-name()="sourceDesc"]//*[local-name()="bibl"]/*[local-name()="date"]/@when)'))
        url = xp('string(//*[local-name()="sourceDesc"]//*[local-name()="bibl"]/*[local-name()="ref"]/@target)')
        if not yr or not (YEAR_LO <= yr <= YEAR_HI) or not url:
            continue
        txt = clean_generic(tei_body_text(r))
        if len(txt.split()) < 600:
            continue
        out.append(Doc('fr', 'Mazarinades', doc, title[:160], author, yr, str(yr), url, lic('CC BY 4.0', 'Mazarinades/Antonomaz'), txt, 'fr_maz', 'edition',
                       min_stop=0.2, max_oov=0.11))
    return out


# ---------------------------------------------------------------- jsonl readers

def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_one(root: Path, pattern: str) -> str:
    g = glob.glob(str(root / pattern))
    return g[0] if g else ''


def load_abu(root: Path):
    out = []
    for d in read_jsonl(find_one(root, 'raw__ABU_French/*/*.jsonl')):
        yr = year_of(d['year'])
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        m = re.search(r'abu\.cnam\.fr\),\s*([^:]+):\s*(.*)$', d['source'])
        author = m.group(1).strip() if m else f'anon:{d["filename"]}'
        title = m.group(2).split('—')[-1].strip() if m else d['filename']
        out.append(Doc('fr', 'ABU_French', d['filename'], title[:160], author, yr, str(yr), 'https://abu.cnam.fr/', lic('public domain', 'ABU_French'),
                       clean_generic(d['raw']), 'fr_abu'))
    return out


def load_gallicorpora(root: Path):
    out = []
    for d in read_jsonl(find_one(root, 'raw__Gallicorpora_French_HTR/*/*.jsonl')):
        yr = year_of(d['year'])
        if d['language'] != 'fra' or not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        fn = d['filename']
        m = re.search(r'(bpt6k\w+|btv1b\w+)', fn)
        url = f'https://gallica.bnf.fr/ark:/12148/{m.group(1)}' if m else 'https://github.com/HTR-United'
        title = re.sub(r'_btv1b\w+$', '', re.sub(r'^(ocr17plus|gallicorpora\w*|cremma\w*)_', '', fn)).replace('_', ' ')
        # lines of the HTR ground truth are short and hyphenated at line ends; verse is rejected by the line-length test
        out.append(Doc('fr', 'Gallicorpora_French_HTR', fn, title[:160], f'anon:{fn}', yr, str(yr), url, lic('public domain', 'Gallicorpora_French_HTR'),
                       clean_generic(d['raw']), 'fr_gall', 'work', 0.10))
    return out


POETS_DE = re.compile(r'Gryphius|Luther|Greiffenberg|Hans Sachs|Speratus|Nicolai|Stegmann|Michael Franck|Rist|Opitz|Brant|Anton Ulrich|Hoffmannswaldau|Fleming|Logau|Spee|Neander')
VERSE_TITLE = re.compile(r'\b(Lied|Lieder|Gedicht|Ode|Sonett|Sonnet|Psalm|Grab-?schr|Hymn|Gesang|Gebet|Epigramm|Reim|Spruch|Volkslied|Tanz|Fabel)', re.I)


def load_ws_renaissance(root: Path, coll: str, lang: str, pool: str, host: str, folder: str):
    out = []
    for d in read_jsonl(find_one(root, f'{folder}/*/*.jsonl')):
        parts = d['source'].split('—')
        author = parts[-1].strip() if len(parts) > 1 else f'anon:{d["filename"]}'
        fn = d['filename']
        title = re.sub(r'\s*\[[^\]]*\]\s*$', '', ':'.join(parts[0].split(':')[1:]).strip() or fn.replace('_', ' '))
        yr = year_of(d['year'])
        precision = 'wikisource_metadata'
        if lang == 'de':
            if POETS_DE.search(author) or VERSE_TITLE.search(title) or VERSE_TITLE.search(fn):
                continue
            if not yr and fn.startswith('Zimmerische_Chronik'):
                yr, precision = 1566, 'work'                  # written 1558-1566
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        wiki = re.sub(r'\s+', '_', title) if title else fn
        out.append(Doc(lang, coll, fn, title[:160], author, yr, str(yr), f'https://{host}/wiki/{wiki}', lic('public domain', coll), clean_generic(d['raw']), pool, precision))
    return out


def load_latin_ws(root: Path):
    out = []
    for d in read_jsonl(find_one(root, 'raw__LatinWikisource/*/*.jsonl')):
        yr = year_of(d['year'])
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        fn = d['filename']
        m = re.match(r'lawiki_(\d+)_(.*)$', fn)
        pid, slug = (m.group(1), m.group(2)) if m else ('', fn)
        out.append(Doc('la', 'LatinWikisource', fn, slug.replace('_', ' ')[:160], f'anon:{fn}', yr, str(yr),
                       f'https://la.wikisource.org/wiki/?curid={pid}' if pid else 'https://la.wikisource.org/', lic('public domain', 'LatinWikisource'),
                       clean_generic(d['raw']), 'la_ws'))
    return out


def load_textgrid(root: Path):
    out = []
    for d in read_jsonl(find_one(root, 'raw__TextGrid_German/*/*.jsonl')):
        yr = year_of(d['year'])
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        fn = d['filename']
        a0 = fn.split('_literatur_')[0]
        rest = fn.split('_literatur_m_')[-1]
        rest = rest[len(a0) + 1:] if rest.startswith(a0) else rest
        genre = rest.split('_')[0]
        if genre not in ('romane', 'roman', 'prosa', 'schwanksammlung', 'satirischer', 'satire', 'predigtliteratur', 'dialoge', 'volksbuch', 'theoretische', 'werk') or a0.startswith('rabelais'):
            continue                                      # verse, plays and a later German translation of Rabelais are left out
        author = fn.split('_literatur_')[0].replace('_', ' ') if '_literatur_' in fn else f'anon:{fn}'
        title = fn.split('_literatur_m_')[-1].replace('_', ' ')[:160]
        out.append(Doc('de', 'TextGrid_German', fn, title, author, yr, str(yr), 'https://textgridrep.org/', lic('public domain', 'TextGrid_German'),
                       clean_generic(d['raw']), 'de_tg'))
    return out


def load_lampeter(root: Path):
    out = []
    for d in read_jsonl(find_one(root, 'raw__Lampeter_English/*/*.jsonl')):
        yr = year_of(d['year'])
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        fn = d['filename']
        out.append(Doc('en', 'Lampeter_English', fn, f'Lampeter Corpus tract {fn}', f'anon:{fn}', yr, str(yr),
                       'https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/3193', lic('public domain', 'Lampeter_English'), clean_generic(d['raw']),
                       'en_lamp'))
    return out


def load_hakluyt(root: Path):
    out = []
    for d in read_jsonl(find_one(root, 'raw__Hakluyt_English/*/*.jsonl')):
        if not d['filename'].startswith('hakluyt_principal'):
            continue
        m = re.search(r'vol(\d+)_pg(\d+)', d['filename'])
        vol, pg = (m.group(1), m.group(2)) if m else ('', '')
        out.append(Doc('en', 'Hakluyt_English', d['filename'], f'Hakluyt, Principal Navigations (1598-1600), vol. {int(vol) if vol else "?"}', 'Richard Hakluyt',
                       1599, '1598-1600', f'https://www.gutenberg.org/ebooks/{pg}' if pg else 'https://www.gutenberg.org/', lic('public domain', 'Hakluyt_English'),
                       clean_generic(d['raw']), 'en_hak'))
    return out


def load_sga(root: Path):
    out = []
    for sub, coll in (('raw__SpanishGoldenAgeProse_WS', 'SpanishGoldenAgeProse_WS'), ('raw__WikisourceES_Renaissance', 'WikisourceES_Renaissance')):
        for d in read_jsonl(find_one(root, f'{sub}/*/*.jsonl')):
            yr = year_of(d['year'])
            if not yr or not (YEAR_LO <= yr <= YEAR_HI) or d.get('fetch_status') not in (None, 'ok'):
                continue
            title = d.get('title') or d['filename']
            out.append(Doc('es', coll, d['filename'], title[:160], f'anon:{d["filename"]}', yr, str(yr), 'https://es.wikisource.org/wiki/' + re.sub(r'\s+', '_', title),
                           lic('public domain', coll), clean_generic(d['raw']), 'es_ws'))
    return out


# ---------------------------------------------------------------- Italian

def load_ws_it(root: Path, exclude: set[str]):
    out = []
    pdir = root / 'ws_it' / 'ws_it_renaissance'
    poetry = re.compile(r'canti|rime|sonett|canzon|gerusalemme|aminta|orlando|stanze|ecloga|capitolo|amorosa|caccia|dubbj|cantate|fior_|pianta|chi_|datemi|come_nell|'
                        r'gi__vidi|ove_son|poi_che|ligiadro|decennali|alcesto|angelica|clizia|commedie|favola_di_belfagor|canzoniere|rerum|laude|madrigal|ottave', re.I)
    for p in sorted(pdir.glob('*.txt')):
        if p.stem in exclude or poetry.search(p.stem) or re.match(r'(O_|Il_divino|Satire|Lirica|Il_Trionfo|Ai_posteri|Elogi_di_uomini)', p.stem):
            continue
        raw = p.read_text(encoding='utf-8', errors='replace')
        if '◄' in raw or 'Questo testo fa parte' in raw:
            continue
        title = (re.search(r'<dc:title>\s*(.*?)\s*</dc:title>', raw) or [None, p.stem])[1]
        creator = (re.search(r'<dc:creator[^>]*>\s*(.*?)\s*</dc:creator>', raw) or [None, ''])[1] or f'anon:{p.stem}'
        date = (re.search(r'<dc:date>\s*(.*?)\s*</dc:date>', raw) or [None, ''])[1]
        ident = (re.search(r'title=([^&<\s]+)', raw) or [None, p.stem])[1]
        yr = year_of(date)
        precision = 'wikisource_metadata'
        ty = year_of(title)
        if ty and (not yr or abs(ty - yr) > 0):
            yr, precision = ty, 'title'
        mc = re.fullmatch(r'\s*(XVI|XVII)\s+secolo\s*', date or '')
        if not yr and mc:
            yr, precision = (1550 if mc.group(1) == 'XVI' else 1650), 'century'
        i = raw.find('Informazioni sulla fonte del testo')
        body = raw[i + len('Informazioni sulla fonte del testo'):] if i >= 0 else raw
        body = html.unescape(re.sub(r'&#(\d+);', lambda mm: chr(int(mm.group(1))), body))
        body = re.sub(r'(?<=\s)\d{1,3}(?=\s+[a-zà-ÿ])', ' ', body)      # verse / line numbers inside the running text
        body = re.sub(r'\[\s*\d+\s*\]', ' ', body)
        body = re.sub(r'\s+', ' ', body).strip()
        if not yr or not (YEAR_LO <= yr <= YEAR_HI) or len(body.split()) < 200:
            continue
        out.append(Doc('it', 'Wikisource-IT-Renaissance', p.stem, title[:160], creator, yr, date or str(yr), f'https://it.wikisource.org/wiki/{ident}',
                       'it.wikisource.org, CC BY-SA 3.0 / GFDL (text archive: nas__ws_it_renaissance)', body, 'it_ws', precision))
    return out


def load_galileo(root: Path, exclude: set[str]):
    out = []
    p = find_one(root, 'galileo/Galileo_Opere_Favaro/Galileo_Opere_Favaro.jsonl')
    for d in read_jsonl(p):
        if (d.get('language') or 'it').lower()[:2] != 'it' or d['filename'] in exclude:
            continue
        txt = re.sub(r'\s+', ' ', d['raw'])
        if len(txt.split()) < 700:
            continue
        yr = year_of(d.get('year')) or 1632
        title = d['filename'].replace('galileo_favaro_', '').replace('_', ' ')
        out.append(Doc('it', 'Galileo-Favaro-Wikisource', d['filename'], title[:160], 'Galileo Galilei', yr, str(yr) if year_of(d.get('year')) else '1590-1642',
                       'https://it.wikisource.org/wiki/Autore:Galileo_Galilei', 'it.wikisource.org, public domain text (Favaro edition) / CC BY-SA 3.0 (transcription)',
                       txt, 'it_gal', 'author' if not year_of(d.get('year')) else 'work', 0.12))
    return out


def load_bibit(root: Path, exclude: set[str]):
    out = []
    for f in sorted(glob.glob(str(root / 'bibit' / '*.xml'))):
        doc = Path(f).stem
        if doc in exclude or doc.lower() in exclude:
            continue
        try:
            r = etree.parse(f, etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)).getroot()
        except Exception:
            continue
        if ''.join(r.xpath('string(//language/@id)')) != 'ita':
            continue
        if has_place_markup(r):
            continue
        nb = len(r.xpath('//body//p'))
        nl = len(r.xpath('//body//l'))
        if nb < 5 or nl > nb:
            continue                                    # verse or drama
        bibl = r.xpath('//sourceDesc/bibl')[0]
        yr = year_of(''.join(bibl.xpath('string(date)')))
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        title = re.sub(r'\s+', ' ', ''.join(r.xpath('string(//titleStmt/title)')).strip())
        author = re.sub(r'\s+', ' ', ''.join(r.xpath('string(//titleStmt/author)')).strip()) or f'anon:{doc}'
        body = r.find('.//body')
        buf = []

        def walk(el):
            tag = el.tag if isinstance(el.tag, str) else None
            if tag in ('note', 'pb', 'cb', 'fw', 'gap', 'argument'):
                return
            if tag in ('p', 'head', 'item', 'salute', 'dateline', 'closer', 'signed', 'sp', 'speaker', 'l', 'lg', 'div1', 'div2', 'div3', 'div4') and buf and not buf[-1].endswith('\n'):
                buf.append('\n')
            if el.text:
                buf.append(el.text)
            for c in el:
                walk(c)
                if c.tail:
                    buf.append(c.tail)
            if tag in ('p', 'head', 'item'):
                buf.append('\n')

        walk(body)
        txt = clean_generic(re.sub(r'[ \t\r\f\v]+', ' ', ''.join(buf)))
        if len(txt.split()) < 800:
            continue
        out.append(Doc('it', 'BibIt', doc, title[:160], author, yr, str(yr), f'https://bibliotecaitaliana.it/', lic('attribution_noncommercial', 'BibIt', 'T2'),
                       txt, 'it_bibit', 'edition'))
    return out


# curated early modern prose of LiberLiber: stem -> (author, title, first year, last year)
LIBERLIBER = {
    '400_leonardo_frammenti_letterari_e_filosofici': ('Leonardo da Vinci', 'Frammenti letterari e filosofici', 1480, 1519),
    '400_sanudo_i_diarii': ('Marino Sanudo', 'I diarii (vol. 1)', 1496, 1533),
    '400_sanudo_i_diarii_tomo_ii': ('Marino Sanudo', 'I diarii (tomo II)', 1496, 1533),
    '400_sanudo_i_diarii_tomo_iv': ('Marino Sanudo', 'I diarii (tomo IV)', 1496, 1533),
    '400_sanudo_i_diarii_tomo_vii': ('Marino Sanudo', 'I diarii (tomo VII)', 1496, 1533),
    '500_accetto_della_dissimulazione_onesta': ('Torquato Accetto', 'Della dissimulazione onesta', 1641, 1641),
    '500_ariosto_lettere': ('Ludovico Ariosto', 'Lettere', 1498, 1532),
    '500_brancaccio_i_carichi_militari': ('Lelio Brancaccio', 'I carichi militari', 1582, 1582),
    '500_bruno_de_la_causa_principio_et_uno': ('Giordano Bruno', 'De la causa, principio et uno', 1584, 1584),
    '500_buonarroti_le_lettere_di_michelangelo_buonarroti': ('Michelangelo Buonarroti', 'Le lettere', 1496, 1563),
    '500_croce_bertoldo_e_bertoldino': ('Giulio Cesare Croce', 'Bertoldo e Bertoldino', 1606, 1608),
    '500_della_casa_galateo_overo_de_costumi': ('Giovanni Della Casa', 'Galateo overo de costumi', 1558, 1558),
    '500_fazello_le_due_deche_dell_historia_di_sicilia': ('Tommaso Fazello', "Le due deche dell'historia di Sicilia", 1573, 1573),
    '500_firenzuola_discacciamento_de_le_nuove_lettere': ('Agnolo Firenzuola', 'Discacciamento de le nuove lettere', 1524, 1524),
    '500_i_marmi': ('Anton Francesco Doni', 'I marmi', 1552, 1553),
    '500_ingrassia_informatione_del_pestifero_morbo': ('Giovanni Filippo Ingrassia', 'Informatione del pestifero morbo', 1576, 1576),
    '500_machiavelli_discorsi_sopra_la_prima_deca_di_tito_livio': ('Niccolò Machiavelli', 'Discorsi sopra la prima deca di Tito Livio', 1531, 1531),
    '500_machiavelli_il_principe': ('Niccolò Machiavelli', 'Il principe', 1532, 1532),
    '500_pigafetta_relazione_del_primo_viaggio_intorno_al_mondo': ('Antonio Pigafetta', 'Relazione del primo viaggio intorno al mondo', 1524, 1525),
    '500_ringhieri_cento_giuochi_liberali': ('Innocentio Ringhieri', 'Cento giuochi liberali', 1551, 1551),
    '500_sarpi_scritti_filosofici_inediti': ('Paolo Sarpi', 'Scritti filosofici inediti', 1585, 1620),
    '500_straparola_le_piacevoli_notti': ('Giovanni Francesco Straparola', 'Le piacevoli notti', 1550, 1553),
    '500_varthema_itinerario': ('Ludovico de Varthema', 'Itinerario', 1510, 1510),
    '600_alberti_viaggio_a_costantinopoli': ('Tommaso Alberti', 'Viaggio a Costantinopoli', 1609, 1621),
    '600_boccalini_de_ragguagli_di_parnaso': ('Traiano Boccalini', 'De ragguagli di Parnaso', 1612, 1613),
    '600_borri_relatione_della_nuova_missione': ('Cristoforo Borri', 'Relatione della nuova missione', 1631, 1631),
    '600_gallo_le_vinti_giornate_dell_agricoltura': ('Agostino Gallo', "Le vinti giornate dell'agricoltura", 1569, 1572),
    '600_historia_di_papa_alessandro': ('Anonimo', 'Historia di Papa Alessandro e di Federico Barbarossa', 1564, 1564),
    '600_negri_il_viaggio_settentrionale': ('Francesco Negri', 'Il viaggio settentrionale', 1700, 1700),
    '500_arcadia': ('Jacopo Sannazaro', 'Arcadia', 1504, 1504),
    '600_galilei_dialogo_sopra_i_due_massimi_sistemi': ('Galileo Galilei', 'Dialogo sopra i due massimi sistemi', 1632, 1632),
    '400_leonardo_trattato_della_pittura': ('Leonardo da Vinci', 'Trattato della pittura', 1651, 1651),
}


def odt_text(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        x = z.read('content.xml').decode('utf-8', 'replace')
    x = re.sub(r'</text:(p|h)>', '\n', x)
    x = re.sub(r'<text:line-break/>', '\n', x)
    x = re.sub(r'<text:s[^>]*/>', ' ', x)
    x = re.sub(r'<[^>]+>', '', x)
    return html.unescape(x)


def epub_text(path: str) -> str:
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.lower().endswith(('.xhtml', '.html', '.htm')):
                x = z.read(n).decode('utf-8', 'replace')
                x = re.sub(r'</(p|h\d|div|li)>', '\n', x)
                x = re.sub(r'<br\s*/?>', '\n', x)
                x = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', x, flags=re.S)
                out.append(html.unescape(re.sub(r'<[^>]+>', '', x)))
    return '\n'.join(out)


def zip_txt(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        n = [x for x in z.namelist() if x.lower().endswith('.txt')]
        raw = z.read(n[0])
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1', 'replace')


def load_liberliber(root: Path):
    out = []
    fdir = root / 'liberliber' / 'liberliber' / 'files'
    for stem, (author, title, y0, y1) in LIBERLIBER.items():
        cands = [fdir / f'LiberLiber_{stem}{ext}' for ext in ('.odt', '.epub', '.zip')]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            continue
        try:
            t = odt_text(str(p)) if p.suffix == '.odt' else epub_text(str(p)) if p.suffix == '.epub' else zip_txt(str(p))
        except Exception as e:                       # a corrupt member in the archive copy: the work is simply not used
            print('LiberLiber: skipped', stem, repr(e)[:80], file=sys.stderr)
            continue
        t = t.replace('\r', '')
        n = len(t)
        t = t[int(n * 0.04): int(n * 0.96)]            # skip the e-book front and back matter
        t = re.sub(r'\[\d+\]|\*+', ' ', t)
        t = clean_generic(t)
        if len(t.split()) < 1500:
            continue
        date = str(y0) if y0 == y1 else f'{y0}-{y1}'
        out.append(Doc('it', 'LiberLiber', f'LiberLiber_{stem}', title, author, y0, date, 'https://www.liberliber.it/online/', lic('public domain', 'LiberLiber'),
                       t, 'it_ll', 'work'))
    return out


# ---------------------------------------------------------------- Spanish

ES_TABLE = {   # file prefix -> (author, year or range)
    'AgustindeRojas': ('Agustín de Rojas Villandrando', 1603), 'Avellaneda': ('Alonso Fernández de Avellaneda', 1614), 'Carvajal': ('Mariana de Carvajal y Saavedra', 1663),
    'Castillo': ('Alonso de Castillo Solórzano', 1620), 'Cervantes': ('Miguel de Cervantes', 1605), 'CervantesSalazar': ('Francisco Cervantes de Salazar', 1566),
    'FrayLuisdeLeon': ('Fray Luis de León', 1583), 'Lazarillo': ('Anónimo (Lazarillo de Tormes)', 1554), 'Lope': ('Lope de Vega', 1620),
    'MateoAleman': ('Mateo Alemán', 1599), 'Mendoza': ('Diego Hurtado de Mendoza', 1627), 'Montalban': ('Juan Pérez de Montalbán', 1624), 'Otalora': ('Juan de Otálora', 1580),
    'Pasamonte': ('Jerónimo de Pasamonte', 1605), 'SuarezFigueroa': ('Cristóbal Suárez de Figueroa', 1617), 'Tirso': ('Tirso de Molina', 1624),
    'Ubeda': ('Francisco López de Úbeda', 1605), 'Valdes': ('Alfonso de Valdés', 1529), 'Zayas': ('María de Zayas', 1637),
}
ES_YEARS = {'Cervantes_Quijote-I': 1605, 'Cervantes__Quijote-II': 1615, 'Cervantes_Galatea': 1585, 'Cervantes_Persiles': 1617, 'Valdes_Mercurio': 1529, 'Valdes_Roma': 1527,
            'Valdes_Lengua': 1535, 'Zayas_Tarde-llega-el-desengaño': 1647, 'Zayas_Estragos-que-causa-el-vicio': 1647, 'Zayas_Verdugo-de-su-esposa': 1647,
            'Zayas_Inocencia-castigada': 1637, 'Lope_dorotea': 1632, 'Lope_Fortunas-de-Diana': 1621, 'Lope_Prudente-venganza': 1624, 'Lope_Guzmán-el-Bravo': 1624,
            'Lope_Desdicha-por-la-honra': 1624, 'Lazarillo_Segunda': 1555, 'Cervantes_numancia': 1585}


def load_estilometria(root: Path):
    out = []
    base = root / 'estilometria' / 'estilometria' / 'Siglo_de_Oro'
    for sub in ('Novelas', 'Entremeses'):
        for p in sorted((base / sub).glob('*.txt')):
            stem = p.stem
            if sub == 'Entremeses' or stem in ('Cervantes_numancia',):
                continue                                # verse or plays; the prose novels are what this set needs
            prefix = stem.split('_')[0]
            if prefix not in ES_TABLE:
                continue
            author, yr = ES_TABLE[prefix]
            if stem == 'Valdes_Lengua':
                author = 'Juan de Valdés'
            yr = ES_YEARS.get(stem, yr)
            raw = p.read_text(encoding='utf-8', errors='replace')
            txt = clean_generic(raw)
            if len(txt.split()) < 1500:
                continue
            title = stem.split('_', 1)[-1].replace('-', ' ').replace('_', ' ')
            out.append(Doc('es', 'Estilometria_SigloOro', stem, title[:160], author, yr, str(yr), 'https://github.com/7PartidasDigital/NovelaBarroca',
                           lic('CC BY', 'Estilometria_SigloOro'), txt, 'es_est', 'work'))
    return out


# ---------------------------------------------------------------- selection

def build_lexicon(docs):
    from ratlas.ner.ner2_passages import ALPHA, fold
    lex = Counter()
    for d in docs:
        lex.update(fold(w) for w in re.findall(rf'[{ALPHA}]+', d.text))
    return lex


def oov_rate(seg, lex):
    from ratlas.ner.ner2_passages import ALPHA, fold
    ws = [fold(w) for w in re.findall(rf'[{ALPHA}]+', seg)]
    ws = [w for w in ws if len(w) >= 4]
    return sum(lex[w] < 3 for w in ws) / max(1, len(ws))



def period_bin(y: int) -> int:
    return 0 if y < 1550 else 1 if y < 1600 else 2 if y < 1650 else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seed-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--v1-docs', help='text file: one v1 source path per line (data_v1 doc_meta); docs whose file name matches are excluded')
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--reject-docs', help='text file: one doc id per line, read by hand after a first run (verse, biographical notes, ...)')
    ap.add_argument('--report', help='json: per-pool and per-language statistics')
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)

    seed_recs = [json.loads(l) for l in open(a.seed_file, encoding='utf-8') if l.strip()]
    seed_docs = {r['doc'] for r in seed_recs}
    v1_names: set[str] = set()
    if a.v1_docs:
        for l in open(a.v1_docs, encoding='utf-8'):
            b = Path(l.strip()).name
            v1_names.add(re.sub(r'\.(xml|txt|json|jsonl)$', '', b, flags=re.I).lower())
    rejected = {l.split('#')[0].strip() for l in open(a.reject_docs, encoding='utf-8')} if a.reject_docs else set()
    excl_ci = lambda s: s in seed_docs or s.lower() in v1_names or s in rejected

    pools: dict[str, list[Doc]] = {}
    en_fill = load_eebo(root, 'eebotcp_fill', 'en', 'en_fill', 'EEBO-TCP', 'https://quod.lib.umich.edu/e/eebo', lic('public domain', 'EEBO-TCP Phase 1'), seed_docs | v1_names)
    en_p2 = load_eebo(root, 'eebo_phase2', 'en', 'en_p2', 'EEBO-TCP', 'https://quod.lib.umich.edu/e/eebo2', lic('public domain', 'EEBO-TCP Phase 2'), seed_docs | v1_names)
    la_eebo = load_eebo(root, 'latineebo', 'la', 'la_eebo', 'EEBO-TCP-Latin', 'https://quod.lib.umich.edu/e/eebo', lic('public domain', 'EEBO-TCP Latin'), seed_docs | v1_names)
    allpools = [en_fill, en_p2, load_lampeter(root), load_hakluyt(root), la_eebo, load_latin_ws(root), load_ws_it(root, seed_docs | v1_names), load_galileo(root, seed_docs | v1_names),
                load_liberliber(root), load_bibit(root, seed_docs | v1_names), load_estilometria(root), load_sga(root), load_abu(root), load_gallicorpora(root),
                load_ws_renaissance(root, 'WikisourceFR_Renaissance', 'fr', 'fr_ws', 'fr.wikisource.org', 'raw__WikisourceFR_Renaissance'), load_mazarinades(root),
                load_ws_renaissance(root, 'WikisourceDE_Renaissance', 'de', 'de_ws', 'de.wikisource.org', 'raw__WikisourceDE_Renaissance'), load_textgrid(root)]
    for lst in allpools:
        for d in lst:
            if excl_ci(d.doc):
                continue
            pools.setdefault(d.pool, []).append(d)
    for k in pools:
        pools[k].sort(key=lambda d: d.doc)
        rng.shuffle(pools[k])
    print({k: len(v) for k, v in pools.items()}, file=sys.stderr)

    lexicons = {'fr': build_lexicon([d for p in ('fr_abu', 'fr_ws', 'fr_gall') for d in pools.get(p, [])])}

    # quotas of NEW passages per pool (the seed adds en 11, la 11, it 8 with their own docs)
    quota = {
        'en': {'en_fill': 13, 'en_p2': 13, 'en_lamp': 5, 'en_hak': 3},
        'la': {'la_eebo': 22, 'la_ws': 12},
        'it': {'it_ws': 18, 'it_ll': 22, 'it_bibit': 16},
        'es': {'es_est': 27, 'es_ws': 8},
        'fr': {'fr_abu': 8, 'fr_gall': 9, 'fr_ws': 13, 'fr_maz': 5},
        'de': {'de_ws': 18, 'de_tg': 12},
    }
    total_target = {'en': 45, 'la': 45, 'it': 60, 'es': 35, 'fr': 35, 'de': 30}
    seed_by_lang = Counter(r['lang'] for r in seed_recs)
    author_cap = {l: max(1, math.floor(0.12 * total_target[l])) for l in total_target}
    bin_cap = {l: math.ceil((0.8 if l == 'es' else 0.42) * total_target[l]) for l in total_target}

    seen_fp: list[set] = [fingerprints(r['text']) for r in seed_recs]
    sel: list[dict] = []
    stats = defaultdict(Counter)
    per_lang_author = defaultdict(Counter)
    per_lang_bin = defaultdict(Counter)
    for r in seed_recs:
        per_lang_author[r['lang']][r.get('author') or r['doc']] += 1

    def dup(fp):
        for o in seen_fp:
            if len(fp & o) >= 0.05 * max(1, min(len(fp), len(o))):
                return True
        return False

    for lang, qs in quota.items():
        for pool, q in qs.items():
            docs = pools.get(pool, [])
            got = 0
            per_doc = Counter()
            for pass_no in (1, 2):                      # pass 2 only if the pool could not give q passages at one per document
                for d in docs:
                    if got >= q:
                        break
                    if per_doc[d.doc] >= pass_no:
                        continue
                    if per_lang_author[lang][d.author] >= author_cap[lang]:
                        stats[pool]['author_cap'] += 1
                        continue
                    if per_lang_bin[lang][period_bin(d.year)] >= bin_cap[lang]:
                        stats[pool]['period_cap'] += 1
                        continue
                    chk = (lambda seg, _l=lexicons.get(lang), _m=d.max_oov: oov_rate(seg, _l) <= _m) if d.max_oov is not None and lexicons.get(lang) else None
                    got_p = draw_passage(d.text, rng, lang, tries=60, min_stop=d.min_stop, extra_check=chk)
                    if not got_p:
                        stats[pool]['no_passage'] += 1
                        continue
                    off, seg = got_p
                    if d.vocab is None:
                        d.vocab = doc_vocab(d.text, lang)
                    seg2, reps = repair_dropcaps(seg, lang, d.vocab)
                    if reps:
                        stats[pool]['dropcap_repairs'] += len(reps)
                    if len(seg2.split()) < 120:
                        stats[pool]['short_after_repair'] += 1
                        continue
                    fp = fingerprints(seg2)
                    if dup(fp):
                        stats[pool]['duplicate'] += 1
                        continue
                    toks, starts = tokenize(seg2)
                    n = per_doc[d.doc] + 1
                    pid = f'gold2/{lang}/{d.doc}' + (f'#{n}' if n > 1 else '')
                    if n > 1:
                        # the first passage keeps the plain id; the second gets '#2'
                        pass
                    rec = {'id': pid, 'lang': lang, 'collection': d.collection, 'doc': d.doc, 'title': d.title, 'author': d.author, 'date': d.date, 'year': d.year,
                           'date_precision': d.date_precision, 'source_url': d.source_url, 'licence_note': d.licence_note, 'offset_in_doc': off, 'text': seg2,
                           'tokens': toks, 'token_starts': starts, 'dropcap_repairs': [list(x) for x in reps], 'pool': pool}
                    sel.append(rec)
                    seen_fp.append(fp)
                    per_doc[d.doc] += 1
                    per_lang_author[lang][d.author] += 1
                    per_lang_bin[lang][period_bin(d.year)] += 1
                    got += 1
                if got >= q:
                    break
            stats[pool]['selected'] = got
            stats[pool]['quota'] = q
            print(f'{lang} {pool}: {got}/{q} (pool {len(docs)} docs)', file=sys.stderr)

    # the seed passages, unchanged, with the extra metadata of the new schema
    seed_out = []
    for r in seed_recs:
        s = {k: r[k] for k in ('id', 'lang', 'collection', 'doc', 'title', 'licence_note', 'text', 'tokens', 'token_starts')}
        s['seed'] = True
        seed_out.append(s)
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in seed_out + sel:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rep = {'pools': {k: dict(v) for k, v in stats.items()},
           'by_lang': Counter([r['lang'] for r in seed_out + sel]),
           'new_by_lang': Counter(r['lang'] for r in sel),
           'author_cap': author_cap}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, default=dict))
    print(json.dumps(rep, indent=1, default=dict), file=sys.stderr)


if __name__ == '__main__':
    main()
