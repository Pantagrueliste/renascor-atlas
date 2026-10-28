"""Build the candidate passages of the model-selection dev set dev3 (Spanish, Italian, Latin; 120 passages per language after the leak check).

dev3 is separate from gold v2 (final test set) and from test_v2 and the v1 train/dev/test editions. It is used to choose teachers, prompts, thresholds
and models; it is agent-annotated.

Reuses the loaders and the passage rules of ``ner2_build_passages.py`` (gold v2): 120-200 words, >= 8 capitalised words, <= 10 % digits, diplomatic
reading kept, hyphens rejoined, detached drop capitals repaired. Adds sources that gold v2 did not use: NOSCEMUS (neo-Latin, OCR), Salamanca TEI
(Latin and Spanish treatises), IMPACT-es ground truth (Spanish printed prose), Spanish Wikisource prose (Quevedo, Gracian).
No model output is used anywhere. Nothing is read from RQ/data/raw or from the frozen database.

    python scripts/ner3_dev3_build.py --root ROOT --excl-docs excl_docs.txt --out cand_dev3.jsonl --report cand_dev3_report.json [--per-lang 156]

ROOT layout: see scripts/ner3_dev3_extract.sh (same layout as the gold v2 build).
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import math
import random
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
import ner2_build_passages as B  # noqa: E402
from ratlas.ner.ner2_passages import (SENT, doc_vocab, fingerprints, fold, guess_lang, is_verse, passage_ok, rejoin_eol_hyphens,  # noqa: E402
                                     repair_dropcaps, stopword_share, tokenize)

Doc = B.Doc
YEAR_LO, YEAR_HI = 1450, 1700
TEI = B.TEI

LICENCE = {  # collection -> (licence tag, tier), from the archive attribution table (metadata field only)
    'NOSCEMUS': ('cc_by', 'T1'), 'Salamanca': ('CC-BY-4.0', 'T1'), 'IMPACTes_OriginalImprint': ('cc_by', 'T1'), 'WikisourceES_Renaissance': ('public_domain', 'T1'),
}


def licnote(coll):
    tag, tier = LICENCE[coll]
    return f'{tag} ({coll}, text archive {tier})'


# ---------------------------------------------------------------- genre (heuristic, from titles and collections)

GENRE_RULES = [
    ('letters', r'\b(lettere?|lettera|carta|cartas|epistol\w*|epistula\w*|briefe?|letters?|correspond\w*|carteggio)\b'),
    ('sermon_theology', r'\b(serm\w+|concio\w*|homil\w+|predic\w+|theolog\w+|teolog\w+|sacrament\w+|confess\w+|penitent\w+|biblia|psalm\w+|catechis\w+|ecclesi\w+|fide\b|dogma\w*|passion\w*|doctrina|virtu\w*|sancti|santa|santo|sant)\b'),
    ('law_politics', r'\b(iur\w+|iust\w+|lex\b|leges|legum|ius\b|derecho|contrat\w+|cambios|usur\w+|politic\w+|principe|principi|discorsi|republic\w+|imperio|encomienda\w*|indiana|indias|gobiern\w+|decret\w+|bull\w*|tractatus de potest\w+)\b'),
    ('history_chronicle', r'\b(histor\w+|cr[oó]nic\w+|annal\w+|diari\w*|diarii|comment\w+|rerum|gesta\w*|vit[ae]\b|vida|vite|relaci[oó]n\w*|relazione|relatione|viaje|viaggio\w*|itinerari\w+|navigat\w+|descript\w+|storia|istor\w+)\b'),
    ('science_medicine', r'\b(medic\w+|anatom\w+|plant\w+|herb\w+|phys\w+|fisic\w+|astronom\w+|mathemat\w+|geometr\w+|chym\w+|chem\w+|botan\w+|animal\w+|corpor\w+|morb\w+|pest\w+|agricoltura|dialogo sopra|sidereus|natur\w+|mundi|stella\w*|cometa\w*|mari\b|oculo\w*|ephemerid\w+|magia|mechanic\w+|scienz\w+|agricult\w+|ajedrez|axedrez|arith\w+)\b'),
    ('oration_dialogue', r'\b(orati\w+|oraci\w+|orazion\w+|discurso\w*|discorso\w*|dialog\w+|colloqui\w+|coloquio\w*|ragguagli|disputat\w+|declamat\w+|panegyr\w+)\b'),
    ('fiction_satire', r'\b(novel\w+|cuento\w*|novella\w*|sue[nñ]os|sueño|buscon|buscón|marmi|notti|fabula\w*|satir\w+|facet\w+|bertoldo|arcadia|lazarillo|persiles|galatea|quijote|guzm[aá]n|pasajero|fortunas|estafa|fregona|fuerza|castigo|engañ\w+|desengañ\w+|celos|amar|envidioso|hermosa|prudente|industria|fantasma|mayor confusi\w+|mundo por|alguacil|tenaza|capitulaciones|execraci\w+)\b'),
    ('treatise_essay', r'\b(trattato|tratado|traite|discorso|galateo|dissimulazione|nombres|perfecta casada|politico|político|lengua|giuochi|causa|principio|arte\b|ars\b|institut\w+|elementa|de ratione|de l\w+)\b'),
]
GENRE_BY_POOL = {
    'es_est': 'fiction_satire', 'es_sga': 'fiction_satire', 'es_wsn': 'fiction_satire', 'es_imp': 'treatise_essay', 'es_sal': 'law_politics',
    'la_nos': 'science_medicine', 'la_eebo': 'sermon_theology', 'la_ws': 'law_politics', 'la_sal': 'law_politics',
    'it_ws': 'treatise_essay', 'it_ll': 'history_chronicle', 'it_bibit': 'treatise_essay', 'it_gal': 'science_medicine',
}


def guess_genre(title: str, pool: str, author: str = '') -> tuple[str, str]:
    t = f'{title} {author}'.lower()
    for g, pat in GENRE_RULES:
        if re.search(pat, t):
            return g, 'title_heuristic'
    return GENRE_BY_POOL.get(pool, 'other_prose'), 'collection_default'


# ---------------------------------------------------------------- new loaders

def tei_choice_text(el):
    """Text of a TEI element tree for printed prose: expansions kept, notes, page breaks and running heads dropped, line-end hyphens rejoined."""
    buf: list[str] = []
    skip = {'note', 'pb', 'fw', 'gap', 'figure', 'teiHeader', 'cb', 'index', 'front', 'back', 'sic', 'abbr', 'orig', 'surplus', 'del', 'milestone'}
    para = {'p', 'head', 'item', 'salute', 'dateline', 'closer', 'signed', 'trailer', 'label', 'div', 'argument', 'byline'}

    def walk(e):
        tag = etree.QName(e).localname if isinstance(e.tag, str) else None
        if tag is None:
            return
        if tag in skip:
            return
        if tag == 'ref' and (e.get('type') or '').startswith('note'):
            return
        if tag == 'lb':
            buf.append('\u0001' if (e.get('break') == 'no' or e.get('rendition') == '#hyphen') else ' ')
            return
        if tag == 'choice':
            pick = None
            for want in ('expan', 'reg', 'corr'):
                for c in e:
                    if isinstance(c.tag, str) and etree.QName(c).localname == want:
                        pick = c
                        break
                if pick is not None:
                    break
            if pick is not None:
                walk_inner(pick)
            return
        if tag in para and buf and not buf[-1].endswith('\n'):
            buf.append('\n')
        walk_inner(e)
        if tag in para:
            buf.append('\n')

    def walk_inner(e):
        if e.text:
            buf.append(e.text)
        for c in e:
            walk(c)
            if c.tail:
                buf.append(c.tail)

    walk(el)
    t = ''.join(buf)
    t = re.sub(r'\u0001\s*', '', t)
    t = re.sub(r'[ \t\r\f\v]+', ' ', t)
    t = re.sub(r'\n\s*', '\n', t)
    return t.strip()


def load_salamanca(root: Path):
    out_la, out_es = [], []
    for f in sorted(glob.glob(str(root / 'salamanca/svsal-tei-master/W*.xml'))):
        doc = Path(f).stem
        r = etree.parse(f).getroot()
        xp = lambda q: ''.join(r.xpath(q)).strip()
        d1 = xp('string(//*[local-name()="imprint"]/*[local-name()="date"][@type="firstEd"]/@when)')
        d2 = xp('string(//*[local-name()="imprint"]/*[local-name()="date"][@type="thisEd"]/@when)')
        yr = B.year_of(d1) or B.year_of(d2)
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
        title = re.sub(r'\s+', ' ', xp('string(//*[local-name()="monogr"]/*[local-name()="title"][@type="short"])')) or doc
        author = re.sub(r'\s+', ' ', xp('string(//*[local-name()="monogr"]/*[local-name()="author"]/*[local-name()="persName"]/@key)')) or f'anon:{doc}'
        texts = r.xpath('//*[local-name()="text"]')
        if not texts:
            continue
        bodies = [c for c in texts[0] if isinstance(c.tag, str) and etree.QName(c).localname == 'body']
        if not bodies:
            continue
        txt = tei_choice_text(bodies[0])
        txt = B.clean_generic(txt)
        if len(txt.split()) < 2000:
            continue
        mid = len(txt) // 2
        lang = guess_lang(txt[mid:mid + 40000])
        if lang not in ('la', 'es'):
            continue
        d = Doc(lang, 'Salamanca', doc, title[:160], author, yr, str(yr) + (f' (first ed.; this ed. {d2})' if d2 and d2 != d1 else ''), 'https://github.com/digicademy/svsal-tei',
                licnote('Salamanca'), txt, f'{lang}_sal', 'work' if d1 else 'edition')
        d.extra = {'work': doc, 'verse': 'ocr'}
        (out_la if lang == 'la' else out_es).append(d)
    return out_la, out_es


NOSC_NAME = re.compile(r'^(?P<author>.+?)_-_(?P<title>.+?)__(?P<place>.+?)\s*_?(?P<year>1[3-9]\d\d)(?:_pdf)?\.txt$', re.S)


def load_noscemus(root: Path):
    out = []
    zp = glob.glob(str(root / 'noscemus' / '**' / '*.zip'), recursive=True)
    if not zp:
        return out
    z = zipfile.ZipFile(zp[0])
    for info in z.infolist():
        if info.file_size < 60000:
            continue
        parts = info.filename.split('/')
        if len(parts) < 3 or not parts[-1].endswith('.txt'):
            continue
        folder, name = parts[-2], parts[-1]
        m = re.match(r'^(?P<a>.+?)_-_(?P<t>.+)__(?P<p>.+?)_+(?P<y>\d{4})_pdf\.txt$', name)
        if not m:
            continue
        yr = int(m.group('y'))
        if not (YEAR_LO <= yr <= YEAR_HI):
            continue
        author = re.sub(r'\s+', ' ', m.group('a').replace('_', ' ')).strip()
        title = re.sub(r'\s+', ' ', m.group('t').replace('_', ' ')).strip()
        raw = z.read(info).decode('utf-8', 'replace')
        n = len(raw)
        mid = raw[n // 2: n // 2 + 40000]
        if guess_lang(mid) != 'la' or stopword_share(mid, 'la') < 0.12:
            continue
        body = raw[int(n * 0.04): int(n * 0.92)].replace('\r', '')
        body = re.sub(r'[ \t]+', ' ', body)
        txt = B.clean_generic(body)
        if len(txt.split()) < 2500:
            continue
        d = Doc('la', 'NOSCEMUS', f'noscemus_{folder}', title[:160], author, yr, str(yr), 'https://github.com/noscemus', licnote('NOSCEMUS'), txt, 'la_nos', 'work', max_oov=0.16)
        d.extra = {'work': f'noscemus_{folder}', 'verse': 'ocr'}
        out.append(d)
    return out


IMPACT = {  # directory -> (author, title, year, precision)
    'AjedrezRuyLopez': ('Ruy López de Segura', 'Libro de la invención liberal y arte del juego del axedrez', 1561, 'work'),
    'ComentariosIncaGarcilaso': ('Inca Garcilaso de la Vega', 'Comentarios reales de los Incas', 1609, 'work'),
    'ObrasStaTeresa': ('Teresa de Jesús', 'Obras de la Santa Madre Teresa de Jesús', 1588, 'work'),
    'CartaSorJuana': ('Sor Juana Inés de la Cruz', 'Carta de Sor Juana Inés de la Cruz', 1691, 'work'),
    'Lazarillo': ('Anónimo', 'Lazarillo de Tormes', 1554, 'work'),
}


def page_lines(path):
    """Paragraph regions of an IMPACT PAGE file (headers, page numbers and marginalia are dropped); one text block per region."""
    r = etree.parse(path).getroot()
    out = []
    for reg in r.iter('{*}TextRegion'):
        if (reg.get('type') or 'paragraph') not in ('paragraph', 'text', 'caption'):
            continue
        u = reg.find('{*}TextEquiv/{*}Unicode')
        if u is not None and u.text and u.text.strip():
            out.append(u.text.strip())
    return out


def load_impactes(root: Path):
    out = []
    for d_, (author, title, yr, prec) in IMPACT.items():
        pages = sorted(glob.glob(str(root / 'impactes' / d_ / '*.xml')))
        if not pages:
            continue
        lines = []
        for p in pages:
            try:
                lines += page_lines(p) + ['']
            except Exception:
                continue
        txt = B.clean_generic('\n'.join(lines))
        if len(txt.split()) < 2500:
            continue
        d = Doc('es', 'IMPACTes_OriginalImprint', f'impactes_{d_}', title, author, yr, str(yr), 'https://github.com/impactcentre/ocr-spanish-gt', licnote('IMPACTes_OriginalImprint'),
                txt, 'es_imp', prec, max_oov=0.06)
        d.extra = {'work': d_, 'verse': 'ocr'}
        out.append(d)
    return out


WSN = {  # Spanish Wikisource prose files of nas__ws_es_renaissance: stem -> (author, title, year)
    'Sue_os_y_discursos_Sue_o_del_infierno': ('Francisco de Quevedo', 'Sueños y discursos: Sueño del infierno', 1627),
    'Sue_os_y_discursos_Sue_o_de_la_muerte': ('Francisco de Quevedo', 'Sueños y discursos: Sueño de la muerte', 1627),
    'Sue_os_y_discursos_El_mundo_por_de_dentro': ('Francisco de Quevedo', 'Sueños y discursos: El mundo por de dentro', 1627),
    'Sue_os_y_discursos_El_alguacil_endemoniado': ('Francisco de Quevedo', 'Sueños y discursos: El alguacil endemoniado', 1627),
    'Sue_os_y_discursos_El_Sue_o_del_Juicio_Final': ('Francisco de Quevedo', 'Sueños y discursos: El sueño del juicio final', 1627),
    'Ep_stolas_del_Caballero_de_la_Tenaza': ('Francisco de Quevedo', 'Epístolas del Caballero de la Tenaza', 1625),
    'Cuento_de_cuentos': ('Francisco de Quevedo', 'Cuento de cuentos', 1626),
    'Capitulaciones_matrimoniales_y_Vida_de_Corte_y_oficios_entretenidos_en_ella': ('Francisco de Quevedo', 'Capitulaciones matrimoniales y Vida de corte', 1627),
    'Execraci_n_contra_los_jud_os': ('Francisco de Quevedo', 'Execración contra los judíos', 1633),
    'El_pol_tico__Graci_n_': ('Baltasar Gracián', 'El político don Fernando el Católico', 1640),
}


def load_wsn(root: Path):
    out = []
    for stem, (author, title, yr) in WSN.items():
        p = root / 'ws_es_renaissance' / f'{stem}.txt'
        if not p.exists():
            continue
        raw = html.unescape(p.read_text(encoding='utf-8', errors='replace'))
        raw = re.sub(r'\[\s*editar\s*\]|\beditar\b', ' ', raw)
        i = raw.find('Preliminares')
        txt = B.clean_generic(re.sub(r'\s+', ' ', raw[i + 12:] if 0 <= i < 3000 else raw))
        if len(txt.split()) < 2500:
            continue
        d = Doc('es', 'WikisourceES_Renaissance', f'WS_es_{stem}', title, author, yr, str(yr), 'https://es.wikisource.org/wiki/' + re.sub(r'\s+', '_', title.split(':')[-1].strip()),
                licnote('WikisourceES_Renaissance'), txt, 'es_wsn', 'work')
        d.extra = {'work': stem, 'verse': 'std'}
        out.append(d)
    return out


BIBIT_GENRE = {'Trattati': 'treatise_essay', 'Documenti': 'diplomatic_report', 'Narrativa': 'fiction_satire', 'Lettere ed epistolari': 'letters', 'Oratoria': 'oration_dialogue',
               'Memorialistica': 'history_chronicle', 'Testi storici e storiografici': 'history_chronicle', 'Commenti, traduzioni e volgarizzamenti': 'treatise_essay'}


def load_bibit2(root: Path, exclude: set):
    """Biblioteca Italiana prose (TEI). Differences from ner2_build_passages.load_bibit: no place-markup filter (this edition is not a v1 training edition; the 8-gram check
    decides), and the date of the work falls back on the century code of profileDesc/creation (500 -> 1550, 600 -> 1650; date_precision 'century') when the header has no year."""
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
        cat = (r.xpath('//keywords[@scheme="CGB"]/term/text()') or [''])[0]
        if cat not in BIBIT_GENRE:
            continue                                    # verse, drama, documents without a genre code
        nb = len(r.xpath('//body//p'))
        nl = len(r.xpath('//body//l'))
        if nb < 5 or nl > nb:
            continue
        bibl = r.xpath('//sourceDesc/bibl')
        title = re.sub(r'\s+', ' ', ''.join(r.xpath('string(//titleStmt/title)')).strip())
        author = re.sub(r'\s+', ' ', ''.join(r.xpath('string(//titleStmt/author)')).strip()) or f'anon:{doc}'
        yr = B.year_of(''.join(bibl[0].xpath('string(date)'))) if bibl else None
        prec = 'edition'
        ty = B.year_of(title)
        if ty and YEAR_LO <= ty <= YEAR_HI:
            yr, prec = ty, 'title'
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            cen = ''.join(r.xpath('string(//profileDesc/creation/date)')).strip()
            yr, prec = {'500': (1550, 'century'), '600': (1650, 'century')}.get(cen, (None, ''))
        if not yr or not (YEAR_LO <= yr <= YEAR_HI):
            continue
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
        txt = B.clean_generic(re.sub(r'[ \t\r\f\v]+', ' ', ''.join(buf)))
        if len(txt.split()) < 1500:
            continue
        d = Doc('it', 'BibIt', doc, title[:160], author, yr, str(yr), 'https://bibliotecaitaliana.it/', B.lic('attribution_noncommercial', 'BibIt', 'T2'), txt, 'it_bibit', prec)
        d.extra = {'work': doc, 'verse': 'std', 'genre': BIBIT_GENRE[cat]}
        out.append(d)
    return out


# ---------------------------------------------------------------- passage drawing

def verse_like_ocr(seg_raw: str) -> bool:
    lines = [l.strip() for l in seg_raw.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    lens = sorted(len(l) for l in lines)
    up = sum(1 for l in lines if l[0].isupper()) / len(lines)
    return lens[len(lens) // 2] < 42 and up > 0.6


def draw2(d, rng: random.Random, lang: str, mode: str, lo=120, hi=200, tries=80, min_stop=0.12, extra_check=None, avoid=()):
    text = d.text
    flat = text.replace('\n', ' ')
    starts = d.extra.get('starts')
    if starts is None:
        starts = d.extra['starts'] = [0] + [m.end() for m in SENT.finditer(flat)]
    if len(starts) < 3:
        return None
    for _ in range(tries):
        j = rng.randrange(len(starts))
        s = starts[j]
        if any(abs(s - a) < 6000 for a in avoid):
            continue
        k = j
        end = s
        while k < len(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(flat)
            k += 1
            if len(flat[s:end].split()) >= lo:
                break
        seg_raw = text[s:end]
        seg = re.sub(r'\s+', ' ', flat[s:end]).strip()
        ok, _ = passage_ok(seg, lang, lo, hi, min_stop=min_stop)
        if not ok:
            continue
        if mode == 'std' and is_verse(seg_raw):
            continue
        if mode == 'ocr' and verse_like_ocr(seg_raw):
            continue
        if extra_check is not None and not extra_check(seg):
            continue
        if not re.search(r'[.!?:;»”"\)]\s*$', seg):
            continue
        return s, seg
    return None


OCR_JUNK = re.compile(r'Digitized by|Google|\u00a7\s*\d+\.\s*\d+')


def period_bin(y: int) -> int:
    return 0 if y < 1550 else 1 if y < 1600 else 2 if y < 1650 else 3


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--excl-docs', required=True, help='text file: one document id / file name per line (gold v2, test_v2, v1 train/dev/test); matched on the file stem, lower case')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report')
    ap.add_argument('--per-lang', type=int, default=156, help='candidates per language (final set: 120 after the leak check)')
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--limit-docs', type=int, default=0, help='debug: keep only N docs per pool')
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    excl = set()
    for l in open(a.excl_docs, encoding='utf-8'):
        l = l.strip()
        if l:
            excl.add(re.sub(r'\.(xml|txt|json|jsonl)$', '', Path(l).name, flags=re.I).lower())
    excl_work_prefix = ['historia de la vida del buscón']          # gold v2 used chapters of this work: the whole work is left out

    def ok_doc(d):
        if d.doc.lower() in excl or re.sub(r'\.(xml|txt)$', '', d.doc, flags=re.I).lower() in excl:
            return False
        return not any(d.title.lower().startswith(p) for p in excl_work_prefix)

    ex = set(excl)
    la_sal, es_sal = load_salamanca(root)
    allp = [
        B.load_eebo(root, 'latineebo/xml', 'la', 'la_eebo', 'EEBO-TCP-Latin', 'https://quod.lib.umich.edu/e/eebo', B.lic('public domain', 'EEBO-TCP Latin'), ex),
        B.load_latin_ws(root), load_noscemus(root), la_sal,
        B.load_estilometria(root), B.load_sga(root), load_wsn(root), es_sal,   # IMPACT-es (load_impactes) left out: ground truth loses 'st'/'ss' ligatures, too many broken words
       
        B.load_ws_it(root, ex), B.load_liberliber(root), load_bibit2(root, ex),
    ]
    pools: dict[str, list] = defaultdict(list)
    n_all = Counter()
    for lst in allp:
        for d in lst:
            n_all[d.pool] += 1
            if not ok_doc(d):
                continue
            if d.pool == 'es_est' and 'Lazarillo' in d.doc:
                continue
            d.extra = d.extra if isinstance(d.extra, dict) and d.extra else {}
            d.extra.setdefault('work', d.doc)
            d.extra.setdefault('verse', 'std')
            if d.pool in ('es_sga',):
                pass
            pools[d.pool].append(d)
    # Spanish Wikisource: the jsonl loader returns pool 'es_ws'; rename for the quota table
    if 'es_ws' in pools:
        pools['es_sga'] = pools.pop('es_ws')
    for k in pools:
        pools[k].sort(key=lambda d: d.doc)
        rng.shuffle(pools[k])
        if a.limit_docs:
            pools[k] = pools[k][:a.limit_docs]
    print({k: (len(v), n_all.get(k)) for k, v in pools.items()}, file=sys.stderr)

    # OOV lexicon of CLEAN Latin (typeset EEBO, Wikisource, Salamanca keyed text; not NOSCEMUS): OCR debris shows as unknown words
    from ratlas.ner.ner2_passages import ALPHA
    lex_clean = Counter()
    for p in ('la_eebo', 'la_ws', 'la_sal'):
        for d in pools.get(p, []):
            lex_clean.update(fold(w) for w in re.findall(rf'[{ALPHA}]+', d.text[:500000]))

    def oov(seg, lang=None):
        ws = [fold(w) for w in re.findall(rf'[{ALPHA}]+', seg)]
        ws = [w for w in ws if len(w) >= 4]
        return sum(lex_clean[w] < 2 for w in ws) / max(1, len(ws))

    # candidate quota per pool = share of the final 120, times the oversampling factor
    final_quota = {
        'la': {'la_nos': 45, 'la_eebo': 30, 'la_ws': 20, 'la_sal': 25},
        'es': {'es_est': 44, 'es_sal': 36, 'es_wsn': 22, 'es_sga': 18},
        'it': {'it_ws': 28, 'it_ll': 36, 'it_bibit': 56},
    }
    over = a.per_lang / 120
    bin_share = [0.18, 0.27, 0.32, 0.23]
    seen_fp: list[set] = []
    sel: list[dict] = []
    stats = defaultdict(Counter)
    author_cap = 24
    work_cap = 6
    genre_cap = int(0.36 * a.per_lang)

    for lang, qs in final_quota.items():
        quota = {p: int(math.ceil(q * over)) for p, q in qs.items()}
        picked = Counter()
        per_doc = Counter()
        per_work = Counter()
        per_author = Counter()
        per_bin = Counter()
        per_genre = Counter()
        offs = defaultdict(list)
        N = a.per_lang
        level = 1

        def try_pick(pool, wanted_bin):
            for d in pools.get(pool, []):
                if wanted_bin is not None and period_bin(d.year) != wanted_bin:
                    continue
                if d.extra.get('fails', 0) >= 8 or per_doc[d.doc] >= level or per_work[d.extra['work']] >= (8 if lang == 'es' else work_cap) or per_author[d.author] >= author_cap:
                    continue
                genre, gsrc = (d.extra['genre'], 'source_category') if d.extra.get('genre') else guess_genre(d.title, d.pool, d.author)
                if per_genre[genre] >= genre_cap:
                    continue
                chk = None
                if lang == 'la':
                    lim = 0.10 if d.pool == 'la_nos' else 0.14
                    chk = (lambda seg, _l=lim: oov(seg) <= _l and not OCR_JUNK.search(seg) and seg.count('¶') <= 1)
                got = draw2(d, rng, lang, d.extra['verse'], min_stop=d.min_stop, extra_check=chk, avoid=offs[d.doc], tries=60)
                if not got:
                    stats[pool]['no_passage'] += 1
                    d.extra['fails'] = d.extra.get('fails', 0) + 1
                    continue
                off, seg = got
                if d.vocab is None:
                    d.vocab = doc_vocab(d.text[:400000], lang)
                seg2, reps = repair_dropcaps(seg, lang, d.vocab)
                if len(seg2.split()) < 120:
                    stats[pool]['short_after_repair'] += 1
                    continue
                fp = fingerprints(seg2)
                if any(len(fp & o) >= 0.05 * max(1, min(len(fp), len(o))) for o in seen_fp):
                    stats[pool]['duplicate'] += 1
                    continue
                toks, starts = tokenize(seg2)
                n = per_doc[d.doc] + 1
                pid = f'dev3/{lang}/{d.doc}' + (f'#{n}' if n > 1 else '')
                rec = {'id': pid, 'lang': lang, 'collection': d.collection, 'doc': d.doc, 'title': d.title, 'author': d.author, 'date': d.date, 'year': d.year,
                       'date_precision': d.date_precision, 'source_url': d.source_url, 'licence_note': d.licence_note, 'offset_in_doc': off, 'text': seg2,
                       'tokens': toks, 'token_starts': starts, 'dropcap_repairs': [list(x) for x in reps], 'pool': pool, 'work': d.extra['work'],
                       'genre': genre, 'genre_source': gsrc, 'period_bin': period_bin(d.year), 'seed': False}
                sel.append(rec)
                seen_fp.append(fp)
                per_doc[d.doc] += 1
                per_work[d.extra['work']] += 1
                per_author[d.author] += 1
                per_bin[period_bin(d.year)] += 1
                per_genre[genre] += 1
                offs[d.doc].append(off)
                picked[pool] += 1
                return True
            return False

        total = 0
        for over_quota in (False, True):
            for level in (1, 2, 3, 4, 5, 6, 7, 8):
                while total < N:
                    # the period bin furthest below its share goes first; within it the pool furthest below its quota
                    bins = sorted(range(4), key=lambda b: (per_bin[b] / max(1, bin_share[b] * N)))
                    pls = sorted([p for p in qs if over_quota or picked[p] < quota[p]], key=lambda p: picked[p] / quota[p])
                    done = False
                    for b in bins + [None]:
                        for p in pls:
                            if try_pick(p, b):
                                done = True
                                break
                        if done:
                            break
                    if not done:
                        break
                    total += 1
                if total >= N:
                    break
            if total >= N:
                break
        print(lang, 'picked', total, dict(picked), 'bins', dict(per_bin), 'genres', dict(per_genre), file=sys.stderr)

    with open(a.out, 'w', encoding='utf-8') as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rep = {'n_by_lang': Counter(r['lang'] for r in sel), 'pools': {k: dict(v) for k, v in stats.items()},
           'pool_docs': {k: len(v) for k, v in pools.items()}, 'n_by_pool': Counter(r['pool'] for r in sel),
           'n_by_bin': Counter((r['lang'], r['period_bin']) for r in sel).most_common(), 'n_by_genre': Counter((r['lang'], r['genre']) for r in sel).most_common()}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(json.dumps(rep, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)), file=sys.stderr)


if __name__ == '__main__':
    main()
