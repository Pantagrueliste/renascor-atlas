"""Gold v3 (RenNER): passages of the human-reviewed test set (LOC + PER), six languages.

gold v3 = the 250 passages of gold v2 (flag ``from_gold2``; their owner-reviewed places kept in ``gold2_loc``; persons to add) + new passages.

New passages: same passage rules as gold v2 / dev3 (``ner2_passages.draw_passage``: sentence start to sentence end, 120-200 words, >= 8 capitalised
words, <= 10 % digits, function-word test, verse refused), at most 2 per document, stratified by language, 50-year period (1450-1700), genre and
collection (collection cap ~10 % within a language, raised only as far as the archive requires). Texts: the NER-layer readings of the text archive
(``ner2_layer_extract.py`` / ``ner2_layer_extra_extract.py``; read-only copies, the archive itself is never touched). No model output is used.

Documents excluded (name, then 8-grams over the whole document): v1 train/dev/test documents (data_v1 doc_meta), gold v2 (all files) and the gold seed,
dev3, test_v2, the silver pool and its reserves, the few-shot examples. Whole collections excluded: v1 editions, London Lives, MAP, private/T3
collections, drama, verse, lexicons, Bibles, the silver quality exclusions. Passage level: >= 2 8-grams shared with any excluded text -> refused.
The 8-gram check against the v1 data and the training mixes (hist_gold, modern_silver, all_private) runs on Hypatia (``ner2_leak_check.py``).

    python scripts/ner3_gold3_build.py excl   --work W
    python scripts/ner3_gold3_build.py cands  --work W
    python scripts/ner3_gold3_build.py select --work W
    (Hypatia: ner2_leak_check.py on W/select_new_plus_reserve.jsonl)
    python scripts/ner3_gold3_build.py final  --work W --leak W/leak_remote.json --out R/gold3
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from ratlas.ner.ner2_passages import draw_passage, fingerprints, fold, tokenize  # noqa: E402
import ner3_silver_pool as SP  # noqa: E402  (8-gram hashes, OCR filters, genre rules: same code as the silver pool)

SEED = 20261024
RQ = Path('/Users/clem/Projects/Papers/RQ')
NER = RQ / 'derived/atlas/ner'
R2 = NER / 'v2'
SCR = Path('/private/tmp/claude-501/-Users-clem-Projects-Papers-Renascor/e0c9bdc2-9edc-46d5-8bde-77513f9fb822/scratchpad')
DB = RQ / 'release/rq_verification_kit_v1.0/data/corpus_master_FROZEN_2026-06-26.sqlite'   # read-only copy of the frozen catalogue
ATTR = RQ / 'release/text_archive_v1.0/ATTRIBUTIONS.tsv'
LANGS = ('la', 'es', 'it', 'fr', 'de', 'en')
TARGET = {'la': 160, 'es': 130, 'it': 130, 'fr': 110, 'de': 110, 'en': 110}
RESERVE_SHARE = 0.30
COLL_SHARE0 = 0.10
GENRE_SHARE = 0.35
AUTHOR_SHARE = 0.04
DOCS_PER_STRATUM = 220          # documents read per (language, collection, 50-year bin) when drawing candidates
TEXT_SOURCES = {                # (language -> shard globs); es/it/la = the silver-pool readings, en/fr/de = NER-layer readings copied from Hypatia
    'la': [SCR / 'ex/texts/la', SCR / 'ex2/texts/la'], 'es': [SCR / 'ex/texts/es', SCR / 'ex2/texts/es'], 'it': [SCR / 'ex/texts/it', SCR / 'ex2/texts/it'],
    'en': [SCR / 'g3/hyp/texts/en', SCR / 'ex1/texts/en'], 'fr': [SCR / 'g3/hyp/texts/fr', SCR / 'ex1/texts/fr'], 'de': [SCR / 'g3/hyp/texts/de', SCR / 'ex1/texts/de'],
}
INVENTORIES = [SCR / 'g3/hyp/docs_all.parquet', SCR / 'ex/docs_all.parquet']
EXTRA_PARTS = [SCR / 'ex1/docs_parts', SCR / 'ex2/docs_parts']

# collections used, per language (everything else is out; reasons in EXCLUDED_WHY)
ALLOWED = {
    'en': {'EEBO_TCP_fill', 'EEBOphase2', 'Lampeter_English', 'Hakluyt_English', 'InterregnumActs', 'StatutesOfTheRealm', 'EvansTCP_American', 'Newton_Project',
           'RoyalSocPre1750', 'EMROC_Recipes', 'PepysDiary'},
    'fr': {'Mazarinades', 'ABU_French', 'Gallicorpora_French_HTR', 'WikisourceFR_Renaissance', 'EEBOphase2', 'FrenchEEBO', 'Nakala', 'CORPUS17_French'},
    'de': {'TextGrid_German', 'WikisourceDE_Renaissance', 'DiBiLit_German', 'DiBiPhil_Renaissance', 'GAMS_BaselRecords', 'GeMi_Midwifery', 'ReF_ENHG', 'EEBOphase2'},
    'la': {'NOSCEMUS', 'CorpusCorp', 'LatinEEBO', 'EMLAP', 'TML', 'LatinWikisource', 'BibIT', 'EEBOphase2', 'Newton_Project', 'Galileo_Opere_Favaro', 'Nakala',
           'Salamanca', 'LiberLiber', 'OSTA'},
    'es': {'Estilometria_SigloOro', 'OSTA', 'WikisourceES_Renaissance', 'TBSO_GH', 'EEBOphase2', 'SpanishGoldenAgeProse_WS', 'Salamanca', 'WikisourceES'},
    'it': {'BibIT', 'LiberLiber', 'Bartoli_Correspondence', 'EEBOphase2', 'WikisourceIT_Renaissance', 'WikisourceIT', 'Galileo_Opere_Favaro'},
}
EXCLUDED_WHY = {
    'v1 edition (train/dev/test of the first NER round)': ['BullingerLetters', 'CKCC_Epistolarium', 'CAMENA', 'CERA', 'CAMENA_CERA_Letters', 'CroALa', 'FreEM', 'DTA',
                                                           'DTA_komplett', 'PostScriptum', 'GerManC', 'Casebooks', 'REED_English', 'BVH_Epistemon_Print', 'Epistemon',
                                                           'Controversia_Confessio', 'Cavriana_Correspondence', 'BDPN_Novohispano', 'QHOD', 'Suriano_Correspondence',
                                                           'OldBaileyOnline', 'ELAN_crenum', 'ELAN_fnvr', 'SSRQ_SwissLaw'],
    'private or T3 (recipe-only) collection': ['CELT', 'CIDTC', 'PCEEC2', 'PerseusEng_Renaissance', 'ABaCus_Baroque'],
    'London Lives / MAP (owner rule)': ['LondonLives', 'London_Lives', 'MAP', 'MapOfEarlyModernLondon'],
    'drama (speaker labels; gold v2 is prose)': ['DraCorEnglish', 'DraCorFrench', 'DraCorGerman', 'EMOTHE', 'OBVIL_HaineTheatre', 'TheatreClassique'],
    'verse': ['EBBA_Ballads', 'CroALa'],
    'lexicons and glossaries, not running text': ['LEME_English'],
    'Bibles (verse numbers, text repeated across editions)': ['WikisourceBibles_Renaissance', 'WikisourceBibles_Italian', 'OpenEDGeS_GermanBibles'],
    'silver-pool quality exclusions (HTR ground truth losing ligatures; modern translations)': ['IMPACTes_OriginalImprint', 'Translatoscope_NeoLat'],
    'Nakala: BVH-Tours texts close to the v1 editions epistemon_bvh / freem_bvh, and modern texts (Balzac, modern translations) in the French part': ['Nakala'],
}
EXCLUDED_COLL = {c for v in EXCLUDED_WHY.values() for c in v}
LL_MAP = re.compile(r'london.?lives|map.?of.?early|\bMAP\b', re.I)

GENRE_DEFAULT_EXTRA = {
    'EEBO_TCP_fill': 'treatise_essay', 'Lampeter_English': 'treatise_essay', 'Hakluyt_English': 'history_chronicle', 'InterregnumActs': 'law_politics',
    'StatutesOfTheRealm': 'law_politics', 'EvansTCP_American': 'sermon_theology', 'RoyalSocPre1750': 'science_medicine', 'EMROC_Recipes': 'science_medicine',
    'PepysDiary': 'history_chronicle', 'Mazarinades': 'law_politics', 'ABU_French': 'fiction_satire', 'Gallicorpora_French_HTR': 'other_prose',
    'WikisourceFR_Renaissance': 'other_prose', 'FrenchEEBO': 'treatise_essay', 'Nakala': 'treatise_essay', 'CORPUS17_French': 'other_prose',
    'TextGrid_German': 'fiction_satire', 'WikisourceDE_Renaissance': 'other_prose', 'DiBiLit_German': 'other_prose', 'DiBiPhil_Renaissance': 'treatise_essay',
    'GAMS_BaselRecords': 'documentary_mixed', 'GeMi_Midwifery': 'science_medicine', 'ReF_ENHG': 'sermon_theology', 'Galileo_Opere_Favaro': 'science_medicine',
}
GENRE_RULES_EXTRA = [   # French, German, English title words (checked after the silver rules)
    ('letters', r'\b(lettres?|missive\w*|sendbrief\w*|epistle\w*)\b'),
    ('sermon_theology', r'\b(sermons?|predigt\w*|catechisme|gebet\w*|andacht\w*|divinit\w*|scripture\w*|gospel|evangel\w*|meditations?|prayers?|christian\w*|sacred|godly|kirch\w*)\b'),
    ('law_politics', r'\b(ordonnance\w*|edict\w*|edit|arrest|declaration|ordnung\w*|mandat\w*|abschied\w*|vertrag\w*|recht\w*|acts?|ordinances?|proclamation\w*|statut\w*|parliament\w*|king\w*|royal\w*|politi\w*|libelle\w*|requeste\w*)\b'),
    ('history_chronicle', r'\b(chroni\w+|histoire\w*|historie\w*|geschicht\w*|relation\w*|voyage\w*|reise\w*|journal\w*|diary|newes|news|zeitung\w*|beschreibung\w*|description|discoverie\w*|travels?|account)\b'),
    ('science_medicine', r'\b(arzn\w*|medecin\w*|médecin\w*|kranck\w*|cure\w*|physick\w*|surgery|chirurg\w*|astrolog\w*|almanack\w*|kalender\w*|recipes?|husbandry|garden\w*|distill\w*)\b'),
    ('fiction_satire', r'\b(roman\w*|nouvelles?|contes?|histori\w* von|schwank\w*|satyr\w*|comique|plaisant\w*|novell\w*)\b'),
    ('treatise_essay', r'\b(traict\w*|trait\w*|discours|essais?|treatise|discourse|abhandlung|bericht\w*|unterricht\w*)\b'),
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sv(x) -> str:
    return '' if x is None or (isinstance(x, float) and x != x) else str(x)


def norm(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', fold(str(s)))


def iter_json(p: Path):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def bin50(y: int) -> int:
    return min(4, max(0, (int(y) - 1450) // 50))


BIN_LABEL = ('1450-1499', '1500-1549', '1550-1599', '1600-1649', '1650-1700')


# ------------------------------------------------------------------------------------------------ stage excl
def excl_sources():
    """name -> list of files whose texts and document names must stay out of the new passages."""
    silver = R2 / 'renner/silver'
    return {
        'gold_v2': sorted((R2 / 'gold2').glob('*.jsonl')) + [NER / 'gold_seed_v1.jsonl'],
        'dev3': [R2 / 'renner/dev3/passages.jsonl', R2 / 'renner/dev3/gold_dev3.jsonl'] + sorted((R2 / 'renner/dev3/annotation').glob('*.jsonl')),
        'test_v2': [R2 / 'test2/test_v2.jsonl'] + sorted((R2 / 'test2').glob('cand_*.jsonl')) + sorted((R2 / 'test2').glob('judged_*.jsonl'))
                   + [R2 / 'test2/test_v2_pool.jsonl.gz'],
        'silver': sorted(silver.glob('pool_*.jsonl')),
        'fewshot': [HERE / 'renner_fewshot_v1train.jsonl'],
    }


def cmd_excl(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    names = defaultdict(set)
    summ = {}
    allg = []
    for src, files in excl_sources().items():
        g = []
        seen = set()
        nf = 0
        for p in files:
            if not p.exists():
                continue
            nf += 1
            for d in iter_json(p):
                t = d.get('text') or d.get('passage_text')
                if isinstance(t, str) and t and t not in seen:
                    seen.add(t)
                    g.append(SP.gram_hashes(t))
                for k in ('doc', 'doc_id', 'source_path'):
                    v = d.get(k)
                    if v:
                        names[src].add(norm(SP.stem_of(str(v))))
        if src == 'fewshot':
            for d in json.load(open(HERE / 'ner3_teacher_fewshot.json')):
                if d.get('text'):
                    g.append(SP.gram_hashes(d['text']))
                    seen.add(d['text'])
        arr = np.unique(np.concatenate(g)) if g else np.empty(0, dtype=np.uint64)
        np.save(work / f'excl_grams_{src}.npy', arr)
        allg.append(arr)
        summ[src] = {'files': nf, 'texts': len(seen), 'distinct_8grams': int(len(arr)), 'doc_names': len(names[src])}
        log(src, summ[src])
    np.save(work / 'excl_grams_all.npy', np.unique(np.concatenate(allg)))
    # v1 documents (train/dev/test of the first round): file names of data_v1 doc_meta
    v1 = set()
    with gzip.open(SCR / 'meta/doc_meta.tsv.gz', 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.split('\t')[0]
            v1.add(norm(SP.stem_of(p)))
            v1.add(norm(Path(p).name))
    names['v1'] = v1
    summ['v1'] = {'doc_names': len(v1), 'source': 'data_v1 doc_meta.tsv.gz (copy of the Hypatia table)'}
    json.dump({k: sorted(x for x in v if x) for k, v in names.items()}, open(work / 'excl_names.json', 'w'))
    json.dump(summ, open(work / 'excl_summary.json', 'w'), indent=1)


# ------------------------------------------------------------------------------------------------ stage cands
def catalogue():
    db = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cat = {}
    src_count = Counter()
    org_count = Counter()
    rows = db.execute('select collection, filename, source, origin, year, date_confidence, edition_type, transmission_medium, eebo_id from texts where is_dup=0').fetchall()
    for r in rows:
        src_count[(r[0], r[2])] += 1
        org_count[(r[0], r[3])] += 1
    coll_n = Counter(r[0] for r in rows)
    for r in rows:
        key = f'{r[0]}/{SP.stem_of(r[1])}'
        generic_src = src_count[(r[0], r[2])] > max(3, 0.3 * coll_n[r[0]])
        generic_org = org_count[(r[0], r[3])] > max(3, 0.3 * coll_n[r[0]])
        cat.setdefault(key, {'source': '' if generic_src else (r[2] or ''), 'origin': '' if generic_org else (r[3] or ''), 'year': r[4], 'date_confidence': r[5],
                             'edition_type': r[6] or '', 'transmission_medium': r[7] or '', 'filename': r[1]})
    return cat


def attributions():
    out = {}
    with open(ATTR, encoding='utf-8') as f:
        hdr = f.readline().rstrip('\n').split('\t')
        for line in f:
            r = dict(zip(hdr, line.rstrip('\n').split('\t')))
            out[r['collection']] = r
    return out


def title_author(coll: str, stem: str, c: dict):
    src, org = c.get('source', ''), c.get('origin', '')
    title = re.sub(r'^(EEBO-TCP|Evans-TCP|TCP)\s*:\s*', '', src).strip() or org or stem.replace('_', ' ')
    author = ''
    if coll.startswith('EEBO') or coll in ('EvansTCP_American', 'LatinEEBO', 'FrenchEEBO'):
        m = re.search(r'\(([^()]*)\)\s*$', org)
        author = m.group(1).strip() if m else ''
    elif coll == 'TextGrid_German' and '_literatur_' in stem:
        author = stem.split('_literatur_')[0].replace('_', ' ')
        title = stem.split('_literatur_m_')[-1].replace('_', ' ')
    elif coll == 'ReF_ENHG':
        title = re.sub(r'^ReF:\s*', '', org or title)
    elif coll.startswith('Wikisource') and ' — ' in title:
        author = title.rsplit(' — ', 1)[1].strip()
    elif coll == 'DiBiLit_German':
        m = re.search(r'\(([a-zäöü]+), ~?\d{4}\)\s*$', title)
        author = m.group(1) if m else ''
    return title[:200], author


def source_url(coll: str, stem: str, attr: dict) -> str:
    if coll in ('EEBO_TCP_fill', 'LatinEEBO'):
        return f'https://quod.lib.umich.edu/e/eebo/{stem}.0001.001'
    if coll in ('EEBOphase2', 'FrenchEEBO'):
        return f'https://quod.lib.umich.edu/e/eebo2/{stem}.0001.001'
    if coll == 'EvansTCP_American':
        return f'https://quod.lib.umich.edu/e/evans/{stem}.0001.001'
    u = (attr.get(coll, {}).get('source_urls') or '').split(';')[0].strip()
    return u if u.startswith('http') else ''


def genre_of(coll: str, stem: str, title: str):
    t = re.sub(r'[_\-]+', ' ', f'{title} {stem}').lower()
    if coll not in ('NOSCEMUS', 'OSTA', 'EMLAP', 'GAMS_BaselRecords'):
        for g, pat in SP.GENRE_RULES + GENRE_RULES_EXTRA:
            if re.search(pat, t):
                return g, 'title_heuristic'
    return SP.GENRE_DEFAULT.get(coll) or GENRE_DEFAULT_EXTRA.get(coll, 'other_prose'), 'collection_default'


TG_GENRES = ('romane', 'roman', 'prosa', 'schwanksammlung', 'satirischer', 'satire', 'predigtliteratur', 'dialoge', 'volksbuch', 'theoretische', 'werk')
POETS_DE = re.compile(r'Gryphius|Luther|Greiffenberg|Hans Sachs|Speratus|Nicolai|Stegmann|Michael Franck|Rist|Opitz|Brant|Anton Ulrich|Hoffmannswaldau|Fleming|Logau|Spee|Neander')
VERSE_TITLE = re.compile(r'\b(Lied|Lieder|Gedicht|Ode|Sonett|Sonnet|Psalm|Grab-?schr|Hymn|Gesang|Gebet|Epigramm|Reim|Spruch|Volkslied|Tanz|Fabel)', re.I)


from ratlas.ner.ner2_passages import STOP as _STOP  # noqa: E402
_FUNC_EXTRA = {'es': 'y que de el la en a los se no con por las del su como mas pero si yo me le lo mi ni o cuando cómo qué'.split(),
               'it': 'e che di il la con ma non per se o né quando come gli vi ch'.split(),
               'fr': 'et que de le la les qui ne en a se mais si ou quand comme'.split(),
               'de': 'und der die das zu in nicht mit so wie wann da ich er sie es'.split(),
               'en': 'and the that with but to of in or nor when as'.split(), 'la': 'et in non ad cum quod qui ut sed si nec atque aut'.split()}
FUNC = {l: set(_STOP.get(l, [])) | set(v) for l, v in _FUNC_EXTRA.items()}
MODERN = re.compile(r'\b(1[89]\d\d|20\d\d)\b|wikisource|reprint|copyright|transcri(bed|ption)|digitali[sz]', re.I)


EN_ONLY = set('the that is was with his he which are have this from they their were been my your would should'.split())


def english_mixed(seg: str) -> bool:
    """Bilingual phrase books and dictionaries (English inside a French, German... passage)."""
    return sum(w.lower() in EN_ONLY for w in re.findall(r'[A-Za-z]+', seg)) >= 4


def verse_flat(seg: str, lang: str) -> bool:
    """Verse in a reading whose line breaks are lost: capitalised function words inside a sentence (line starts), or verse numbers every 5 lines."""
    toks = seg.split()
    n = 0
    for x, y in zip(toks, toks[1:]):
        if x[-1] in '.!?:;»”"':
            continue
        w = re.sub(r'^[¿¡"“«(\']+', '', y)
        if w[:1].isupper() and w.lower().strip('.,;:!?') in FUNC[lang]:
            n += 1
    lineno = len(re.findall(r'(?<=\s)(?:[1-9]\d*[05])\s+[A-ZÀ-Ý]', ' ' + seg))
    return n >= (9 if lang == 'en' else 5) or lineno >= 3


def gold2_reader_rules(coll: str, stem: str, title: str, author: str) -> str | None:
    """The document filters of the gold v2 readers that the layer reading does not apply (verse, drama, later translations)."""
    if coll == 'TextGrid_German':
        a0 = stem.split('_literatur_')[0]
        rest = stem.split('_literatur_m_')[-1]
        rest = rest[len(a0) + 1:] if rest.startswith(a0) else rest
        if rest.split('_')[0] not in TG_GENRES or a0.startswith('rabelais'):
            return 'textgrid_genre'
    if coll == 'WikisourceDE_Renaissance' and (POETS_DE.search(author or '') or POETS_DE.search(title or '') or VERSE_TITLE.search(title or '') or VERSE_TITLE.search(stem)):
        return 'verse_title'
    if coll == 'Hakluyt_English' and not stem.startswith('hakluyt_principal'):
        return 'not_principal_navigations'
    return None


def cmd_cands(a):
    import pandas as pd
    import pickle
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ex = Path(a.excl or a.work)
    names = {k: set(v) for k, v in json.load(open(ex / 'excl_names.json')).items()}
    parts = {s: np.load(ex / f'excl_grams_{s}.npy') for s in excl_sources()}
    langs = tuple(a.langs) if a.langs else LANGS
    silver_doc = {}                 # option "withdraw from silver": doc_id -> 8-gram hashes of its silver passages
    if a.allow_silver:
        names.pop('silver')
        allg = np.unique(np.concatenate([v for k, v in parts.items() if k != 'silver']))
        sil = defaultdict(list)
        sil_n = Counter()
        for p in sorted((R2 / 'renner/silver').glob('pool_*.jsonl')):
            for d in iter_json(p):
                sil[d['doc_id']].append(SP.gram_hashes(d['text']))
                sil_n[d['doc_id']] += 1
        silver_doc = {k: np.unique(np.concatenate(v)) for k, v in sil.items()}
        silver_all = parts['silver']
        log('silver documents', len(silver_doc))
    else:
        allg = np.load(ex / 'excl_grams_all.npy')
    long_names = [(n, src) for src, v in names.items() if src != 'v1' for n in v if len(n) >= 8]

    def n_shared(h, doc_id):
        """8-grams of h found in the excluded texts; in the silver option, silver 8-grams of the document itself do not count, those of other documents do."""
        hit = SP.in_sorted(h, allg)
        if a.allow_silver:
            hs = SP.in_sorted(h, silver_all)
            own = silver_doc.get(doc_id)
            if own is not None:
                hs &= ~SP.in_sorted(h, own)
            hit |= hs
        return int(hit.sum())
    lex = pickle.load(open(SCR / 'w/lex_la.pkl', 'rb'))
    cat = catalogue()
    attr = attributions()
    # inventory rows (one per document)
    frames = [pd.read_parquet(p) for p in INVENTORIES]
    rows = []
    for d_ in EXTRA_PARTS:
        for p in sorted(d_.glob('*.jsonl')):
            rows += [json.loads(l) for l in open(p, encoding='utf-8')]
    if rows:
        frames.append(pd.DataFrame(rows))
    inv = pd.concat(frames, ignore_index=True)
    inv = inv[inv['lang'].isin(LANGS)]
    inv = inv.sort_values('status', key=lambda s: s != 'ok').drop_duplicates(['doc_id', 'lang'], keep='first')
    log('inventory rows', len(inv), inv.groupby('lang').size().to_dict())
    # which shard holds each document
    where = {}
    for lang, dirs in TEXT_SOURCES.items():
        if lang not in langs:
            continue
        for d_ in dirs:
            for p in sorted(Path(d_).glob('*.jsonl.gz')):
                where.setdefault(lang, []).append(p)
    cache = ex / 'shard_ids.json'
    if cache.exists():
        present = {tuple(x) for x in json.load(open(cache))}
    else:
        present = set()
        for lang, shards in where.items():
            for p in shards:
                for line in gzip.open(p, 'rt', encoding='utf-8'):
                    m_ = re.match(r'\{"id": "((?:[^"\\]|\\.)*)"', line)
                    if m_:
                        present.add((lang, json.loads('"' + m_.group(1) + '"')))
        json.dump(sorted(present), open(cache, 'w'))
    reasons = defaultdict(Counter)
    elig = defaultdict(list)       # (lang, coll, bin) -> rows
    meta = {}
    for r in inv.itertuples():
        lang, coll = r.lang, r.collection
        if lang not in langs:
            continue
        if a.allow_silver and lang not in ('la', 'es', 'it'):
            continue
        if r.status != 'ok':
            reasons[lang][f'status:{r.status}'] += 1
            continue
        if coll in EXCLUDED_COLL or LL_MAP.search(coll):
            reasons[lang][f'collection_excluded:{coll}'] += 1
            continue
        if coll not in ALLOWED[lang]:
            reasons[lang][f'collection_not_used:{coll}'] += 1
            continue
        if (attr.get(coll, {}).get('tier') or 'T1') == 'T3':
            reasons[lang][f'collection_T3:{coll}'] += 1
            continue
        stem = r.doc_id.split('/', 1)[1]
        ns = norm(SP.stem_of(stem))
        hit = None
        for src, v in names.items():
            if ns in v:
                hit = src
                break
        if hit:
            reasons[lang][f'doc_name:{hit}'] += 1
            continue
        c = cat.get(r.doc_id, {})
        yr = r.year if (r.year == r.year and r.year) else c.get('year')
        if not yr or not (1450 <= int(yr) <= 1700):
            reasons[lang]['year_out'] += 1
            continue
        title, author = title_author(coll, stem, c)
        why = gold2_reader_rules(coll, stem, title, author)
        if why:
            reasons[lang][why] += 1
            continue
        if (r.n_words if r.n_words == r.n_words else 0) < 250:
            reasons[lang]['short_doc'] += 1
            continue
        if (lang, r.doc_id) not in present:
            reasons[lang]['text_not_in_copied_shards'] += 1
            continue
        g, gs = genre_of(coll, stem, title)
        m = {'doc_id': r.doc_id, 'lang': lang, 'collection': coll, 'stem': stem, 'year': int(yr), 'date_confidence': sv(r.date_confidence) or c.get('date_confidence') or '',
             'title': title, 'author': author, 'genre': g, 'genre_source': gs, 'tier': sv(r.tier), 'licence_tag': sv(r.licence_tag), 'n_words': int(r.n_words or 0), 'ns': ns,
             'edition_type': c.get('edition_type', ''), 'transmission_medium': c.get('transmission_medium', ''), 'source_url': source_url(coll, stem, attr),
             'silver_doc': r.doc_id in silver_doc, 'silver_passages_in_doc': int(sil_n.get(r.doc_id, 0)) if a.allow_silver else 0}
        meta[(lang, r.doc_id)] = m
        elig[(lang, coll, bin50(yr))].append(m)
    # documents to read: all of a small stratum, a seeded sample of a large one
    want = {}
    for k in sorted(elig):
        lst = sorted(elig[k], key=lambda m: m['doc_id'])
        random.Random(f'{SEED}/docs/{k[0]}/{k[1]}/{k[2]}').shuffle(lst)      # one seed per stratum: a change in one stratum leaves the others as they are
        k_ = 0
        for m in lst:
            if k_ >= DOCS_PER_STRATUM:
                break
            ns = m.pop('ns')
            hit = None
            if len(ns) >= 8:                     # name inclusion (other volumes or parts of an excluded work), checked only on the documents read
                for n, src in long_names:
                    if n in ns or ns in n:
                        hit = src
                        break
            if hit:
                reasons[m['lang']][f'doc_name_inclusion:{hit}'] += 1
                continue
            want[(m['lang'], m['doc_id'])] = m
            k_ += 1
    log('eligible documents', len(meta), 'to read', len(want), Counter(l for l, _ in want))
    lex_fr = Counter()
    if 'fr' in where:                      # lexicon of keyed / ground-truth French (gold v2: ABU, Wikisource, Gallicorpora), for the Mazarinades OCR filter
        from ratlas.ner.ner2_passages import ALPHA
        for p in where['fr']:
            for d in iter_json(p):
                if d['id'].split('/')[0] in ('ABU_French', 'WikisourceFR_Renaissance', 'Gallicorpora_French_HTR', 'FrenchEEBO', 'EEBOphase2'):
                    lex_fr.update(fold(w) for w in re.findall(rf'[{ALPHA}]+', d['t'][:500000]))
        lex_fr = Counter({k: v + 1 for k, v in lex_fr.items()})     # SP.oov_rate counts lex[w] < 2 as unknown; +1 gives the gold v2 rule (< 3)
        log('French lexicon', len(lex_fr))
    out = open(work / 'cands.jsonl', 'w', encoding='utf-8')
    docs_log = open(work / 'docs_read.jsonl', 'w', encoding='utf-8')
    st = defaultdict(Counter)
    for lang, shards in where.items():
        for p in shards:
            for d in iter_json(p):
                m = want.get((lang, d.get('id')))
                if m is None:
                    continue
                want.pop((lang, d['id']))
                text = d['t']
                coll = m['collection']
                if coll == 'OSTA':
                    text = SP.osta_norm(text)
                text = SP.EOL_EQ.sub('', text)
                if coll in SP.TRIM_COLLECTIONS:
                    n = len(text)
                    text = text[int(n * 0.04):int(n * 0.92)]
                h = SP.gram_hashes(text)
                nm = n_shared(h, m['doc_id'])
                frac = nm / max(1, len(h))
                rec = {'doc_id': m['doc_id'], 'lang': lang, 'collection': coll, 'year': m['year'], 'matched_8grams': nm, 'frac': round(frac, 5)}
                if nm >= 25 or frac >= 0.01:
                    rec['excluded'] = 'doc_8gram'
                    rec['by_source'] = {s: int(SP.in_sorted(h, e).sum()) for s, e in parts.items()}
                    rec['silver_doc'] = m['silver_doc']
                    docs_log.write(json.dumps(rec) + '\n')
                    st[lang]['doc_8gram_excluded'] += 1
                    continue
                if lang == 'la':
                    lim = SP.OCR_COLLECTIONS.get(coll, 0.14)
                    chk0 = (lambda seg, _l=lim: not SP.STRAY.search(seg) and SP.oov_rate(seg, lex) <= _l and not SP.OCR_JUNK.search(seg) and seg.count('¶') <= 1)
                elif lang == 'en':
                    chk0 = (lambda seg: not SP.STRAY.search(seg) and not SP.OCR_JUNK.search(seg))
                else:
                    chk0 = (lambda seg: not SP.STRAY.search(seg) and not SP.OCR_JUNK.search(seg) and not SP.is_drama(seg))
                chk1 = (lambda seg, _c=chk0, _l=lang: _c(seg) and not MODERN.search(seg) and not verse_flat(seg, _l) and (_l == 'en' or not english_mixed(seg)))
                if coll == 'Mazarinades':          # OCR: the gold v2 rule (share of words unknown to a lexicon of clean French at most 11 %)
                    chk = (lambda seg, _c=chk1: _c(seg) and SP.oov_rate(seg, lex_fr) <= 0.11)
                else:
                    chk = chk1
                drng = random.Random(f'{SEED}/{m["doc_id"]}')
                got = []
                for attempt in range(6):
                    if len(got) >= 2:
                        break
                    r_ = draw_passage(text, drng, lang, extra_check=chk)
                    if not r_:
                        break
                    off, seg = r_
                    if any(abs(off - o) < 6000 for o, _, _ in got):
                        continue
                    gp = SP.gram_hashes(seg)
                    shared = n_shared(gp, m['doc_id'])
                    if shared >= 2:
                        st[lang]['passage_8gram_excluded'] += 1
                        continue
                    got.append((off, seg, shared))
                rec['n_passages'] = len(got)
                docs_log.write(json.dumps(rec) + '\n')
                st[lang]['docs_read'] += 1
                st[lang]['docs_with_passage'] += bool(got)
                for rank, (off, seg, shared) in enumerate(sorted(got)):
                    out.write(json.dumps({**m, 'offset_in_doc': off, 'text': seg, 'n_words': len(seg.split()), 'local_shared_8grams': shared,
                                          'draw_rank': rank, 'bin': bin50(m['year'])}, ensure_ascii=False) + '\n')
                    st[lang]['passages'] += 1
        log(lang, dict(st[lang]))
    out.close()
    docs_log.close()
    st_all = {'reasons_doc_level': {k: dict(v) for k, v in reasons.items()}, 'draw': {k: dict(v) for k, v in st.items()},
              'eligible_docs': Counter(l for l, _ in meta), 'eligible_by_coll': {f'{l}/{c}': n for (l, c), n in Counter((m['lang'], m['collection']) for m in meta.values()).items()},
              'not_found_in_shards': len(want), 'not_found_examples': [k[1] for k in list(want)[:20]]}
    json.dump(st_all, open(work / 'cands_summary.json', 'w'), indent=1, ensure_ascii=False, default=dict)
    log('not found in shards', len(want))


# ------------------------------------------------------------------------------------------------ stage select
def water_fill(avail: list[int], total: int) -> list[float]:
    k = len(avail)
    tgt = [0.0] * k
    left = float(total)
    open_ = [i for i in range(k) if avail[i] > 0]
    while left > 1e-9 and open_:
        share = left / len(open_)
        nxt = []
        for i in open_:
            room = avail[i] - tgt[i]
            add = min(share, room)
            tgt[i] += add
            left -= add
            if avail[i] - tgt[i] > 1e-9:
                nxt.append(i)
        if nxt == open_ and share <= 1e-9:
            break
        open_ = nxt
    return tgt


def author_key(a: str) -> str:
    """First word of the author name, folded (Wickram in TextGrid = wickram in DiBiLit); '' for anonymous."""
    if not a or a.startswith('anon'):
        return ''
    m = re.search(r'[^\W\d_]+', fold(a))
    return m.group(0) if m else ''


def select_lang(cands: list[dict], T: int, rng: random.Random, taken_fp=None, base_counts=None, coll_cap=None):
    """Greedy stratified pick. Returns (selected, info)."""
    by_doc = defaultdict(list)
    for c in cands:
        by_doc[c['doc_id']].append(c)
    n_coll = Counter(c['collection'] for c in cands)
    if coll_cap is None:
        share = COLL_SHARE0
        while sum(min(math.ceil(share * T), n) for n in n_coll.values()) < T and share < 1.0:
            share = round(share + 0.01, 2)
        coll_cap = math.ceil(share * T)
    cap_g = math.ceil(GENRE_SHARE * T)
    cap_a = max(2, math.ceil(AUTHOR_SHARE * T))
    avail_bin = [0] * 5
    for c in cands:
        if c['draw_rank'] == 0:
            avail_bin[c['bin']] += 1
    tgt_bin = water_fill([min(x, T) for x in avail_bin], min(T, sum(avail_bin)))
    sel = []
    cnt = base_counts or {'coll': Counter(), 'bin': Counter(), 'genre': Counter(), 'author': Counter(), 'doc': Counter()}
    fps = list(taken_fp or [])
    order = list(cands)
    rng.shuffle(order)
    pools = defaultdict(list)
    for c in order:
        pools[(c['bin'], c['collection'])].append(c)
    used = set()

    def ok(c, level, relax):
        if id(c) in used or cnt['doc'][c['doc_id']] >= level or c['draw_rank'] >= level:
            return False
        if cnt['coll'][c['collection']] >= coll_cap:
            return False
        if relax < 2 and cnt['genre'][c['genre']] >= cap_g:
            return False
        au = author_key(c['author'])
        if relax < 2 and au and cnt['author'][au] >= cap_a:
            return False
        fp = fingerprints(c['text'])
        if any(len(fp & o) >= 0.05 * max(1, min(len(fp), len(o))) for o in fps):
            used.add(id(c))
            return False
        c['_fp'] = fp
        return True

    def take(c):
        used.add(id(c))
        sel.append(c)
        fps.append(c.pop('_fp'))
        cnt['coll'][c['collection']] += 1
        cnt['bin'][c['bin']] += 1
        cnt['genre'][c['genre']] += 1
        cnt['doc'][c['doc_id']] += 1
        if author_key(c['author']):
            cnt['author'][author_key(c['author'])] += 1

    for level, relax in ((1, 0), (2, 0), (2, 1), (2, 2)):
        while len(sel) < T:
            bins = sorted(range(5), key=lambda b: (cnt['bin'][b] / tgt_bin[b]) if tgt_bin[b] > 0 else 9e9)
            if relax == 0:
                bins = [b for b in bins if tgt_bin[b] > 0 and cnt['bin'][b] < math.ceil(tgt_bin[b])]
            done = False
            for b in bins:
                colls = sorted({k[1] for k in pools if k[0] == b}, key=lambda x: (cnt['coll'][x] / coll_cap, x))
                for col in colls:
                    for c in pools[(b, col)]:
                        if ok(c, level, relax):
                            take(c)
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if not done:
                break
    info = {'target': T, 'selected': len(sel), 'collection_cap': coll_cap, 'collection_cap_share': round(coll_cap / max(1, T), 3), 'genre_cap': cap_g, 'author_cap': cap_a,
            'bin_targets': {BIN_LABEL[i]: round(t, 1) for i, t in enumerate(tgt_bin)}, 'candidate_passages': len(cands), 'candidate_docs': len(by_doc),
            'candidates_by_collection': dict(n_coll.most_common())}
    return sel, info, cnt, fps


def cmd_select(a):
    work = Path(a.work)
    cands = defaultdict(list)
    drop = set()
    rej = {}
    if a.reject_docs:                     # documents set aside after reading their titles and a sample (verse, drama, modern texts, unreliable dates)
        for line in open(a.reject_docs, encoding='utf-8'):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 4 and f[0] != 'lang':
                rej[f[1]] = f[3]
    if a.drop_docs:                       # documents flagged by the whole-document check against v1 and the mixes (ner3_gold3_doc_leak.py, Hypatia)
        dl = json.load(open(a.drop_docs))
        drop = {r['doc_id'] for r in dl['flagged']}
        checked = {r['doc_id'] for r in dl['per_doc']}
    nd = Counter()
    for d in iter_json(work / 'cands.jsonl'):
        if d['doc_id'] in drop:
            nd[d['lang']] += 1
            continue
        if a.drop_docs and d['doc_id'] not in checked:    # only documents that went through the whole-document check
            nd[d['lang'] + ':not_checked'] += 1
            continue
        if d['doc_id'] in rej:
            nd[d['lang'] + ':read'] += 1
            continue
        cands[d['lang']].append(d)
    log('candidates dropped by the document check', dict(nd))
    final_sel, reserve, report = [], [], {}
    for lang in LANGS:
        T = TARGET[lang]
        rng = random.Random(f'{SEED}/select/{lang}')
        sel, info, cnt, fps = select_lang(cands[lang], T, rng)
        sel_ids = {id(c) for c in sel}
        rest = [c for c in cands[lang] if id(c) not in sel_ids]
        R = math.ceil(RESERVE_SHARE * T)
        res, _, _, _ = select_lang(rest, R, random.Random(f'{SEED}/reserve/{lang}'), taken_fp=fps, base_counts=cnt,
                                   coll_cap=max(info['collection_cap'], math.ceil(info['collection_cap_share'] * (T + R))))
        for i, c in enumerate(sel):
            c['select_rank'] = i
        for i, c in enumerate(res):
            c['select_rank'] = len(sel) + i
            c['reserve'] = True
        final_sel += sel
        reserve += res
        info['reserve'] = len(res)
        info['shortfall'] = max(0, T - len(sel))
        report[lang] = info
        log(lang, 'selected', len(sel), '/', T, 'reserve', len(res), 'coll cap', info['collection_cap'], info['collection_cap_share'])
    with open(work / 'select_new_plus_reserve.jsonl', 'w', encoding='utf-8') as f:
        for c in final_sel + reserve:
            pid = f"gold3/{c['lang']}/{c['doc_id']}#{c['draw_rank'] + 1}"
            c['id'] = pid
            c['doc'] = c['stem']
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    json.dump(report, open(work / 'select_report.json', 'w'), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ stage final
def verify_local(recs, allow_silver=False):
    """Independent check with string 8-grams (ner2_passages.fingerprints), as ner3_silver_verify.py: number and share of each passage's 8-grams found in
    every excluded source (texts streamed; only the 8-grams of the checked passages are kept in memory), plus a document-name check."""
    fps = {r['id']: fingerprints(r['text']) for r in recs}
    doc_of = {r['id']: r['doc_id'] for r in recs}
    index = defaultdict(list)
    for pid, fp in fps.items():
        for g in fp:
            index[g].append(pid)
    found = defaultdict(lambda: defaultdict(set))
    dnames = defaultdict(set)
    for src, files in excl_sources().items():
        seen = set()
        for p in files:
            if not p.exists():
                continue
            for d in iter_json(p):
                t = d.get('text')
                if isinstance(t, str) and t not in seen:
                    seen.add(t)
                    for g in fingerprints(t):
                        for pid in index.get(g, ()):
                            if allow_silver and src == 'silver' and d.get('doc_id') == doc_of[pid]:
                                continue            # option: the passage's own silver document (to be withdrawn) does not count
                            found[pid][src].add(g)
                for k in ('doc', 'doc_id'):
                    if d.get(k):
                        n = norm(SP.stem_of(str(d[k])))
                        if len(n) >= 6:
                            dnames[src].add(n)
        log('verify: streamed', src, len(seen), 'texts')
    out = {}
    for r in recs:
        fp = fps[r['id']]
        sh = {s: len(v) for s, v in found[r['id']].items() if v}
        out[r['id']] = {'shared': sh, 'max_share': round(max(sh.values(), default=0) / max(1, len(fp)), 4),
                        'name_hit': [s for s, v in dnames.items() if norm(r['doc']) in v and not (allow_silver and s == 'silver')]}
    return out


def cmd_final(a):
    work = Path(a.work)
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    recs = list(iter_json(work / 'select_new_plus_reserve.jsonl'))
    remote = json.load(open(a.leak)) if a.leak else None
    flagged_remote = {}
    per_remote = {}
    if remote:
        for o in remote['per_passage']:
            per_remote[o['id']] = o
        for o in remote['flagged']:
            flagged_remote[o['id']] = o
    loc = verify_local(recs, a.allow_silver)
    keep, dropped = [], []
    by_lang = defaultdict(list)
    for r in recs:
        by_lang[r['lang']].append(r)
    for lang in LANGS:
        lst = sorted(by_lang[lang], key=lambda r: r['select_rank'])
        T = TARGET[lang]
        main = [r for r in lst if not r.get('reserve')]
        res = [r for r in lst if r.get('reserve')]
        # refused: flagged by ner2_leak_check (share >= 0.02 or v1 document name), >= 2 8-grams shared with v1 / the mixes (the local rule),
        # >= 1 % shared with an excluded set in the independent local check, or a document name of an excluded set
        bad = lambda r: (r['id'] in flagged_remote or (per_remote.get(r['id'], {}).get('n_found') or 0) >= 2 or loc[r['id']]['max_share'] >= 0.01
                         or loc[r['id']]['name_hit'])
        good_main = [r for r in main if not bad(r)]
        for r in main + res:
            if bad(r):
                dropped.append({'id': r['id'], 'lang': lang, 'reserve': bool(r.get('reserve')), 'remote': flagged_remote.get(r['id']), 'local': loc[r['id']]})
        need = len(main) - len(good_main)
        repl = [r for r in res if not bad(r)][:need]
        for r in repl:
            r['replacement_from_reserve'] = True
        keep += good_main + repl
    new_out = []
    for r in keep:
        toks, starts = tokenize(r['text'])
        rp = per_remote.get(r['id'], {})
        leak = {'local_8gram': {'method': 'string 8-grams (ner2_passages.fingerprints)', **loc[r['id']]},
                'local_hash_shared_8grams': r['local_shared_8grams'],
                'v1_and_mixes': ({'share': rp.get('share'), 'n_found': rp.get('n_found'), 'files': rp.get('files'), 'doc_name_in_v1': rp.get('doc_name_in_v1')} if rp else 'not run')}
        new_out.append({'id': r['id'], 'lang': r['lang'], 'collection': r['collection'], 'doc': r['stem'], 'doc_id': r['doc_id'], 'title': r['title'], 'author': r['author'],
                        'date': str(r['year']), 'year': r['year'], 'date_precision': SP.precision_of(r['date_confidence']), 'date_basis': r['date_confidence'],
                        'period_bin_50y': BIN_LABEL[r['bin']], 'genre': r['genre'], 'genre_source': r['genre_source'], 'source_url': r['source_url'],
                        'licence_note': f"{r.get('licence_tag') or 'n/a'} ({r['collection']}, text archive {r.get('tier') or 'n/a'})",
                        'edition_type': r['edition_type'], 'transmission_medium': r['transmission_medium'], 'seed': False, 'offset_in_doc': r['offset_in_doc'],
                        'n_words': r['n_words'], 'text': r['text'], 'tokens': toks, 'token_starts': starts, 'dropcap_repairs': [], 'from_gold2': False,
                        'replacement_from_reserve': bool(r.get('replacement_from_reserve')), 'leak_check': leak})
    if a.allow_silver:
        # option: new la/es/it passages taken from silver documents; nothing here is part of passages.jsonl
        od = outd / 'option_retrait_silver'
        od.mkdir(parents=True, exist_ok=True)
        silver_n = defaultdict(Counter)
        for p in sorted((R2 / 'renner/silver').glob('pool_*.jsonl')):
            kind = 'reserve' if 'reserve' in p.name else 'pool'
            for d in iter_json(p):
                silver_n[d['doc_id']][kind] += 1
        langs_opt = ('la', 'es', 'it')
        new_out = [r for r in new_out if r['lang'] in langs_opt]
        for r in new_out:
            r['silver_doc'] = r['doc_id'] in silver_n
            r['option'] = 'retrait_silver'
        with open(od / 'passages_option.jsonl', 'w', encoding='utf-8') as f:
            for r in new_out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        docs = sorted({(r['lang'], r['doc_id'], r['collection']) for r in new_out if r['silver_doc']})
        with open(od / 'silver_docs_a_retirer.tsv', 'w', encoding='utf-8') as f:
            f.write('lang\tdoc_id\tcollection\tsilver_pool_passages\tsilver_reserve_passages\tgold3_passages\n')
            for l, did, c in docs:
                f.write(f"{l}\t{did}\t{c}\t{silver_n[did]['pool']}\t{silver_n[did]['reserve']}\t{sum(r['doc_id'] == did for r in new_out)}\n")
        st = {}
        for l in langs_opt:
            L = [r for r in new_out if r['lang'] == l]
            dl = {r['doc_id'] for r in L if r['silver_doc']}
            st[l] = {'new': len(L), 'target': TARGET[l], 'docs': len({r['doc_id'] for r in L}), 'silver_docs_to_withdraw': len(dl),
                     'silver_pool_passages_lost': sum(silver_n[d]['pool'] for d in dl), 'silver_reserve_passages_lost': sum(silver_n[d]['reserve'] for d in dl),
                     'by_collection': dict(Counter(r['collection'] for r in L).most_common()), 'by_period': dict(sorted(Counter(r['period_bin_50y'] for r in L).items())),
                     'by_genre': dict(Counter(r['genre'] for r in L).most_common()), 'max_per_doc': max(Counter(r['doc_id'] for r in L).values()) if L else 0,
                     'tokens': sum(len(r['tokens']) for r in L), 'replacements': sum(r['replacement_from_reserve'] for r in L),
                     'v1_max_share': max((r['leak_check']['v1_and_mixes']['share'] or 0 for r in L if isinstance(r['leak_check']['v1_and_mixes'], dict)), default=None),
                     'local_max_share': max((r['leak_check']['local_8gram']['max_share'] for r in L), default=0)}
        json.dump({'by_lang': st, 'dropped_after_leak': [d for d in dropped if d['lang'] in langs_opt]}, open(od / 'option_stats.json', 'w'), indent=1, ensure_ascii=False)
        log('option', json.dumps({l: v['new'] for l, v in st.items()}))
        return
    # gold v2 passages, with the owner-reviewed places
    g2 = []
    for r in iter_json(R2 / 'gold2/final_gold_v2.jsonl'):
        rec = {k: v for k, v in r.items() if k not in ('spans', 'reviewed', 'review_note', 'span_convention')}
        rec['from_gold2'] = True
        rec['gold2_loc'] = r['spans']
        rec['gold2_loc_convention'] = r.get('span_convention')
        rec['gold2_review_note'] = r.get('review_note', '')
        rec['period_bin_50y'] = BIN_LABEL[bin50(r['year'])] if r.get('year') else None
        rec['n_words'] = len(r['text'].split())
        g2.append(rec)
    allp = g2 + new_out
    # double annotation: 200 passages, stratified by language (largest remainder), then by from_gold2 inside a language; seed 20261024
    rng = random.Random(SEED)
    n_by = Counter(r['lang'] for r in allp)
    N = sum(n_by.values())
    quota = {l: 200 * n / N for l, n in n_by.items()}
    base = {l: int(q) for l, q in quota.items()}
    for l in sorted(quota, key=lambda l: -(quota[l] - base[l]))[:200 - sum(base.values())]:
        base[l] += 1
    dbl = []
    for l in LANGS:
        grp = defaultdict(list)
        for r in sorted((r for r in allp if r['lang'] == l), key=lambda r: r['id']):
            grp[r['from_gold2']].append(r['id'])
        k = base.get(l, 0)
        n_l = sum(len(v) for v in grp.values())
        q2 = {g: k * len(v) / n_l for g, v in grp.items()}
        b2 = {g: int(q) for g, q in q2.items()}
        for g in sorted(q2, key=lambda g: -(q2[g] - b2[g]))[:k - sum(b2.values())]:
            b2[g] += 1
        for g, v in sorted(grp.items()):
            dbl += rng.sample(v, b2[g])
    dset = set(dbl)
    for r in allp:
        r['double_annotation'] = r['id'] in dset
    order = {l: i for i, l in enumerate(LANGS)}
    allp.sort(key=lambda r: (order[r['lang']], not r['from_gold2'], r['id']))
    ids = [r['id'] for r in allp]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    for r in allp:
        assert all(r['text'][s:s + len(t)] == t for s, t in zip(r['token_starts'], r['tokens'])), r['id']
    with open(outd / 'passages.jsonl', 'w', encoding='utf-8') as f:
        for r in allp:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    (outd / 'double_annotation_ids.txt').write_text('\n'.join(sorted(dbl, key=lambda i: (order[i.split('/')[1]], i))) + '\n')
    json.dump(dropped, open(work / 'dropped_after_leak.json', 'w'), indent=1, ensure_ascii=False)
    # statistics for the README
    stats = {'n_total': len(allp), 'by_lang': {}, 'double': Counter(i.split('/')[1] for i in dbl), 'dropped_after_leak': len(dropped),
             'remote_leak': ({k: remote[k] for k in ('n_passages', 'n_with_any_hit', 'max_share', 'streamed_files')} if remote else None)}
    for l in LANGS:
        L = [r for r in allp if r['lang'] == l]
        new = [r for r in L if not r['from_gold2']]
        stats['by_lang'][l] = {'total': len(L), 'from_gold2': len(L) - len(new), 'new': len(new), 'target_new': TARGET[l], 'new_docs': len({r['doc_id'] for r in new}),
                               'new_max_per_doc': max(Counter(r['doc_id'] for r in new).values()) if new else 0,
                               'new_by_collection': dict(Counter(r['collection'] for r in new).most_common()),
                               'new_by_period': dict(sorted(Counter(r['period_bin_50y'] for r in new).items())),
                               'all_by_period': dict(sorted(Counter(r['period_bin_50y'] for r in L).items())),
                               'new_by_genre': dict(Counter(r['genre'] for r in new).most_common()),
                               'new_tokens': sum(len(r['tokens']) for r in new), 'all_tokens': sum(len(r['tokens']) for r in L),
                               'new_words_median': int(np.median([r['n_words'] for r in new])) if new else 0,
                               'new_replacements': sum(r['replacement_from_reserve'] for r in new),
                               'new_local_max_share': max((r['leak_check']['local_8gram']['max_share'] for r in new), default=0),
                               'new_v1_max_share': max((r['leak_check']['v1_and_mixes']['share'] or 0 for r in new if isinstance(r['leak_check']['v1_and_mixes'], dict)), default=None),
                               'new_with_any_v1_8gram': sum(1 for r in new if isinstance(r['leak_check']['v1_and_mixes'], dict) and (r['leak_check']['v1_and_mixes']['n_found'] or 0) > 0),
                               'new_with_any_local_8gram': sum(1 for r in new if r['leak_check']['local_8gram']['shared'])}
    json.dump(stats, open(outd / 'gold3_stats.json', 'w'), indent=1, ensure_ascii=False, default=dict)
    log(json.dumps({l: (v['total'], v['new']) for l, v in stats['by_lang'].items()}), 'total', len(allp), 'dropped', len(dropped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('excl', 'cands', 'select', 'final'))
    ap.add_argument('--work', required=True)
    ap.add_argument('--leak', default='')
    ap.add_argument('--out', default=str(R2 / 'gold3'))
    ap.add_argument('--excl', default='', help='directory of the excl stage (default: --work)')
    ap.add_argument('--langs', nargs='*', default=[])
    ap.add_argument('--reject-docs', default='', help='tsv lang, doc_id, title, reason: documents set aside after reading')
    ap.add_argument('--drop-docs', default='', help='doc_leak.json of ner3_gold3_doc_leak.py: flagged documents are left out')
    ap.add_argument('--allow-silver', action='store_true', help='option: documents of the silver pool allowed (to be withdrawn from silver if the owner agrees); la, es, it only')
    a = ap.parse_args()
    {'excl': cmd_excl, 'cands': cmd_cands, 'select': cmd_select, 'final': cmd_final}[a.cmd](a)


if __name__ == '__main__':
    main()
