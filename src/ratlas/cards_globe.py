"""Data shards of the private globe page, built on Atlas v2.1 (run on Hypatia through Slurm, no network).

    python -I -m ratlas.cards_globe build --final DIR --mentions mentions.parquet --documents documents.parquet \
        --cards DIR --entries DIR --tgn DIR --gn-index gn_index.parquet --wd-entities wd_entities.parquet \
        --wd-class-labels wd_class_labels.tsv --images wd_images.tsv --img-dir DIR --img-meta meta_w320.json \
        --issues wd_places_issues.tsv --out DIR

Output (``--out``): ``core.json``, ``wd.json``, ``docs/NN.json`` (NN = 00..63), ``img/NN.json`` (NN = 00..31), ``build_report.json``.

Points. One point of the globe is one entry of ``globe_places.json`` (a ``globe_id``: places of the atlas that share a normalised name
within 20 km are merged), in the order of that file (in-period mentions, most mentioned first). The row index of a point is its position
in ``core.json["rows"]``; its documents are in ``docs/<index mod 64>.json`` under the key ``str(index)``.

Documents. A (point, document) pair is listed when the point has at least one canonical, non-excluded, in-period place mention in that
document (the atlas counts: nothing excluded is listed). The counts of a point add up to the globe's ``n``.

Names. ``name_en``: Wikidata English label; else the curated English exonym of the exonym table; else the English term of the point's TGN
subject; else the GeoNames name of its feature; else the Casebooks directory entry; else the most frequent edition form. ``original_name``:
the most frequent form in the editions (surface of the tagged mention).

Kinds (v2.2, five-way). ``c`` city, town, village, city quarter; ``a`` country, region, territory; ``n`` river, lake, sea, island, mountain, valley, pass, cape,
forest; ``b`` building, castle, monastery, street, square, gate, bridge, site; ``u`` other. The atlas computes it for every point of ``globe_places.json`` (field
``k``, ``placekind.kind5``: Wikidata classes of the accepted item, GeoNames feature code, TGN place type, register type); a point the atlas leaves ``u`` is read
again here with the English class labels of the Wikidata cards. A three-way value (city / area / other) of an older ``globe_places.json`` is converted.

Letters (v2.2). ``letters.json`` = ``[[from_row, to_row, count, [10 quarter counts], [[slug_idx, count], ...]], ...]``: one edge per ordered pair of points
(rows of ``core.json``) with the number of letters sent from the first to the second (documents with a canonical, non-excluded, in-period place mention of
role ``sent_from`` and one of role ``sent_to``, each resolved to ONE point of the globe; a letter whose end names two different points, or has none on the
globe, is not an edge), by quarter-century of the document year and by edition (``slugs`` of ``core.json``). ``build_report.json`` has the coverage per
edition: ``letters``, ``sender_place``, ``addressee_place`` (any resolved place, on the globe or not), ``both_on_globe`` and ``edges_letters``.

Images. Only thumbnails of places that are on the globe; the 'weak' images listed by the earlier check (maps, flags, arms, collages) are
dropped; an image without artist gets the Commons Credit line, else ``unknown author``.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import generic_terms as GEN
from . import placekind as PK

QUARTERS = [1450 + 25 * i for i in range(10)]
DOC_SHARDS = 64
IMG_SHARDS = 32
CONTEXT_MAX = 120
LETTER_ENDS = ('sent_from', 'sent_to')
UNNAMED = 'Unnamed place'
TIER_CONTEXT = ('T1', 'T2')           # T3 mentions carry no context in the page's data

#: the source codes whose point is inferred from a NAME (the page says 'located by name'): propagation, GeoNames / TGN name, exonym and polity table, Latin name (v2.3)
NAME_BASED_CODES = ('p', 'n', 'e', 'l')

#: legend of the coordinate source codes (as in globe_places.json)
SOURCE_CODES = {'r': 'register', 'w': 'Wikidata', 'g': 'GeoNames id', 't': 'Getty TGN id', 'c': 'Casebooks directory',
                'p': 'name propagation', 'n': 'GeoNames / TGN name match', 'e': 'exonym table / historical polities', 'l': 'Latin name (located by name)'}

# Wikidata items that the v1 check found wrong or doubtful and that v2.1 still maps: (display name or None, keep the card?)
DOUBTFUL_ITEMS = {
    'Q705438': ('Anhalt', False),         # the Free State of Anhalt (1918-1945), the editions mean the principality
    'Q202311': ('Egypt', False),          # Roman Egypt, the editions mean Egypt / Ottoman Egypt
    'Q2960786': (None, False),            # Charleville, the editions say Masieres
}
# Wikidata duplicate items that v2.1 still carries -> the item they duplicate (found by the v1 check: same GeoNames id / name within 5 km)
DUPLICATE_ITEMS = {'Q32192592': 'Q7033', 'Q32749642': 'Q4065', 'Q31971850': 'Q2024532', 'Q32067884': 'Q3931', 'Q31939750': 'Q3214',
                   'Q31976906': 'Q2100'}          # v2.3: 'lsjbot Duisburg', a Wikipedia-bot item, is Duisburg
#: a Wikidata label that is a bot's working title, not a name ('lsjbot Duisburg'): never a display name
BOT_LABEL = re.compile(r'\b(lsjbot|bot[- ]generated|wikibot)\b', re.I)

GENERIC_TYPES = {
    'human settlement', 'big city', 'geographic location', 'geographical feature', 'geographic entity', 'geographical object',
    'tourist destination', 'largest city', 'major regional center', 'single entity of population', 'territorial entity',
    'administrative territorial entity', 'location', 'place', 'populated place', 'geographic region', 'physical object',
    'wikimedia disambiguation page', 'wikimedia list article', 'object', 'entity', 'toponym', 'city or town', 'region of europe',
}
GENERIC_OWN_WORK = re.compile(r'^(own work|self[- ]?made|own photograph|own photo|selbst fotografiert|eigenes werk|travail personnel|opera propria)\b', re.I)


# ======================================================================================================================
# small pure functions (unit tests)
# ======================================================================================================================
def doc_shard(row_index: int) -> int:
    return row_index % DOC_SHARDS


def img_shard(qid: str) -> int:
    """sha1(qid) mod 32, the digest read as one big-endian integer."""
    return int.from_bytes(hashlib.sha1(qid.encode('utf-8')).digest(), 'big') % IMG_SHARDS


def kind_letter(base_kind, wd_labels=(), gn_fcodes=(), tgn_types=(), notes=(), exonym_kind=None) -> str:
    """'c' / 'a' / 'n' / 'b' / 'u' (``placekind.kind5``). ``base_kind``: the atlas's letter, or the three-way word (city / area / other) of an older file.
    A base that is already a decided kind (c a n b) is kept unless the labels say otherwise with more votes; a base 'u' or 'other' is read from the sources."""
    return PK.kind5(base_kind, wd_labels, gn_fcodes, tgn_types, notes, exonym_kind)


def meaningful_types(labels, limit: int = 3) -> list:
    """The first ``limit`` P31 labels that say something (no raw Q ids, no generic super-classes), without duplicates."""
    out, seen = [], set()
    for l in labels or []:
        if not isinstance(l, str):
            continue
        s = l.strip()
        if not s or re.fullmatch(r'Q\d+', s) or s.lower() in GENERIC_TYPES or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= limit:
            break
    return out


def first_label(labels):
    for l in labels or []:
        if isinstance(l, str) and l.strip() and not re.fullmatch(r'Q\d+', l.strip()):
            return l.strip()
    return ''


def inception_value(text, year, precision):
    """A year (int) when the date is precise to the year or the day, else the text as Wikidata renders it (century ...), else None."""
    try:
        if year is not None and year == year and (precision is None or precision != precision or precision >= 9):
            return int(year)
    except (TypeError, ValueError):
        pass
    if isinstance(text, str) and text.strip():
        return text.strip()
    try:
        if year is not None and year == year:
            return int(year)
    except (TypeError, ValueError):
        pass
    return None


def lang_names(label, aliases, cap: int = 4) -> list:
    """Label first, then aliases, no duplicates (case-insensitive), at most ``cap`` names."""
    out, seen = [], set()
    for x in [label] + list(aliases if aliases is not None else []):
        if not isinstance(x, str):
            continue
        s = x.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
        if len(out) >= cap:
            break
    return out


_TAG = re.compile(r'<[^>]+>')


def clean_html(s, n: int = 300) -> str:
    if not s:
        return ''
    s = html.unescape(_TAG.sub(' ', str(s)))
    return re.sub(r'\s+', ' ', s).strip()[:n]


def pick_artist(artist, credit) -> str:
    """The Commons artist; else the Credit line; else 'unknown author' (a bare 'Own work' says nothing about the author)."""
    a = clean_html(artist)
    if a:
        return a
    c = clean_html(credit)
    if c and not GENERIC_OWN_WORK.match(c):
        return c
    return 'unknown author'


def clean_context(s, tier) -> str | None:
    if tier not in TIER_CONTEXT or not isinstance(s, str):
        return None
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:CONTEXT_MAX] if len(s) >= 3 else None


_LEAD_PREP = re.compile(r'^(of|in|at)\s+(?=\S{3})', re.I)


def surface_form(s) -> str:
    """Whitespace-normalised surface of a mention, without a leading English preposition ('of Boughton' -> 'Boughton'; Casebooks, Stow)."""
    t = re.sub(r'\s+', ' ', str(s or '')).strip(' ,;:')
    return _LEAD_PREP.sub('', t, count=1)


def orig_name(most_frequent, atlas_name, name_en) -> str:
    """The edition's own form of the name: the most frequent non-generic surface; else the atlas name unless it is a common noun; else the English name."""
    for c in (most_frequent, atlas_name):
        if isinstance(c, str) and c.strip() and GEN.place_generic(c) is None:
            return c
    return name_en


def doc_sort_key(r):
    yf = r[3] if isinstance(r[3], int) else 9999
    return (yf, r[2] or '~', r[0])


def _int(x):
    try:
        return None if x is None or x != x else int(x)
    except (TypeError, ValueError):
        return None


def _t(msg):
    print(f'[cards_globe {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def dump(obj, path: Path) -> int:
    s = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, 'utf-8')
    return len(s.encode('utf-8'))


# ======================================================================================================================
# letters
# ======================================================================================================================
def quarter_index(year):
    """0..9 for the quarter-century of 1450..1699, else None."""
    try:
        if year is None or year != year:
            return None
        y = int(year)
    except (TypeError, ValueError):
        return None
    return (y - 1450) // 25 if 1450 <= y < 1700 else None


def build_letters(ends, row_of: dict, sidx: dict, letter_docs=None, doc_key_fn=None, with_docs: bool = False) -> tuple:
    """Letter network between the points of the globe.

    ``ends``: DataFrame(coll_slug, source_path, doc_id, role, entity_key, doc_year), one row per canonical, non-excluded, in-period place mention of role
    ``sent_from`` / ``sent_to``. ``row_of``: {(coll_slug, entity_key): row of core.json} for the places that are on the globe. ``sidx``: {slug: index in core.json}.
    ``letter_docs``: DataFrame(doc_key, coll_slug) of the documents typed letter and in period (the denominator of the coverage; None = only the documents that have
    an end). Returns (edges, coverage): edges = [[from_row, to_row, count, [10 quarter counts], [[slug_idx, count], ...]], ...] sorted by count; coverage = list of
    dicts per edition.

    A letter is an edge when its sender end and its addressee end each resolve to exactly one point of the globe (a self-loop, a letter inside one city, is an edge
    with from_row == to_row). An end that names two different points (a letter dated in two places) is ambiguous and is not an edge.

    ``with_docs``: also return, as a third value, the doc_keys of the letters of each edge (same order as the edges), for the per-edge letter lists."""
    import pandas as pd
    from .docs_table import doc_key as _dk
    dk = doc_key_fn or _dk
    e = ends.copy()
    if len(e):
        e['doc_key'] = [dk(a, b, c) for a, b, c in zip(e['coll_slug'], e['source_path'], e['doc_id'])]
        e['row'] = [row_of.get((a, b)) for a, b in zip(e['coll_slug'], e['entity_key'])]
    else:
        e['doc_key'] = []
        e['row'] = []
    per: dict = {}
    for (dkey, role), g in e.groupby(['doc_key', 'role']):
        rows = sorted({int(r) for r in g['row'].dropna()})
        yrs = [y for y in g['doc_year'].dropna()]
        per[(dkey, role)] = (g['coll_slug'].iat[0], rows, min(yrs) if yrs else None)
    docs = sorted({k[0] for k in per})
    cov: dict = {}

    def slot(slug):
        return cov.setdefault(slug, dict(slug=slug, letters=0, sender_place=0, addressee_place=0, both_places=0, ambiguous=0, both_on_globe=0))
    edges: dict = {}
    for d in docs:
        a, b = per.get((d, 'sent_from')), per.get((d, 'sent_to'))
        slug = (a or b)[0]
        c = slot(slug)
        c['letters'] += 1
        c['sender_place'] += int(a is not None)
        c['addressee_place'] += int(b is not None)
        if a is None or b is None:
            continue
        c['both_places'] += 1
        if len(a[1]) > 1 or len(b[1]) > 1:
            c['ambiguous'] += 1
            continue
        if not a[1] or not b[1]:
            continue
        c['both_on_globe'] += 1
        ed = edges.setdefault((a[1][0], b[1][0]), dict(n=0, q=[0] * 10, sl={}, docs=[]))
        ed['n'] += 1
        ed['docs'].append(d)
        qi = quarter_index(a[2] if a[2] is not None else b[2])
        if qi is not None:
            ed['q'][qi] += 1
        if slug in sidx:
            ed['sl'][sidx[slug]] = ed['sl'].get(sidx[slug], 0) + 1
    if letter_docs is not None and len(letter_docs):
        have = set(docs)
        for slug, g in letter_docs.groupby('coll_slug'):
            c = slot(slug)
            c['letters'] += int((~g['doc_key'].isin(have)).sum())          # typed letters with no sender / addressee place at all
    out = [[f, t, ed['n'], ed['q'], sorted(([int(i), int(n)] for i, n in ed['sl'].items()), key=lambda x: (-x[1], x[0]))] for (f, t), ed in edges.items()]
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    edge_docs = [edges[(e_[0], e_[1])]['docs'] for e_ in out]
    covl = []
    for slug, c in sorted(cov.items(), key=lambda kv: (-kv[1]['letters'], kv[0])):
        c['share_both_on_globe'] = round(c['both_on_globe'] / c['letters'], 4) if c['letters'] else None
        covl.append(c)
    if with_docs:
        return out, covl, edge_docs
    return out, covl


def _s(v):
    return v if isinstance(v, str) and v.strip() else None


def letter_lists(edge_docs, docs, sidx: dict) -> dict:
    """{edge index (str): [[title, date, year_from, sender, addressee, slug_idx, url], ...]} sorted by year, for the letters of each edge of letters.json.
    ``docs``: DataFrame indexed by doc_key with title, date, year_from, author, sender, addressee, url, coll_slug. A letter missing from ``docs`` keeps its doc_key
    as title."""
    out = {}
    for i, keys in enumerate(edge_docs):
        rows = []
        for k in keys:
            if k in docs.index:
                r = docs.loc[k]
                y = r['year_from']
                rows.append([_s(r['title']) or k, _s(r['date']), int(y) if y == y and y is not None else None, _s(r['sender']) or _s(r['author']),
                             _s(r['addressee']), sidx.get(r['coll_slug']), _s(r['url'])])
            else:
                rows.append([k, None, None, None, None, None, None])
        rows.sort(key=lambda x: (x[2] if x[2] is not None else 9999, x[1] or '', x[0]))
        out[str(i)] = rows
    return out



def register_name(lg):
    """The name an edition's register gives a place: the entry of the most mentioned local place that has a register entry (its key is not ``surf:...``, the name is not a
    common noun), shortened to the part before a comma / bracket where that leaves a name ('Marignano', not the surface form 'Marian' of a letter). None if there is none."""
    d = lg[~lg['entity_key'].astype(str).str.startswith('surf:')].sort_values('n_in_period', ascending=False)
    for n in d['name']:
        if not isinstance(n, str) or not n.strip() or GEN.place_generic(n) is not None or BOT_LABEL.search(n):
            continue
        return surface_form(n)[:60]
    return None


# ======================================================================================================================
# build
# ======================================================================================================================
def build(a) -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from .docs_table import doc_key
    from .merge import _norm

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {'generated': time.strftime('%Y-%m-%d %H:%M')}

    # ------------------------------------------------------------------------------------------------ points
    G = json.loads((a.final / 'globe_places.json').read_text('utf-8'))
    items = G['places']
    rows_id = [p['id'] for p in items]
    row_of = {g: i for i, g in enumerate(rows_id)}
    _t(f'{len(items)} globe points')
    pg = pd.read_parquet(a.final / 'places_global.parquet',
                         columns=['id', 'globe_id', 'name', 'qid', 'geonames', 'tgn', 'label_en', 'n_in_period', 'classes', 'place_kind', 'coord_source'])
    pg = pg[pg['globe_id'].isin(row_of)].copy()
    pg['row'] = pg['globe_id'].map(row_of)
    pl = pd.read_parquet(a.final / 'places_local_geo.parquet',
                         columns=['coll_slug', 'entity_key', 'global_id', 'name', 'n_in_period', 'note_type', 'gn_fcode', 'tgn_type', 'tgn_id',
                                  'gc_cand', 'gc_cand_name', 'cb_entry', 'coord_source', 'cb_used'])
    pg_row = dict(zip(pg['id'], pg['row']))
    pl['row'] = pl['global_id'].map(pg_row)

    # ------------------------------------------------------------------------------------------------ mentions -> (point, document)
    _t('mentions')
    cols = ['coll_slug', 'kind', 'canonical', 'excluded', 'local_key', 'surface', 'in_period', 'source_path', 'doc_id', 'tier', 'context', 'role', 'doc_year']
    t = pq.read_table(a.mentions, columns=cols)
    t = t.filter(pc.and_(pc.and_(pc.equal(t['kind'], 'place'), pc.equal(t['canonical'], True)), pc.equal(t['in_period'], True)))
    m = t.to_pandas()
    m = m[~m['excluded'].fillna(False).astype(bool)].copy()
    surf = m['surface'].fillna('')
    nm = {s: _norm(s) for s in surf.unique()}
    nn = surf.map(nm)
    m['entity_key'] = np.where(m['local_key'].notna(), m['local_key'], 'surf:' + nn)
    m = m[(m['local_key'].notna() | (nn != '')).to_numpy()]
    key = pl[pl['row'].notna()][['coll_slug', 'entity_key', 'row']].drop_duplicates(['coll_slug', 'entity_key'])
    m_all = m[m['role'].isin(LETTER_ENDS)][['coll_slug', 'source_path', 'doc_id', 'role', 'entity_key', 'doc_year']]     # letters: before the join with the globe
    m = m.merge(key, on=['coll_slug', 'entity_key'], how='inner')
    m['row'] = m['row'].astype('int64')
    _t(f'{len(m)} mentions on globe points')
    m['doc_key'] = [doc_key(s, p, d) for s, p, d in zip(m['coll_slug'], m['source_path'], m['doc_id'])]
    m['ctx'] = [clean_context(c, tr) for c, tr in zip(m['context'], m['tier'])]
    m['form'] = [surface_form(s) for s in m['surface']]

    # most frequent edition form per point
    ok_form = m['form'].str.len().between(1, 60)
    fc = m[ok_form].groupby(['row', 'form']).size().rename('n').reset_index()
    fc['generic'] = [GEN.place_generic(f) is not None for f in fc['form']]          # v2.2: a common noun ('alhier', 'this towne') is never the original name
    fc = fc.sort_values(['row', 'generic', 'n', 'form'], ascending=[True, True, False, True]).drop_duplicates('row')
    original = {r: f for r, f, g in zip(fc['row'], fc['form'], fc['generic']) if not g}
    rep['original_name_only_generic_forms'] = int(fc['generic'].sum())

    # documents per point
    grp = m.groupby(['row', 'doc_key'], sort=False)
    dd = grp.agg(n=('doc_key', 'size'), coll_slug=('coll_slug', 'first'), tier=('tier', 'first')).reset_index()
    mc = m[m['ctx'].notna()].drop_duplicates(['row', 'doc_key'])[['row', 'doc_key', 'ctx']]
    dd = dd.merge(mc, on=['row', 'doc_key'], how='left')
    n_by_row = dd.groupby('row')['n'].sum()
    bad = [(i, int(n_by_row.get(i, 0)), int(items[i]['n'])) for i in range(len(items)) if int(n_by_row.get(i, 0)) != int(items[i]['n'])]
    rep['docs_mentions_vs_globe_n_mismatch_points'] = len(bad)
    rep['docs_mentions_vs_globe_n_mismatch_examples'] = bad[:10]
    _t(f'{len(dd)} (point, document) pairs; {len(bad)} points whose mention count differs from the globe n')

    docs = pd.read_parquet(a.documents, columns=['doc_key', 'title', 'date', 'year_from', 'author', 'sender', 'addressee', 'doc_type', 'url', 'coll_slug', 'in_period'])
    docs = docs.drop_duplicates('doc_key').set_index('doc_key')
    dd['has_doc'] = dd['doc_key'].isin(docs.index)
    rep['docs_pairs_without_document_row'] = int((~dd['has_doc']).sum())

    # ------------------------------------------------------------------------------------------------ slugs and entries
    slugs = sorted(set(dd['coll_slug']) | {c for p in items for c in (p.get('cq') or {})} | {c for p in items for c in p.get('cols', [])})
    sidx = {s: i for i, s in enumerate(slugs)}
    hdr = {c['slug']: c for c in G['header']['collections']}
    rid = pq.read_table(a.mentions, columns=['coll_slug', 'renascor_id', 'kind']).to_pandas()
    rid = rid[rid['kind'] == 'place']
    ids_of: dict = defaultdict(set)
    for s, r, _k in rid.drop_duplicates(['coll_slug', 'renascor_id']).itertuples(index=False, name=None):
        for x in str(r or '').split(';'):
            if x:
                ids_of[s].add(x)
    entries = []
    for s in slugs:
        lst = []
        for x in sorted(ids_of.get(s) or set((hdr.get(s, {}).get('renascor_ids') or '').split(';')) - {''}):
            p = a.entries / f'{x}.json'
            title = json.loads(p.read_text('utf-8')).get('title') if p.exists() else None
            lst.append([x, title or x])
        entries.append(lst)
    tiers = [hdr.get(s, {}).get('tier') for s in slugs]
    licences = [hdr.get(s, {}).get('licence') for s in slugs]

    # ------------------------------------------------------------------------------------------------ Wikidata cards and names
    _t('Wikidata')
    cards = pd.read_parquet(a.cards / 'wd_places.parquet').set_index('qid')
    wde = pd.read_parquet(a.wd_entities).set_index('qid')
    cl = pd.read_csv(a.wd_class_labels, sep='\t', dtype=str, keep_default_na=False)
    classlab = dict(zip(cl.iloc[:, 0], cl.iloc[:, 1]))

    def ent_labels(q):
        if q in wde.index:
            s = wde.at[q, 'labels']
            try:
                return json.loads(s) if isinstance(s, str) else {}
            except ValueError:
                return {}
        return {}

    def ent_aliases(q):
        if q in wde.index:
            s = wde.at[q, 'aliases']
            try:
                return json.loads(s) if isinstance(s, str) else {}
            except ValueError:
                return {}
        return {}

    # qid of each point (with the duplicate-item fix)
    qid_of = {}
    for i, p in enumerate(items):
        q = p.get('qid')
        if q in DUPLICATE_ITEMS and (DUPLICATE_ITEMS[q] in cards.index or DUPLICATE_ITEMS[q] in wde.index):
            rep.setdefault('duplicate_items_replaced', []).append([q, DUPLICATE_ITEMS[q]])
            q = DUPLICATE_ITEMS[q]
        qid_of[i] = q
    qids = sorted({q for q in qid_of.values() if q and q not in DOUBTFUL_ITEMS})   # a wrong item is neither linked nor described

    def wd_en(q):
        if q in cards.index and isinstance(cards.at[q, 'label_en'], str) and cards.at[q, 'label_en'].strip() and not BOT_LABEL.search(cards.at[q, 'label_en']):
            return cards.at[q, 'label_en'].strip()
        l = ent_labels(q).get('en')
        return l.strip() if isinstance(l, str) and l.strip() and not BOT_LABEL.search(l) else None

    # TGN and GeoNames English names, only for the ids that are needed
    def ids_of_col(series):
        res = {}
        for r, v in zip(series.index, series):
            res[r] = [x for x in re.split(r'[;,|\s]+', str(v)) if x.isdigit()] if isinstance(v, str) else []
        return res

    mem = pg.sort_values(['row', 'n_in_period'], ascending=[True, False])
    need_tgn, need_gn = set(), set()
    for _r, g in mem.groupby('row'):
        for v in g['tgn'].dropna():
            need_tgn.update(x for x in re.split(r'[;,|\s]+', str(v)) if x.isdigit())
        for v in g['geonames'].dropna():
            need_gn.update(x for x in re.split(r'[;,|\s]+', str(v)) if x.isdigit())
    plr = pl[pl['row'].notna()]
    for v in plr['tgn_id'].dropna():
        need_tgn.update(x for x in re.split(r'[;,|\s]+', str(v)) if x.isdigit())
    for v in plr['gc_cand'].dropna():
        if str(v).startswith('gn:'):
            need_gn.add(str(v)[3:])
    tgn_en, tgn_pref = {}, {}
    tp = pq.read_table(a.tgn / 'tgn_places.parquet', columns=['subject_id', 'name'], filters=[('subject_id', 'in', sorted(need_tgn))]).to_pandas()
    tgn_pref = dict(zip(tp['subject_id'], tp['name']))
    tn = pq.read_table(a.tgn / 'tgn_names.parquet', columns=['subject_id', 'name', 'language', 'is_preferred', 'historical'],
                       filters=[('subject_id', 'in', sorted(need_tgn))]).to_pandas()
    tn = tn[(tn['language'] == 'en') & ~tn['historical'].fillna(False).astype(bool)]
    tn = tn.sort_values(['subject_id', 'is_preferred'], ascending=[True, False]).drop_duplicates('subject_id')
    tgn_en = dict(zip(tn['subject_id'], tn['name']))
    gi = pq.read_table(a.gn_index, columns=['gid', 'name', 'prim'], filters=[('gid', 'in', sorted(need_gn))]).to_pandas()
    gi = gi.sort_values(['gid', 'prim'], ascending=[True, False]).drop_duplicates('gid')
    gn_name = dict(zip(gi['gid'], gi['name']))
    _t(f'TGN names for {len(tgn_en)}/{len(need_tgn)} subjects, GeoNames names for {len(gn_name)}/{len(need_gn)} features')

    members = {r: g for r, g in mem.groupby('row')}
    locs = {r: g for r, g in plr.groupby('row')}
    name_en, name_src, kinds = {}, Counter(), {}
    kind_src = Counter()
    for i, p in enumerate(items):
        g = members.get(i)
        lg = locs.get(i)
        q = qid_of[i]
        nme, src = None, None
        if q and q in DOUBTFUL_ITEMS and DOUBTFUL_ITEMS[q][0]:
            nme, src = DOUBTFUL_ITEMS[q][0], 'display_fix'
        if nme is None and q and q not in DOUBTFUL_ITEMS:
            nme, src = wd_en(q), 'wikidata'
        if nme is None and g is not None:
            for qq in g['qid'].dropna():
                if qq in DOUBTFUL_ITEMS:
                    continue
                nme = wd_en(qq)
                if nme:
                    src = 'wikidata'
                    break
        if nme is None and g is not None:
            for v in g['label_en'].dropna():
                if isinstance(v, str) and v.strip() and not BOT_LABEL.search(v):
                    nme, src = v.strip(), 'exonym'
                    break
        if nme is None and g is not None:
            for v in g['tgn'].dropna():
                for x in re.split(r'[;,|\s]+', str(v)):
                    if x in tgn_en or x in tgn_pref:
                        nme, src = tgn_en.get(x) or tgn_pref.get(x), 'tgn'
                        break
                if nme:
                    break
        if nme is None and g is not None:
            for v in g['geonames'].dropna():
                for x in re.split(r'[;,|\s]+', str(v)):
                    if x in gn_name:
                        nme, src = gn_name[x], 'geonames'
                        break
                if nme:
                    break
        if nme is None and lg is not None:
            nme, src = register_name(lg), 'register_name'                 # v2.3: the edition's own register form, before any surface form
            if not nme:
                nme, src = None, None
        if nme is None and lg is not None:
            for gc, gcn in zip(lg.sort_values('n_in_period', ascending=False)['gc_cand'], lg.sort_values('n_in_period', ascending=False)['gc_cand_name']):
                if isinstance(gc, str) and gc.startswith('gn:'):
                    nme, src = gn_name.get(gc[3:]) or (gcn if isinstance(gcn, str) else None), 'geonames'
                    break
        if nme is None and lg is not None:                                # v2.3: a point copied from, or named like, another place of the atlas: that place's name
            for gc, gcn in zip(lg.sort_values('n_in_period', ascending=False)['gc_cand'], lg.sort_values('n_in_period', ascending=False)['gc_cand_name']):
                if isinstance(gc, str) and not gc.startswith(('gn:', 'tgn:')) and isinstance(gcn, str) and gcn.strip() and GEN.place_generic(gcn) is None \
                        and not BOT_LABEL.search(gcn):
                    nme, src = gcn.strip(), 'candidate_name'
                    break
        if nme is None and lg is not None:
            for ce, cu in zip(lg.sort_values('n_in_period', ascending=False)['cb_entry'], lg.sort_values('n_in_period', ascending=False)['cb_used']):
                if isinstance(ce, str) and ce.strip() and cu:
                    nme, src = ce.strip(), 'casebooks'
                    break
        if nme is None:
            nme, src = original.get(i) or surface_form(p.get('name')), 'surface'
            if not nme or GEN.place_generic(nme) is not None:          # v2.2: a common noun is never the display name, whatever else is missing
                nme, src = UNNAMED, 'unnamed_generic_only'
        name_en[i] = nme
        name_src[src] += 1
        # kind
        wdl = []
        if q:
            if q in cards.index and cards.at[q, 'instance_of_en'] is not None:
                wdl += [x for x in cards.at[q, 'instance_of_en']]
            elif q in wde.index and wde.at[q, 'classes'] is not None:
                wdl += [classlab.get(c, '') for c in wde.at[q, 'classes']]
        gnf = list(lg['gn_fcode'].dropna()) if lg is not None and 'gn_fcode' in lg else []
        tgt = list(lg['tgn_type'].dropna()) if lg is not None else []
        nts = list(lg['note_type'].dropna()) if lg is not None else []
        base = p.get('k')
        if base in ('c', 'a', 'n', 'b'):                      # decided by the atlas from all its sources
            k = base
        else:                                                 # 'u' (or the three-way word of an older file): read again with the cards' class labels
            k = kind_letter(base, wdl, gnf, tgt, nts)
        kinds[i] = k
        kind_src[(base, k)] += 1
    rep['name_en_sources'] = dict(name_src)
    rep['kinds'] = dict(Counter(kinds.values()))
    rep['kind_from_atlas_kind'] = {f'{a_}>{b_}': v for (a_, b_), v in kind_src.items()}
    _t(f'names {dict(name_src)}; kinds {dict(Counter(kinds.values()))}')

    # ------------------------------------------------------------------------------------------------ images
    _t('images')
    im = pd.read_csv(a.images, sep='\t', dtype=str, keep_default_na=False)
    meta = json.loads(Path(a.img_meta).read_text('utf-8'))
    iss = pd.read_csv(a.issues, sep='\t', dtype=str, keep_default_na=False)
    weak = set(iss.loc[iss['issue'] == 'image is a map, flag, arms or collage', 'qid'])
    in_core = set(qids)
    img_rec, img_drop = {}, Counter()
    for r in im.itertuples(index=False):
        if r.qid not in in_core:
            img_drop['place_not_on_globe'] += 1
            continue
        if r.qid in weak:
            img_drop['weak_image_map_flag_arms_collage'] += 1
            continue
        if r.qid in DOUBTFUL_ITEMS:
            img_drop['doubtful_item'] += 1
            continue
        f = Path(a.img_dir) / r.local
        if not f.exists():
            img_drop['file_missing'] += 1
            continue
        credit = ((meta.get(r.qid) or {}).get('info') or {}).get('meta', {}).get('Credit', '')
        art = pick_artist(r.artist, credit)
        img_rec[r.qid] = (f, art, r.licence, r.licence_url, r.file_page, bool(not clean_html(r.artist)))
    rep['images'] = {'kept': len(img_rec), 'dropped': dict(img_drop),
                     'artist_from_credit_or_unknown': sum(1 for v in img_rec.values() if v[5]),
                     'unknown_author': sum(1 for v in img_rec.values() if v[1] == 'unknown author')}

    # ------------------------------------------------------------------------------------------------ wd.json
    wd = {}
    miss_cards = []
    for q in qids:
        lab = ent_labels(q)
        ali = ent_aliases(q)
        if q in cards.index:
            r = cards.loc[q]
            desc = r['desc_en'] if isinstance(r['desc_en'], str) and r['desc_en'].strip() else None
            types = meaningful_types(list(r['instance_of_en']) if r['instance_of_en'] is not None else [])
            country = first_label(list(r['country_en']) if r['country_en'] is not None else [])
            loc = first_label(list(r['located_in_en']) if r['located_in_en'] is not None else [])
            inc = inception_value(r['inception'], r['inception_year'], r['inception_precision'])
            langs = {}
            for lg_ in ('la', 'it', 'fr', 'de', 'es', 'nl'):
                v = lang_names(r.get('label_' + lg_), r.get('aliases_' + lg_))
                if v:
                    langs[lg_] = v
        else:
            miss_cards.append(q)
            desc, inc, loc = None, None, ''
            types = meaningful_types([classlab.get(c, '') for c in (wde.at[q, 'classes'] if q in wde.index and wde.at[q, 'classes'] is not None else [])])
            cq_ = wde.at[q, 'country_qid'] if q in wde.index else None
            country = ''
            if isinstance(cq_, str):
                country = (ent_labels(cq_).get('en') if cq_ in wde.index else None) or classlab.get(cq_, '')
                country = '' if re.fullmatch(r'Q\d+', country or '') else (country or '')
            langs = {}
            for lg_ in ('la', 'it', 'fr', 'de', 'es', 'nl'):
                v = lang_names(lab.get(lg_), ali.get(lg_))
                if v:
                    langs[lg_] = v
        if q in DOUBTFUL_ITEMS and not DOUBTFUL_ITEMS[q][1]:
            desc, types, country, loc, inc, langs = None, [], '', '', None, {}
        wd[q] = [desc, types, country, loc, inc, langs, img_shard(q) if q in img_rec else None]
    rep['wd'] = {'qids': len(qids), 'with_card': len(qids) - len(miss_cards), 'without_card_offline_only': len(miss_cards),
                 'with_description': sum(1 for v in wd.values() if v[0]), 'with_country': sum(1 for v in wd.values() if v[2]),
                 'with_image': sum(1 for v in wd.values() if v[6] is not None)}
    (out / 'wd_qids_without_card.txt').write_text('\n'.join(miss_cards) + '\n')

    # ------------------------------------------------------------------------------------------------ core.json
    rows = []
    for i, p in enumerate(items):
        cq = p.get('cq') or {}
        per = sorted(((sidx[c], [int(x) for x in v]) for c, v in cq.items() if c in sidx), key=lambda x: (-sum(x[1]), x[0]))
        rows.append([
            name_en[i], round(float(p['lat']), 3), round(float(p['lon']), 3), [int(x) for x in p['q']], int(p.get('c') or 1),
            [sidx[c] for c in p.get('cols', []) if c in sidx], None if qid_of[i] in DOUBTFUL_ITEMS else qid_of[i], p.get('s'), int(p.get('t') or 0), int(p['n']),
            orig_name(original.get(i), surface_form(p.get('name')), name_en[i]), kinds[i], per, doc_shard(i)])
    core = {
        'quarters': QUARTERS, 'slugs': slugs, 'tiers': tiers, 'licences': licences, 'entries': entries, 'rows': rows,
        'mentions_total': int(sum(r[9] for r in rows)), 'places_total': len(rows),
        'row_fields': ['name_en', 'lat', 'lon', 'q', 'n_collections', 'top_slugs', 'qid', 'coord_source', 'worst_tier', 'n_total',
                       'original_name', 'kind', 'per_slug_q', 'doc_shard'],
        'source_codes': SOURCE_CODES, 'located_by_name': list(NAME_BASED_CODES), 'kinds': PK.KIND_LABEL,
        'notes': [
            'rows follow globe_places.json order (in-period mentions, most mentioned first); row index = position in rows',
            'q = mentions per quarter-century 1450..1675 by document year (sum can be below n_total: documents dated across a boundary)',
            'per_slug_q = [[slug_idx, q10], ...] sorted by total; top_slugs = the 3 editions with most mentions',
            'worst_tier: 1 = T1, 2 = T2, 3 = T3 (no context stored; published per owner decision of 2026-11-04) among all contributing editions',
            'coordinates are rounded to 3 decimals (about 100 m); high / medium confidence only',
            'documents of row i: docs/<i mod 64>.json, key str(i)',
        ]}
    rep['core_bytes'] = dump(core, out / 'core.json')
    rep['wd_bytes'] = dump(wd, out / 'wd.json')

    # ------------------------------------------------------------------------------------------------ letters.json
    _t('letters')
    row_of_ent = {(c, k): int(r) for c, k, r in zip(key['coll_slug'], key['entity_key'], key['row'])}
    lett_docs = docs[(docs['doc_type'] == 'letter') & (docs['in_period'] == True)].reset_index()[['doc_key', 'coll_slug']]      # noqa: E712
    edges, cov, edocs = build_letters(m_all, row_of_ent, sidx, lett_docs, with_docs=True)
    rep['letters'] = {'edges': len(edges), 'letters_in_edges': int(sum(e_[2] for e_ in edges)), 'self_loops': int(sum(1 for e_ in edges if e_[0] == e_[1])),
                      'by_edition': cov,
                      'total': {k_: int(sum(c[k_] for c in cov)) for k_ in ('letters', 'sender_place', 'addressee_place', 'both_places', 'ambiguous', 'both_on_globe')}}
    rep['letters']['total']['share_both_on_globe'] = round(rep['letters']['total']['both_on_globe'] / max(rep['letters']['total']['letters'], 1), 4)
    rep['letters_bytes'] = dump(edges, out / 'letters.json')
    rep['letters_docs_bytes'] = dump(letter_lists(edocs, docs, sidx), out / 'letters_docs.json')
    pd.DataFrame(cov).to_csv(out / 'letters_coverage.tsv', sep='\t', index=False)
    _t(f"letters: {len(edges)} edges for {rep['letters']['total']['both_on_globe']} of {rep['letters']['total']['letters']} letters")

    # ------------------------------------------------------------------------------------------------ docs shards
    _t('document shards')
    dd = dd.merge(docs[['title', 'date', 'year_from', 'author', 'sender', 'addressee', 'doc_type', 'url']], left_on='doc_key', right_index=True, how='left')
    shards: dict = {k: {} for k in range(DOC_SHARDS)}
    n_pairs = 0
    for r in dd.itertuples(index=False):
        i = int(r.row)
        snd = r.sender if isinstance(r.sender, str) and r.sender else (r.author if isinstance(r.author, str) and r.author else None)
        title = r.title if isinstance(r.title, str) and r.title else r.doc_key
        rec = [r.doc_key, title[:160], r.date if isinstance(r.date, str) else None, _int(r.year_from), snd,
               r.addressee if isinstance(r.addressee, str) and r.addressee else None, r.doc_type if isinstance(r.doc_type, str) else None,
               sidx[r.coll_slug], int(r.n), r.url if isinstance(r.url, str) and r.url else None,
               r.ctx if isinstance(r.ctx, str) else None]
        shards[doc_shard(i)].setdefault(str(i), []).append(rec)
        n_pairs += 1
    sizes = {}
    for k, d in shards.items():
        for lst in d.values():
            lst.sort(key=doc_sort_key)
        sizes[f'{k:02d}'] = dump(d, out / 'docs' / f'{k:02d}.json')
    rep['docs'] = {'pairs': n_pairs, 'total_bytes': sum(sizes.values()), 'largest_shard': max(sizes.items(), key=lambda kv: kv[1]),
                   'smallest_shard': min(sizes.items(), key=lambda kv: kv[1]), 'sizes': sizes,
                   'points_with_docs': int(dd['row'].nunique()), 'distinct_documents': int(dd['doc_key'].nunique()),
                   'with_context': int(dd['ctx'].notna().sum()), 'with_url': int(dd['url'].notna().sum()),
                   'by_tier': {str(k): int(v) for k, v in dd['tier'].value_counts().items()}}

    # ------------------------------------------------------------------------------------------------ image shards
    import base64
    ish: dict = {k: {} for k in range(IMG_SHARDS)}
    for q, (f, art, lic, lurl, fpage, _x) in img_rec.items():
        b = base64.b64encode(Path(f).read_bytes()).decode('ascii')
        ish[img_shard(q)][q] = {'src': 'data:image/webp;base64,' + b, 'artist': art, 'licence': lic, 'licence_url': lurl, 'file_page': fpage}
    isizes = {f'{k:02d}': dump(d, out / 'img' / f'{k:02d}.json') for k, d in ish.items()}
    rep['img'] = {'images': len(img_rec), 'total_bytes': sum(isizes.values()), 'largest_shard': max(isizes.items(), key=lambda kv: kv[1]), 'sizes': isizes}

    # places whose card has an image in Commons but no thumbnail was fetched (needs the network, not done)
    have_p18 = [q for q in qids if q in cards.index and isinstance(cards.at[q, 'image'], str) and cards.at[q, 'image'] and q not in img_rec
                and q not in weak and q not in set(im['qid'])]
    rep['qids_with_p18_but_no_thumbnail'] = len(have_p18)
    (out / 'qids_with_p18_no_thumbnail.txt').write_text('\n'.join(have_p18) + '\n')
    (out / 'build_report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    return rep


def letters_only(a) -> dict:
    """Rebuild letters.json and letters_docs.json from the final tables, the mentions and the documents, without the rest of the card build (same steps as
    ``build``). With --check, the edges must equal an existing letters.json."""
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from .merge import _norm
    import numpy as np
    G = json.loads((a.final / 'globe_places.json').read_text('utf-8'))
    row_of = {p['id']: i for i, p in enumerate(G['places'])}
    pg = pd.read_parquet(a.final / 'places_global.parquet', columns=['id', 'globe_id'])
    pg = pg[pg['globe_id'].isin(row_of)]
    pg_row = dict(zip(pg['id'], pg['globe_id'].map(row_of)))
    pl = pd.read_parquet(a.final / 'places_local_geo.parquet', columns=['coll_slug', 'entity_key', 'global_id'])
    pl['row'] = pl['global_id'].map(pg_row)
    cols = ['coll_slug', 'kind', 'canonical', 'excluded', 'local_key', 'surface', 'in_period', 'source_path', 'doc_id', 'role', 'doc_year']
    t = pq.read_table(a.mentions, columns=cols)
    t = t.filter(pc.and_(pc.and_(pc.equal(t['kind'], 'place'), pc.equal(t['canonical'], True)), pc.equal(t['in_period'], True)))
    t = t.filter(pc.is_in(t['role'], value_set=pa_array(LETTER_ENDS)))
    m = t.to_pandas()
    m = m[~m['excluded'].fillna(False).astype(bool)].copy()
    surf = m['surface'].fillna('')
    nn = surf.map({s: _norm(s) for s in surf.unique()})
    m['entity_key'] = np.where(m['local_key'].notna(), m['local_key'], 'surf:' + nn)
    m = m[(m['local_key'].notna() | (nn != '')).to_numpy()]
    key = pl[pl['row'].notna()].drop_duplicates(['coll_slug', 'entity_key'])
    row_of_ent = {(c, k): int(r) for c, k, r in zip(key['coll_slug'], key['entity_key'], key['row'])}
    core = json.loads(a.core.read_text('utf-8'))
    sidx = {s: i for i, s in enumerate(core['slugs'])}
    docs = pd.read_parquet(a.documents, columns=['doc_key', 'title', 'date', 'year_from', 'author', 'sender', 'addressee', 'doc_type', 'url', 'coll_slug', 'in_period'])
    docs = docs.drop_duplicates('doc_key').set_index('doc_key')
    lett_docs = docs[(docs['doc_type'] == 'letter') & (docs['in_period'] == True)].reset_index()[['doc_key', 'coll_slug']]      # noqa: E712
    ends = m[['coll_slug', 'source_path', 'doc_id', 'role', 'entity_key', 'doc_year']]
    edges, cov, edocs = build_letters(ends, row_of_ent, sidx, lett_docs, with_docs=True)
    rep = {'edges': len(edges), 'letters_in_edges': int(sum(e_[2] for e_ in edges))}
    if a.check:
        old = json.loads(a.check.read_text('utf-8'))
        rep['same_as_check'] = old == edges
        if old != edges:
            raise SystemExit(f'edges differ from {a.check}: {len(old)} vs {len(edges)}')
    rep['letters_bytes'] = dump(edges, a.out / 'letters.json')
    rep['letters_docs_bytes'] = dump(letter_lists(edocs, docs, sidx), a.out / 'letters_docs.json')
    rep['without_document_row'] = int(sum(1 for ks in edocs for k in ks if k not in docs.index))
    return rep


def pa_array(values):
    import pyarrow as pa
    return pa.array(list(values))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas.cards_globe')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--final', type=Path, required=True, help='directory with globe_places.json, places_global.parquet, places_local_geo.parquet')
    b.add_argument('--mentions', type=Path, required=True)
    b.add_argument('--documents', type=Path, required=True)
    b.add_argument('--cards', type=Path, required=True, help='directory with wd_places.parquet')
    b.add_argument('--entries', type=Path, required=True, help='Renascor data/entries')
    b.add_argument('--tgn', type=Path, required=True)
    b.add_argument('--gn-index', type=Path, required=True, dest='gn_index')
    b.add_argument('--wd-entities', type=Path, required=True, dest='wd_entities')
    b.add_argument('--wd-class-labels', type=Path, required=True, dest='wd_class_labels')
    b.add_argument('--images', type=Path, required=True)
    b.add_argument('--img-dir', type=Path, required=True, dest='img_dir')
    b.add_argument('--img-meta', type=Path, required=True, dest='img_meta')
    b.add_argument('--issues', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    lt = sub.add_parser('letters', help='letters.json and letters_docs.json only')
    lt.add_argument('--final', type=Path, required=True)
    lt.add_argument('--mentions', type=Path, required=True)
    lt.add_argument('--documents', type=Path, required=True)
    lt.add_argument('--core', type=Path, required=True, help='core.json of the same build (edition order)')
    lt.add_argument('--check', type=Path, help='letters.json the edges must equal')
    lt.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'letters':
        print(json.dumps(letters_only(a), indent=1))
        return 0
    rep = build(a)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('docs', 'img')}, indent=1, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
