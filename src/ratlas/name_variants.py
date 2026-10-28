"""Names of the places of the atlas: one reference form, the variants with language, edition and count, a folded key (v2.4 preparation; pure functions plus one builder).

Why. A global place is named by dozens of surface forms ('Zürich', 'Zuerich', 'Tigurum', 'Zurych', 'Tiguri', 'Zurich in der Schweiz') in editions of five languages; the page shows one of them,
the NER fine-tuning needs all of them with their language. This module gives each global place

* one **reference form**: the English Wikidata label, else the name of the edition's register, else the exonym table's English label, else the most frequent surface (``reference_form``,
  ``REFERENCE_ORDER``);
* its **variants**: the surface forms of its mentions with language, edition and count, the forms that are the same name merged into one entry (``group_variants``). Merging is by the
  folded key and **only inside one place**: two places whose names fold together are two places (the key is for matching, never an identity);
* the **folded key** (``fold_key``): lower case, no diacritics, 'ß' -> 'ss', the digraphs ae oe ue as a o u (German umlauts and the ligatures æ œ), u/v, i/j, y/i. Measured on the 63,084 distinct
  surface forms of the 14,798 points of v2.3 against plain accent folding: 2,606 pairs of forms of one point merge, 137 pairs of forms of different points only (19:1); without the
  digraph step 1,791 and 96; adding w/u would merge 54 more pairs across points for 54 more inside (1:1), so it is not done.

``build_extra`` builds the ``extra.json`` of the globe page from the final tables of a build and the mentions: per row of ``core.json`` the fine type (``placetypes``), the ten
most frequent variants and the count of mentions located by a name (fix 5 of the check of v2.3: 'located by name' is the share of the mentions whose point was inferred from a name:
name propagation, GeoNames / TGN name match, exonym and polity table, Latin name; most Latin mentions are filed under Wikidata or the register on the globe, so the row's own
source code hides them).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import generic_terms as GEN
from .namegate import fold as _ng_fold
from .namegate import small_letters

#: order in which the sources of a reference form are tried (the owner's rule of 2026-10-20); a source that gives nothing is skipped
REFERENCE_ORDER = ('wikidata', 'register', 'exonym', 'surface')
#: coord_source values of ``places_local_geo`` that mean 'located by a name' (see ``placetypes.NAME_BASED_SOURCES``)
NAME_BASED_SOURCES = ('name_propagation', 'name_geonames', 'name_exonym', 'name_latin')
TOP_VARIANTS = 10
MAX_FORM_LEN = 60

#: a Wikidata label that is a bot's working title ('lsjbot Duisburg'), never a name (as ``cards_globe.BOT_LABEL``)
BOT_LABEL = re.compile(r'\b(lsjbot|bot[- ]generated|wikibot)\b', re.I)


# ======================================================================================================================
# 1. folding and cleaning
# ======================================================================================================================
def fold_key(s) -> str:
    """Folded key of a name: ``fold_key('Zürich') == fold_key('Zuerich') == fold_key('ZURICH')``; ``fold_key('Aegypten') == fold_key('Ägypten')``; ``fold_key('Jena') == fold_key('Iena')``.
    Steps: small combining letters of print read as letters, case folded, 'ß' -> 'ss', ligatures and Nordic letters transliterated
    (``namegate.fold``), diacritics removed, ae oe ue -> a o u, v -> u, j -> i, y -> i, everything that is not a letter or digit a space, spaces collapsed. Not an identity."""
    if not isinstance(s, str) or not s:
        return ''
    t = _ng_fold(small_letters(s).replace('ſ', 's').replace('ę', 'ae').replace('Ę', 'ae'))        # e caudata (ę) is ae / oe in early-modern Latin print
    t = re.sub(r'[^a-z0-9]+', ' ', t)
    t = t.replace('ae', 'a').replace('oe', 'o').replace('ue', 'u').replace('v', 'u').replace('j', 'i').replace('y', 'i')
    return ' '.join(t.split())


_LEAD_PREP = re.compile(r'^(of|in|at)\s+(?=\S{3})', re.I)


def clean_form(s) -> str:
    """Whitespace-normalised surface form without edge punctuation and without a leading English preposition ('of Boughton' -> 'Boughton', as ``cards_globe.surface_form``);
    NFC. Empty for a non-string."""
    if not isinstance(s, str):
        return ''
    t = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', s)).strip(' ,;:')
    return _LEAD_PREP.sub('', t, count=1)


def usable_form(s) -> bool:
    """A form that can be a variant: 2 to 60 characters, at least one letter, not a common noun or deictic ('alhier', 'this towne': ``generic_terms.place_generic``)."""
    if not isinstance(s, str) or not (2 <= len(s) <= MAX_FORM_LEN):
        return False
    if not any(c.isalpha() for c in s):
        return False
    return GEN.place_generic(s) is None


# ======================================================================================================================
# 2. languages
# ======================================================================================================================
_LANG = {
    'de': 'de', 'deu': 'de', 'ger': 'de', 'gsw': 'de', 'goh': 'de', 'gmh': 'de', 'bar': 'de', 'fr': 'fr', 'fra': 'fr', 'fre': 'fr', 'frm': 'fr', 'fro': 'fr', 'en': 'en', 'eng': 'en',
    'enm': 'en', 'ang': 'en', 'sco': 'en', 'la': 'la', 'lat': 'la', 'it': 'it', 'ita': 'it', 'spa': 'es', 'es': 'es', 'nl': 'nl', 'nld': 'nl', 'dut': 'nl', 'dum': 'nl', 'pt': 'pt',
    'por': 'pt', 'pl': 'pl', 'pol': 'pl', 'cs': 'cs', 'ces': 'cs', 'cze': 'cs', 'sv': 'sv', 'swe': 'sv', 'da': 'da', 'dan': 'da', 'hu': 'hu', 'hun': 'hu', 'el': 'el', 'ell': 'el',
    'gre': 'el', 'grc': 'grc', 'he': 'he', 'heb': 'he', 'hbo': 'he', 'ca': 'ca', 'cat': 'ca', 'ro': 'ro', 'ron': 'ro', 'rum': 'ro', 'ru': 'ru', 'rus': 'ru', 'tr': 'tr', 'tur': 'tr',
    'ota': 'ota', 'gez': 'gez', 'sk': 'sk', 'slk': 'sk', 'hr': 'hr', 'hrv': 'hr', 'sl': 'sl', 'slv': 'sl', 'nds': 'nds', 'gml': 'nds', 'lb': 'lb', 'fy': 'fy', 'fry': 'fy',
    'rm': 'rm', 'roh': 'rm', 'cy': 'cy', 'cym': 'cy', 'ga': 'ga', 'gle': 'ga', 'gd': 'gd', 'gla': 'gd', 'eu': 'eu', 'eus': 'eu', 'ar': 'ar', 'ara': 'ar', 'fa': 'fa', 'fas': 'fa',
}


def lang_code(code) -> str:
    """Two-letter code (three-letter for languages without one: 'grc', 'ota', 'gez') of the many spellings the editions use: de, deu, ger, de-gml -> 'de' or 'nds', frm -> 'fr',
    lat -> 'la'. '' for none or unknown ('?', None)."""
    if not isinstance(code, str):
        return ''
    c = code.strip().lower()
    if not c or c in ('?', 'und', 'unknown', 'nan', 'none', 'mul', 'zxx'):
        return ''
    if c in _LANG:
        return _LANG[c]
    base = re.split(r'[-_]', c)[0]
    sub = re.split(r'[-_]', c)[1] if re.search(r'[-_]', c) else ''
    if sub in _LANG and base in ('de', 'deu') and sub in ('gml', 'nds'):
        return 'nds'
    return _LANG.get(base, base if len(base) in (2, 3) and base.isalpha() else '')


def edition_languages(counts: Iterable, min_known: int = 50, min_share: float = 0.8) -> dict:
    """Language of an edition: {slug: code} for the editions in which at least ``min_known`` place mentions carry a language and ``min_share`` of them the same one.
    ``counts``: iterable of (edition, lang, n). Used only to fill the language of a variant when the mention carries none."""
    per: dict = defaultdict(Counter)
    for ed, lg, n in counts:
        lg = lang_code(lg)
        if lg:
            per[ed][lg] += n
    out = {}
    for ed, c in per.items():
        tot = sum(c.values())
        lg, n = c.most_common(1)[0]
        if tot >= min_known and n / tot >= min_share:
            out[ed] = lg
    return out


# ======================================================================================================================
# 3. variants
# ======================================================================================================================
@dataclass
class Variant:
    form: str                                              # the most frequent spelling of the group
    key: str                                               # fold_key
    lang: str                                              # most frequent known language of the group ('' if none)
    n: int                                                 # mentions
    editions: dict = field(default_factory=dict)           # {edition slug: mentions}
    spellings: dict = field(default_factory=dict)          # {spelling: mentions} (the forms merged into this variant)
    langs: dict = field(default_factory=dict)              # {lang: mentions}
    n_inferred_lang: int = 0                               # mentions whose language was taken from the edition


def group_variants(counts: Iterable, edition_lang: Optional[dict] = None, drop_unusable: bool = True) -> list:
    """Variants of ONE place from its surface forms. ``counts``: iterable of (form, lang, edition, n). Forms are cleaned (``clean_form``), the unusable ones (common nouns, one
    letter ...) dropped, the forms with the same ``fold_key`` merged (they are the same name), counts added. The representative spelling is the most frequent one (ties: the one
    with diacritics, then alphabetical); the language is the most frequent known one of the merged forms, taken from the edition's language where a mention has none.
    Sorted by mentions, then form. Never merges across places: the caller passes the forms of one place."""
    groups: dict = {}
    for form, lang, ed, n in counts:
        f = clean_form(form)
        if drop_unusable and not usable_form(f):
            continue
        k = fold_key(f)
        if not k:
            continue
        g = groups.setdefault(k, Variant(form='', key=k, lang='', n=0))
        n = int(n)
        g.n += n
        g.spellings[f] = g.spellings.get(f, 0) + n
        g.editions[ed] = g.editions.get(ed, 0) + n
        lg = lang_code(lang)
        if not lg and edition_lang:
            lg = edition_lang.get(ed, '')
            if lg:
                g.n_inferred_lang += n
        if lg:
            g.langs[lg] = g.langs.get(lg, 0) + n
    out = []
    for g in groups.values():
        g.form = sorted(g.spellings.items(), key=lambda kv: (-kv[1], -sum(1 for c in kv[0] if ord(c) > 127), kv[0]))[0][0]
        g.lang = sorted(g.langs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if g.langs else ''
        out.append(g)
    out.sort(key=lambda v: (-v.n, v.form))
    return out


def top_variants(variants: Sequence, k: int = TOP_VARIANTS) -> list:
    """The ``extra.json`` form: [[variant, lang, n], ...] for the ``k`` most frequent."""
    return [[v.form, v.lang, v.n] for v in variants[:k]]


# ======================================================================================================================
# 4. the reference form
# ======================================================================================================================
def _ok_label(s) -> Optional[str]:
    if isinstance(s, str) and s.strip() and not BOT_LABEL.search(s) and GEN.place_generic(s.strip()) is None:
        return clean_form(s)[:MAX_FORM_LEN]
    return None


def register_name(entries: Iterable) -> Optional[str]:
    """The name an edition's register gives the place: the first entry (most mentioned first) that is a register entry and a name. ``entries``: iterable of (entity_key, name);
    a key starting with 'surf:' is a surface form, not a register entry. Shortened to the part before a comma or bracket where that leaves a name is NOT done here: the register's
    own string is the name ('Marignano'); a bracketed alternative stays ('Cracow (Kraków)' -> 'Cracow')."""
    for key, name in entries:
        if str(key).startswith('surf:'):
            continue
        n = _ok_label(name)
        if n:
            n = re.sub(r'\s*[\(\[].*$', '', n).strip() or n
            return n[:MAX_FORM_LEN]
    return None


def reference_form(wd_label_en: Optional[str] = None, register_names: Sequence = (), exonym: Optional[str] = None, variants: Sequence = (),
                   order: Sequence = REFERENCE_ORDER) -> tuple:
    """(form, source) of the reference form: the English Wikidata label, else the register's name, else the exonym table's English label, else the most frequent variant
    (source ``wikidata`` / ``register`` / ``exonym`` / ``surface``; (None, None) if there is nothing). A bot's label ('lsjbot Duisburg') and a common noun are never a name.
    ``register_names``: candidates in order of preference (already a name each, see ``register_name``); ``variants``: from ``group_variants`` (most frequent first)."""
    for src in order:
        if src == 'wikidata':
            f = _ok_label(wd_label_en)
        elif src == 'register':
            f = next((x for x in (_ok_label(r) for r in register_names) if x), None)
        elif src == 'exonym':
            f = _ok_label(exonym)
        elif src == 'surface':
            f = variants[0].form if variants else None
        else:
            raise ValueError(f'unknown reference source {src!r}')
        if f:
            return f, src
    return None, None


# ======================================================================================================================
# 5. the page's extra.json
# ======================================================================================================================
def mentions_frame(mentions_path, final_dir):
    """The place mentions of the atlas counts (canonical, not excluded, in period) with the place they belong to: columns coll_slug, entity_key, form (cleaned surface), lang (raw),
    global_id, globe_id, coord_source (of the local place). Same selection and the same entity key as ``cards_globe.build``."""
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from .merge import _norm
    t = pq.read_table(mentions_path, columns=['coll_slug', 'kind', 'canonical', 'excluded', 'local_key', 'surface', 'in_period', 'lang'])
    t = t.filter(pc.and_(pc.and_(pc.equal(t['kind'], 'place'), pc.equal(t['canonical'], True)), pc.equal(t['in_period'], True))).to_pandas()
    t = t[~t['excluded'].fillna(False).astype(bool)].copy()
    surf = t['surface'].fillna('')
    nn = surf.map({s: _norm(s) for s in surf.unique()})
    t['entity_key'] = np.where(t['local_key'].notna(), t['local_key'], 'surf:' + nn)
    t = t[(t['local_key'].notna() | (nn != '')).to_numpy()]
    fd = Path(final_dir)
    pl = pd.read_parquet(fd / 'places_local_geo.parquet', columns=['coll_slug', 'entity_key', 'global_id', 'coord_source'])
    pg = pd.read_parquet(fd / 'places_global.parquet', columns=['id', 'globe_id']).rename(columns={'id': 'global_id'})
    pl = pl.merge(pg, on='global_id', how='left')
    t = t.merge(pl, on=['coll_slug', 'entity_key'], how='left')
    t['form'] = t['surface'].map(clean_form)
    return t[['coll_slug', 'entity_key', 'form', 'lang', 'global_id', 'globe_id', 'coord_source']]


def name_based_count(coord_sources: Iterable) -> int:
    """Mentions located by a name: how many of these (one per mention) come from a local place whose ``coord_source`` is a name-based path."""
    return sum(1 for c in coord_sources if c in NAME_BASED_SOURCES)


def variant_table(m, key: str, edition_lang: Optional[dict] = None) -> dict:
    """{group id: [Variant]} for a mentions frame grouped by the column ``key`` ('global_id' or 'globe_id')."""
    g = m[m[key].notna()].groupby([key, 'form', 'lang', 'coll_slug'], dropna=False).size().reset_index(name='n')
    out: dict = {}
    for k, sub in g.groupby(key, sort=False):
        out[k] = group_variants(zip(sub['form'], sub['lang'], sub['coll_slug'], sub['n']), edition_lang)
    return out


def reference_forms(final_dir, wd_labels: Optional[dict] = None, variants_by_global: Optional[dict] = None, order: Sequence = REFERENCE_ORDER) -> dict:
    """{global_id: (form, source)} for every global place of ``places_global.parquet``: Wikidata English label (``label_en`` of the table when the place has a qid, or ``wd_labels``
    {qid: label}), register name from the locals (most mentioned register entry), exonym table label (``label_en`` of an exonym place), else the most frequent variant."""
    import pandas as pd
    from .cards_globe import DOUBTFUL_ITEMS, DUPLICATE_ITEMS
    fd = Path(final_dir)
    pg = pd.read_parquet(fd / 'places_global.parquet', columns=['id', 'qid', 'label_en', 'name'])
    pl = pd.read_parquet(fd / 'places_local_geo.parquet', columns=['global_id', 'entity_key', 'name', 'n_in_period'])
    pl = pl.sort_values('n_in_period', ascending=False)
    regs = {gid: list(zip(sub['entity_key'], sub['name'])) for gid, sub in pl.groupby('global_id', sort=False)}
    out = {}
    for gid, q, lab, nm in zip(pg['id'], pg['qid'], pg['label_en'], pg['name']):
        wd = exo = None
        if isinstance(q, str) and q and q not in DOUBTFUL_ITEMS:
            wd = (wd_labels or {}).get(DUPLICATE_ITEMS.get(q, q)) or (lab if isinstance(lab, str) else None)
        elif isinstance(lab, str):
            exo = lab
        reg = register_name(regs.get(gid, ()))
        out[gid] = reference_form(wd, [reg] if reg else [], exo, (variants_by_global or {}).get(gid, ()), order)
    return out


def build_extra(final_dir, mentions_path, out_path=None, core_path=None, wd_entities=None, gn_index=None, orbis_tsv=None, casebooks_tsv=None, globals_out=None,
                reviewed=None) -> dict:
    """``extra.json`` for a globe build: {"types": {code: [label_en, parent_code, coarse_kind]}, "rows": {"<row index>": [type_code, [[variant, lang, n], ... top 10 by n],
    name_based_mentions]}}. The row index is the position in ``globe_places.json`` (= ``core.json["rows"]``); ``core.json`` (``core_path``), if given, supplies the kinds the page uses
    (so the fine type never contradicts the page) and is checked for the number of rows. ``globals_out``: a TSV with one line per global place (type, reference form and its source,
    folded key, number of variants, top variants). Read-only on the tables; run it on the Mac (a minute) or on Slurm."""
    import pandas as pd
    from . import placetypes as PT
    fd = Path(final_dir)
    items = json.loads((fd / 'globe_places.json').read_text('utf-8'))['places']
    row_of = {p['id']: i for i, p in enumerate(items)}
    kinds = None
    if core_path:
        core = json.loads(Path(core_path).read_text('utf-8'))
        if len(core['rows']) != len(items):
            raise ValueError(f'core.json has {len(core["rows"])} rows, globe_places.json {len(items)}')
        kinds = {p['id']: core['rows'][i][11] for i, p in enumerate(items)}
    df = PT.local_evidence_frame(fd, wd_entities, gn_index, orbis_tsv, casebooks_tsv, point_kinds=kinds)
    df = PT.classify_frame(df, reviewed)
    ptype = df[df['point'].notna()].drop_duplicates('point').set_index('point')['type']
    gtype = df.drop_duplicates('global_id').set_index('global_id')['type']
    m = mentions_frame(mentions_path, fd)
    edl = edition_languages(zip(m['coll_slug'], m['lang'], [1] * len(m)))
    vb_point = variant_table(m, 'globe_id', edl)
    nb = m[m['globe_id'].notna()].groupby('globe_id')['coord_source'].apply(name_based_count)
    rows = {}
    for gid, i in row_of.items():
        vs = vb_point.get(gid, [])
        rows[str(i)] = [ptype.get(gid, 'other'), top_variants(vs), int(nb.get(gid, 0))]
    extra = {'types': PT.types_json(), 'rows': rows}
    if out_path:
        Path(out_path).write_text(json.dumps(extra, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    if globals_out:
        vb_g = variant_table(m, 'global_id', edl)
        refs = reference_forms(fd, None, vb_g)
        recs = []
        for gid, (form, src) in refs.items():
            vs = vb_g.get(gid, [])
            recs.append(dict(global_id=gid, type=gtype.get(gid, 'other'), reference_form=form, reference_source=src, folded_key=fold_key(form or ''), n_variants=len(vs),
                             variants=json.dumps(top_variants(vs), ensure_ascii=False)))
        pd.DataFrame(recs).to_csv(globals_out, sep='\t', index=False)
    return extra


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='python -m ratlas.name_variants')
    sub = ap.add_subparsers(dest='cmd', required=True)
    ex = sub.add_parser('extra', help='build extra.json (types, top variants, name-based mentions per globe row)')
    ex.add_argument('--final', type=Path, required=True)
    ex.add_argument('--mentions', type=Path, required=True)
    ex.add_argument('--core', type=Path)
    ex.add_argument('--wd-entities', type=Path)
    ex.add_argument('--gn-index', type=Path)
    ex.add_argument('--orbis', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'orbis_latinus.tsv')
    ex.add_argument('--casebooks', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'casebooks_place_map.tsv')
    ex.add_argument('--reviewed', type=Path)
    ex.add_argument('--out', type=Path, required=True)
    ex.add_argument('--globals-out', type=Path)
    a = ap.parse_args(argv)
    e = build_extra(a.final, a.mentions, a.out, a.core, a.wd_entities, a.gn_index, a.orbis, a.casebooks, a.globals_out, a.reviewed)
    print(f'{len(e["rows"])} rows, {len(e["types"])} types -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
