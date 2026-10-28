"""The nine checks of the audit of 2026-10-16, as blocking checks on the v2 tables (rapport, section 5, phase 4).

    python -m ratlas audit v2 --all out_v2/all --final out_v2/reconcile/final --rdir reconcile_v2 --raw raw --out audit_v2 \\
        [--geonames-zip allCountries.zip] [--verify-hashes] [--checks 1,2,...]

On Hypatia (Slurm, never on the login node): ``slurm/audit_v2.sbatch``. The pytest file ``tests/test_audit_v2.py`` tests the logic of every
check on tiny synthetic tables (always) and runs the checks on the real tables (``RATLAS_AUDIT_V2=1``; the Slurm job sets it).

======  =====================================================================================================================
check   rule (a failure is a row of ``<out>/check<N>_<name>.tsv``; the exit code is 1 when a blocking check has a failure)
======  =====================================================================================================================
1       No confirmed name / authority MISMATCH among the accepted ids. Yardstick: the audit's own matchers, frozen in
        ``audit_namecmp`` (persons) and ``audit_compat`` (places), not ``namegate``. An accepted pair whose names the audit's
        matcher calls a mismatch fails unless a second independent id agrees. Coordinates alone do not excuse it: a place accepted
        because a point lies near the item needs a GeoNames feature that carries the entry's name and one of the item's names.
        No pair the audit confirmed as wrong (``data/audit_confirmed_wrong.tsv``) is accepted again. A pair a person confirmed by
        reading the sources (reason ``manual_review``, ``data/id_overrides.tsv``) stands.
2       No non-human QID on a person unless the published class (``person_class_final``) allows it (mythological, biblical,
        dynasty, character, people); no disambiguation page / given name / list / office behind a person QID; a metonym has no id.
3       No person (accepted QID) born after the latest document citing them (Wikidata birth, precision-aware, 5 years of slack).
4       No point farther than 50 km from every GeoNames feature that carries the place's name (names folded phonetically on both
        sides, qualifier words dropped), unless another source vouches for it (an accepted authority's own point, a register point
        within 50 km of it, the curated exonym table, an area) or the entity is on the allow list. Editorial points (the editors'
        coordinates alone, or a copy of one) are listed in ``check4_review_editorial_points.tsv``, not failed.
5       No capital-for-country point: a country or first-order region (register type, GeoNames code, Wikidata class) is an area;
        a name only countries / first-order regions carry that sits on a capital or a big seat, away from the polity's own
        point, fails; polity names placed elsewhere as coarse points are listed in ``check5_review_polity_points.tsv``.
6       Nothing of the commentary / apparatus / footnote / regest / repository zones, no character, celestial body, placeholder,
        provenance or quotation source in the default counts; the tables' counts are the sums of the default mentions. Pronouns
        tagged by the editors are counted and reported (policy), not failed.
7       No duplicated mention: Casebooks has one person / org mention per (case, entity). Repeated header rows of other
        editions are information (a multi-letter file repeats its senders).
8       No midpoint date for a range wider than 20 years; ``doc_year`` stays inside its range; ``in_period`` follows the range.
9       Stale sources: every file of every edition comes from a harvest folder with a ``SOURCES.tsv`` (and a manifest that holds)
        or is listed in ``data/legacy_sources.tsv``; an edition that changed upstream after its harvest is never legacy.
======  =====================================================================================================================
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import audit_compat as AC
from . import audit_namecmp as AN

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'

PERIOD = (1450, 1700)
GN_THRESHOLD_KM = 50.0
GN_CLASSES = frozenset('PATHLS')
CHRONO_SLACK = 5
WIDE_RANGE = 20
HEADER_DUP_SHARE = 0.03

# classes of the thing mentioned that may sit on a non-human item
NONHUMAN_OK = {'mythological', 'biblical', 'dynasty', 'character', 'people'}
EXCLUDED_ZONES = ('editorial_commentary', 'apparatus', 'footnote', 'regest', 'repository')
EXCLUDED_CLASSES = ('character', 'celestial')
EXCLUDED_ROLES = ('provenance', 'quote_source')
BAD_ITEM_CLASS = re.compile(r'disambiguation|given name|family name|surname|list of|wikimedia|position|office|occupation|'
                            r'human settlement', re.I)
COUNTRY_CLASS = re.compile(r'\b(sovereign state|country|historical country|former country|kingdom|empire|duchy|principality|'
                           r'federated state|free state|state of germany|voivodeship|autonomous community|chartered community|'
                           r'republic|grand duchy|electorate|archduchy|protectorate|dominion|confederation|viceroyalty|satrapy)\b', re.I)
COUNTRY_FCODES = ('PCLI', 'PCLD', 'PCLF', 'PCLS', 'PCLIX', 'PCL', 'PCLH', 'ADM1', 'TERR', 'CONT')
COUNTRY_NOTE = {'country', 'coun', 'countr', 'region', 'kingdom', 'continent'}
PRONOUNS = set('''ich du er sie es wir ihr mich mir dich dir ihn ihm ihnen uns euch sich man he she we you they me him them us
io tu lui lei noi voi loro egli ella je il elle nous vous ils elles moi toi eux ego nos vos ille illa ipse ipsa qui quis quae
yo él ella nosotros eu ele ela ik hij zij wij jij gij'''.split())


def _t(msg):
    print(f'[audit_v2 {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    n: int
    name: str
    title: str
    n_checked: int = 0
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    allowed: int = 0
    info: dict = field(default_factory=dict)
    skipped: str = ''
    blocking: bool = True
    extra: dict = field(default_factory=dict)         # name -> DataFrame written beside the failures (non-blocking review lists)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def summary(self) -> dict:
        return {'check': self.n, 'name': self.name, 'title': self.title, 'checked': int(self.n_checked), 'failures': int(len(self.failures)),
                'allowed': int(self.allowed), 'blocking': self.blocking, 'skipped': self.skipped, 'info': self.info}


def load_allow(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or DATA / 'audit_v2_allow.tsv'
    if not Path(p).exists():
        return pd.DataFrame(columns=['check', 'coll_slug', 'key', 'reason'])
    # comment lines start with '#' (a '#' inside a key is part of it: religionsfrieden's keys have one)
    lines = [l for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    return pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', dtype=str, keep_default_na=False)


def apply_allow(res: Result, allow: pd.DataFrame, key_cols=('coll_slug', 'key')) -> Result:
    """Move the failures that are on the allow list (check, coll_slug, key) out of ``failures``."""
    f = res.failures
    if f.empty or allow.empty or 'key' not in f:
        return res
    a = allow[allow['check'].astype(str) == str(res.n)]
    if a.empty:
        return res
    ok = set(zip(a['coll_slug'], a['key']))
    hit = np.array([(c, k) in ok for c, k in zip(f['coll_slug'], f['key'])])
    res.allowed += int(hit.sum())
    res.failures = f[~hit].reset_index(drop=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------------------------------------------------
class Tables:
    """Lazy loader. ``all``: out_v2/all (mentions, entities, *_local); ``final``: the reconcile tables; ``rdir``: reconcile_v2 inputs."""

    MENTION_COLS = ['coll_slug', 'source_path', 'doc_id', 'doc_year', 'doc_year_from', 'doc_year_to', 'date_source', 'in_period', 'kind',
                    'tag', 'role', 'zone', 'person_class', 'excluded', 'excluded_reason', 'surface', 'local_key', 'canonical']

    def __init__(self, all_dir, final_dir, rdir=None, raw=None, out=None, tgn_dir=None):
        self.tgn_dir = Path(tgn_dir) if tgn_dir else None
        self.all = Path(all_dir)
        self.final = Path(final_dir)
        self.rdir = Path(rdir) if rdir else None
        self.raw = Path(raw) if raw else None
        self.out = Path(out) if out else None
        self._c: dict = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def mentions(self) -> pd.DataFrame:
        return self._get('mentions', lambda: pq.read_table(self.all / 'mentions.parquet', columns=self.MENTION_COLS).to_pandas())

    def final_t(self, name, columns=None) -> pd.DataFrame:
        return self._get((name, tuple(columns or ())), lambda: pq.read_table(self.final / f'{name}.parquet', columns=columns).to_pandas())

    def all_t(self, name, columns=None) -> pd.DataFrame:
        return self._get(('all' + name, tuple(columns or ())), lambda: pq.read_table(self.all / f'{name}.parquet', columns=columns).to_pandas())

    def auth(self):
        def load():
            from .reconcile_gates import AuthData
            return AuthData(self.rdir, self.tgn_dir)
        return self._get('auth', load)


def _norm_name(s) -> str:
    s = unicodedata.normalize('NFKD', str(s or '').casefold())
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', ''.join(c for c in s if not unicodedata.combining(c))).split())


# ---------------------------------------------------------------------------------------------------------------------
# check 1: names against authorities
# ---------------------------------------------------------------------------------------------------------------------
def person_name_verdicts(pairs: pd.DataFrame, labels: Callable[[pd.Series], list], names: dict, weak: set) -> pd.DataFrame:
    """The frozen audit matcher on accepted person pairs. ``labels(row)`` -> the authority's names, ``names[(coll, key)]`` -> entry names.
    Returns the pairs with ``verdict`` (ok | ok_weak | given_only | mismatch | no_label | no_name)."""
    out = []
    for r in pairs.itertuples(index=False):
        lab = labels(r)
        ent = names.get((r.coll_slug, r.entity_key), [])
        st = AN.compare(ent, lab, weak)[0] if lab else 'no_label'
        out.append(st)
    p = pairs.copy()
    p['verdict'] = out
    return p


def place_name_verdicts(pairs: pd.DataFrame, labels: Callable[[pd.Series], list], names: dict) -> pd.DataFrame:
    out = []
    for r in pairs.itertuples(index=False):
        lab = labels(r)
        nm = names.get((r.coll_slug, r.entity_key))
        ek = AC.name_keys(nm[0], nm[1]) if nm else set()
        ak = AC.auth_keys(lab) if lab else set()
        out.append(AC.relation(ek, ak)[0])
    p = pairs.copy()
    p['verdict'] = out
    return p


COORD_EVIDENCE = ('near_register', 'near_geonames_alias')


def check_names(pairs: pd.DataFrame, person_v: pd.DataFrame, place_v: pd.DataFrame) -> pd.DataFrame:
    """Failures of check 1 from the verdicts: the audit matcher says mismatch / none, no second id, no coordinate evidence."""
    rows = []
    pv = person_v[(person_v['verdict'] == 'mismatch') & (person_v['second_id'].fillna('') == '')]
    for r in pv.itertuples(index=False):
        rows.append(dict(kind='person', coll_slug=r.coll_slug, key=r.entity_key, name=r.name, scheme=r.scheme, id=r.id, qid=r.qid,
                         label=r.label, verdict=r.verdict, gate_level=r.level, n_mentions=r.n_mentions))
    qv = place_v[place_v['verdict'].astype(str).str.startswith('none') & (place_v['second_id'].fillna('') == '')
                 & ~place_v['level'].isin(COORD_EVIDENCE)]          # coordinate-only acceptances: shared-feature test in run_check1
    for r in qv.itertuples(index=False):
        rows.append(dict(kind='place', coll_slug=r.coll_slug, key=r.entity_key, name=r.name, scheme=r.scheme, id=r.id, qid=r.qid,
                         label=r.label, verdict=r.verdict, gate_level=r.level, n_mentions=r.n_mentions))
    return pd.DataFrame(rows, columns=['kind', 'coll_slug', 'key', 'name', 'scheme', 'id', 'qid', 'label', 'verdict', 'gate_level', 'n_mentions'])


def confirmed_wrong_back(pairs: pd.DataFrame, confirmed: pd.DataFrame, pl: pd.DataFrame, pg_ent: pd.DataFrame) -> pd.DataFrame:
    """Pairs the audit confirmed as wrong that v2 accepts again.

    ``confirmed``: kind, coll_slug, entity_key, entry_name, scheme, id, qid, bad_qid, audit_check. ``bad_qid`` set: the id may be right but the
    Wikidata item it led to is wrong (Balingen's GeoNames id sits on the item Bisingen): back when an accepted pair of the entity has that item.
    Otherwise the id itself is wrong: back when an accepted pair has the same (coll_slug, scheme, id). Either way the entity is the same key or
    the same folded name (the keys of a re-harvested edition may have changed). A person row is back when the entity (same key) has the bad
    QID as its accepted QID with default mentions."""
    acc = pairs[pairs['status'] == 'accepted']
    rows = []
    accp = acc[acc['kind'] == 'place']
    idx = {}
    for r in accp.itertuples(index=False):
        idx.setdefault((r.coll_slug, r.scheme, str(r.id)), []).append((r.entity_key, _norm_name(r.name), r.qid if isinstance(r.qid, str) else ''))
    for r in confirmed[confirmed['kind'] == 'place'].itertuples(index=False):
        for ek, nn, qid in idx.get((r.coll_slug, r.scheme, str(r.id)), []):
            if r.bad_qid and qid != r.bad_qid:
                continue
            if ek == r.entity_key or nn == _norm_name(r.entry_name):
                rows.append(dict(kind='place', coll_slug=r.coll_slug, key=ek, name=r.entry_name, scheme=r.scheme, id=r.id,
                                 audit_check=r.audit_check, note='accepted again' + (f' (item {qid})' if r.bad_qid else '')))
    if not pg_ent.empty:
        q = {(c, k): (qid, n) for c, k, qid, n in zip(pg_ent['coll_slug'], pg_ent['entity_key'], pg_ent['qid'], pg_ent['n_mentions'])}
        # a person confirmed by hand after the audit (reason manual_review: a modern scholar cited in the editor's notes) stands
        manual = {(r.coll_slug, r.entity_key, r.qid) for r in acc[(acc['kind'] == 'person') & (acc['reason'] == 'manual_review')].itertuples(index=False)}
        for r in confirmed[confirmed['kind'] == 'person'].itertuples(index=False):
            v = q.get((r.coll_slug, r.entity_key))
            if (r.coll_slug, r.entity_key, r.bad_qid or r.qid) in manual:
                continue
            if v and v[0] == (r.bad_qid or r.qid) and v[1] and v[1] > 0:
                rows.append(dict(kind='person', coll_slug=r.coll_slug, key=r.entity_key, name=r.entry_name, scheme='wikidata', id=r.qid,
                                 audit_check=r.audit_check, note='accepted again'))
    return pd.DataFrame(rows, columns=['kind', 'coll_slug', 'key', 'name', 'scheme', 'id', 'audit_check', 'note'])


def shared_feature_evidence(items: pd.DataFrame, zip_path: Path, near_km: float = 10.0, log=_t) -> pd.Series:
    """For place pairs accepted on coordinates alone (the entry's name is unrelated to the item's labels): is there a GeoNames feature that
    carries the entry's name within ``near_km`` of the authority's point AND one of the item's names (Venedig and Venezia are names of one
    feature; Halle and Altstadt are not)? ``items``: entry_keys (set), label_keys (set), lat, lon. -> bool Series (index of items)."""
    want: dict = {}
    for i, ek in items['entry_keys'].items():
        for k in ek:
            want.setdefault(k, []).append(i)
    nrm = lru_cache(maxsize=2_000_000)(lambda x: fkey(AC.nrm(x)))
    ok = pd.Series(False, index=items.index)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z, z.open('allCountries.txt') as fh:
        for n, line in enumerate(io.TextIOWrapper(fh, encoding='utf-8', newline='')):
            f = line.split('\t', 9)
            if len(f) < 8 or f[6] not in GN_CLASSES:
                continue
            names = {nrm(f[1]), nrm(f[2])} | {nrm(a) for a in f[3].split(',') if a}
            hit = names & want.keys()
            if not hit:
                continue
            la, lo = float(f[4]), float(f[5])
            for k in hit:
                for i in want[k]:
                    if ok[i]:
                        continue
                    if haversine_km(la, lo, items.at[i, 'lat'], items.at[i, 'lon']) <= near_km and names & items.at[i, 'label_keys']:
                        ok[i] = True
            if n % 4_000_000 == 0:
                log(f'shared-feature scan {n} lines ({time.time() - t0:.0f}s)')
    return ok


def run_check1(T: Tables, zip_path: Optional[Path] = None) -> Result:
    res = Result(1, 'names', 'No confirmed name / authority mismatch among the accepted ids')
    auth = T.auth()
    pairs = T.final_t('gate_pairs')
    acc = pairs[pairs['status'] == 'accepted']
    if auth.tgn_dir is not None:        # v2.1: the accepted TGN ids are read in the local Getty index with the audit's own (frozen) matcher
        auth.load_tgn(acc.loc[acc['scheme'] == 'tgn', 'id'].unique())
    pl_local = T.all_t('places_local', ['coll_slug', 'entity_key', 'name', 'variants'])
    pe_local = T.all_t('persons_local', ['coll_slug', 'entity_key', 'name', 'variants'])
    pn = {(c, k): (n, list(v) if v is not None else []) for c, k, n, v in
          zip(pl_local['coll_slug'], pl_local['entity_key'], pl_local['name'], pl_local['variants'])}
    en = {}
    for c, k, n, v in zip(pe_local['coll_slug'], pe_local['entity_key'], pe_local['name'], pe_local['variants']):
        names = ([n] if isinstance(n, str) else []) + [x for x in (list(v) if v is not None else []) if isinstance(x, str)]
        en[(c, k)] = list(dict.fromkeys(names))[:14]
    weak, _ = AN.build_weak(auth.person_labels(), 40)

    def labels_of(r):
        if r.src == 'wd':
            e = auth.wd.get(r.qid) if isinstance(r.qid, str) else None
            return list(e['names']) if e and not e.get('missing') else []
        if r.src == 'gnd':
            e = auth.gnd.get(str(r.id))
            return list(e['names']) if e else []
        if r.src == 'geonames':
            e = auth.gn.get(str(r.id))
            return sorted(e['names']) if e else []
        if r.src == 'tgn':
            e = auth.tgn.get(str(r.id))
            return list(e['names']) if e else []
        return [r.label] if isinstance(r.label, str) else []

    # a pair that a person confirmed by reading the sources (reason manual_review, data/id_overrides.tsv) stands: the override is the exception
    reviewed = acc['reason'] == 'manual_review'
    res.info['accepted_by_manual_review (not judged by the matchers)'] = int(reviewed.sum())
    pp = acc[(acc['kind'] == 'person') & acc['src'].isin(['wd', 'gnd']) & (acc['level'] != 'title') & ~reviewed]
    qq = acc[(acc['kind'] == 'place') & acc['src'].isin(['wd', 'gnd', 'geonames', 'tgn']) & (acc['level'] != 'title') & ~reviewed]
    _t(f'check 1: {len(pp)} person pairs, {len(qq)} place pairs')
    pv = person_name_verdicts(pp, labels_of, en, weak)
    qv = place_name_verdicts(qq, labels_of, pn)
    res.n_checked = len(pp) + len(qq)
    f = check_names(acc, pv, qv)
    # coordinate-only acceptances (name unrelated to every label, no second id): a GeoNames feature must carry both names
    qn = qv[qv['verdict'].astype(str).str.startswith('none') & (qv['second_id'].fillna('') == '') & qv['level'].isin(COORD_EVIDENCE)]
    res.info['coordinate_only_place_pairs'] = int(len(qn))
    if len(qn):
        if zip_path is not None and Path(zip_path).exists():
            its = pd.DataFrame({'entry_keys': [place_keys(*pn[(c, k)]) if (c, k) in pn else set() for c, k in zip(qn['coll_slug'], qn['entity_key'])],
                                'label_keys': [{fkey(k) for k in AC.auth_keys(labels_of(r))} for r in qn.itertuples(index=False)],
                                'lat': [auth.wd.get(q, {}).get('lat') if isinstance(q, str) else (auth.tgn.get(str(i)) or {}).get('lat')
                                        for q, i in zip(qn['qid'], qn['id'])],
                                'lon': [auth.wd.get(q, {}).get('lon') if isinstance(q, str) else (auth.tgn.get(str(i)) or {}).get('lon')
                                        for q, i in zip(qn['qid'], qn['id'])]}, index=qn.index)
            has_pt = its['lat'].notna()
            shared = pd.Series(False, index=its.index)
            shared[has_pt] = shared_feature_evidence(its[has_pt], Path(zip_path))
            bad = qn[~shared]
            res.info['coordinate_only_with_shared_feature'] = int(shared.sum())
            f = pd.concat([f, pd.DataFrame({'kind': 'place', 'coll_slug': bad['coll_slug'], 'key': bad['entity_key'], 'name': bad['name'],
                                            'scheme': bad['scheme'], 'id': bad['id'], 'qid': bad['qid'], 'label': bad['label'],
                                            'verdict': 'coordinates only', 'gate_level': bad['level'], 'n_mentions': bad['n_mentions']})],
                          ignore_index=True)
        else:
            res.info['coordinate_only_place_pairs_unchecked'] = 'no GeoNames dump: pass --geonames-zip'
    # the audit's confirmed errors must not come back
    conf_path = DATA / 'audit_confirmed_wrong.tsv'
    back = pd.DataFrame()
    if conf_path.exists():
        conf = pd.read_csv(conf_path, sep='\t', dtype=str, keep_default_na=False)
        loc = T.final_t('persons_local_global', ['coll_slug', 'entity_key', 'qid', 'n_mentions', 'scope'])
        loc = loc[(loc['scope'] != 'excluded') & loc['qid'].notna()]
        back = confirmed_wrong_back(pairs, conf, T.final_t('places_local_geo', ['coll_slug', 'entity_key']), loc)
        res.info['audit_confirmed_rows'] = int(len(conf))
    res.info['given_name_only_persons'] = int((pv['verdict'] == 'given_only').sum())
    res.info['person_pairs_mismatch_with_second_id'] = int(((pv['verdict'] == 'mismatch') & (pv['second_id'].fillna('') != '')).sum())
    res.info['audit_pairs_accepted_again'] = int(len(back))
    f = pd.concat([f, back.assign(verdict='audit_confirmed_wrong')], ignore_index=True) if len(back) else f
    res.failures = f
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 2: classes of the person QIDs
# ---------------------------------------------------------------------------------------------------------------------
def nonhuman_failures(loc: pd.DataFrame, items: dict, class_labels: dict) -> pd.DataFrame:
    """``loc``: coll_slug, entity_key, name, qid, person_class (the FINAL class: the global table's ``person_class_final``). ``items``: qid -> list
    of P31 classes (None = unknown). A person with a QID whose item is not a human needs a final class of NONHUMAN_OK; a QID of a disambiguation
    page, a given name, a list, an office ... is never allowed (whatever the class); a metonym carries no QID."""
    rows = []
    for r in loc.itertuples(index=False):
        if not isinstance(r.qid, str) or not r.qid:
            continue
        pc = r.person_class if isinstance(r.person_class, str) else 'human'
        cl = items.get(r.qid)
        if pc == 'metonym':
            rows.append(dict(coll_slug=r.coll_slug, key=r.entity_key, name=r.name, qid=r.qid, person_class=pc, why='metonym with an id'))
            continue
        if cl is None:
            continue
        labs = [class_labels.get(c, '') for c in cl]
        if any(BAD_ITEM_CLASS.search(x) for x in labs if x) and 'Q5' not in cl:
            rows.append(dict(coll_slug=r.coll_slug, key=r.entity_key, name=r.name, qid=r.qid, person_class=pc,
                             why='bad class: ' + ' | '.join(x for x in labs if x)[:120]))
            continue
        if 'Q5' not in cl and pc not in NONHUMAN_OK:
            rows.append(dict(coll_slug=r.coll_slug, key=r.entity_key, name=r.name, qid=r.qid, person_class=pc,
                             why='not human: ' + ' | '.join(x for x in labs if x)[:120]))
    return pd.DataFrame(rows, columns=['coll_slug', 'key', 'name', 'qid', 'person_class', 'why'])


def run_check2(T: Tables) -> Result:
    res = Result(2, 'person_classes', 'No non-human QID on persons unless the class allows it')
    auth = T.auth()
    loc = T.final_t('persons_local_global', ['coll_slug', 'entity_key', 'name', 'qid', 'person_class', 'global_id', 'scope', 'n_mentions'])
    loc = loc[loc['qid'].notna() & (loc['scope'] != 'excluded')]
    pg = T.final_t('persons_global', ['id', 'qid', 'wd_human', 'person_class_final', 'name', 'n_mentions'])
    final = dict(zip(pg['id'], pg['person_class_final']))
    # the class a person is published with: the global table's final class (it reads the item's own category), else the local one
    loc = loc.assign(person_class=[final.get(g) or c for g, c in zip(loc['global_id'], loc['person_class'])])
    items = {q: (e['classes'] if not e.get('missing') else None) for q, e in auth.wd.items()}
    res.n_checked = len(loc)
    f = nonhuman_failures(loc, items, auth.classlab)
    g = pg[(pg['wd_human'].astype(str) == 'False') & ~pg['person_class_final'].isin(NONHUMAN_OK) & (pg['n_mentions'] > 0)]
    res.info['global_persons_with_qid'] = int(pg['qid'].notna().sum())
    res.info['global_nonhuman_qid'] = int((pg['wd_human'].astype(str) == 'False').sum())
    res.info['global_nonhuman_by_final_class'] = {k: int(v) for k, v in pg[pg['wd_human'].astype(str) == 'False']['person_class_final'].value_counts().items()}
    if len(g):
        f = pd.concat([f, pd.DataFrame({'coll_slug': 'GLOBAL', 'key': g['id'], 'name': g['name'], 'qid': g['qid'],
                                        'person_class': g['person_class_final'], 'why': 'global: non-human QID, class human'})], ignore_index=True)
    res.failures = f
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 3: born after the documents
# ---------------------------------------------------------------------------------------------------------------------
def born_after_failures(persons: pd.DataFrame, ymax: pd.Series, slack: int = CHRONO_SLACK) -> pd.DataFrame:
    """``persons``: id, qid, name, label, birth, birth_precision, person_class_final; ``ymax``: id -> latest document year (upper bound)."""
    p = persons[persons['qid'].notna() & persons['birth'].notna()].copy()
    p['ymax'] = p['id'].map(ymax)
    p = p[p['ymax'].notna()]
    prec = p['birth_precision'].astype('float').fillna(9)
    # the earliest year the birth can be: a century / decade precision is a range, not a year
    lo = p['birth'] - np.where(prec <= 7, 99, np.where(prec == 8, 9, 0))
    p['gap'] = lo - p['ymax']
    f = p[p['gap'] > slack]
    return pd.DataFrame({'coll_slug': 'GLOBAL', 'key': f['id'], 'name': f['name'], 'label': f['label'], 'qid': f['qid'], 'birth': f['birth'],
                         'last_document': f['ymax'], 'gap_years': f['gap']}).reset_index(drop=True)


def latest_document_year(m: pd.DataFrame, loc: pd.DataFrame) -> pd.Series:
    """Global person id -> latest document year over the default mentions of all its locals (the upper end of the date range)."""
    d = m[(m['kind'] == 'person') & ~m['excluded'].fillna(False).astype(bool) & m['canonical'].fillna(True).astype(bool)]
    y = d['doc_year_to'].fillna(d['doc_year'])
    g = pd.DataFrame({'coll_slug': d['coll_slug'].to_numpy(), 'entity_key': d['local_key'].to_numpy(), 'y': y.to_numpy()})
    g = g.dropna().groupby(['coll_slug', 'entity_key'], as_index=False)['y'].max()
    j = loc[['coll_slug', 'entity_key', 'global_id']].merge(g, on=['coll_slug', 'entity_key'])
    return j.groupby('global_id')['y'].max()


def run_check3(T: Tables) -> Result:
    res = Result(3, 'born_after', 'No person is born after the latest document that cites them')
    loc = T.final_t('persons_local_global', ['coll_slug', 'entity_key', 'global_id'])
    ymax = latest_document_year(T.mentions(), loc)
    pg = T.final_t('persons_global', ['id', 'qid', 'name', 'label', 'birth', 'birth_precision', 'person_class_final'])
    res.n_checked = int((pg['qid'].notna() & pg['birth'].notna() & pg['id'].isin(ymax.index)).sum())
    f = born_after_failures(pg, ymax)
    # a person confirmed by hand (reason manual_review: a later scholar cited in the editorial apparatus, the register's own dates agree with
    # Wikidata) stands
    gp = T.final_t('gate_pairs', ['kind', 'status', 'reason', 'detail', 'qid'])
    mr = set(gp.loc[(gp['kind'] == 'person') & (gp['status'] == 'accepted') & (gp['reason'] == 'manual_review')
                    & gp['detail'].fillna('').str.contains('chronology'), 'qid'].dropna())
    keep = ~f['qid'].isin(mr)
    res.info['born_after_confirmed_by_manual_review'] = int((~keep).sum())
    res.failures = f[keep].reset_index(drop=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 4: points far from every same-name GeoNames feature
# ---------------------------------------------------------------------------------------------------------------------
def haversine_min(lat: float, lon: float, pts: np.ndarray):
    p1, p2 = np.radians(lat), np.radians(pts[:, 0])
    x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(pts[:, 1] - lon) / 2) ** 2
    d = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))
    i = int(d.argmin())
    return float(d[i]), i


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


POLITY_FCODES = ('PCLI', 'PCLD', 'PCLF', 'PCLS', 'PCLIX', 'PCL', 'PCLH', 'ADM1', 'TERR', 'CONT')
SEAT_FCODES = ('PPLC', 'PPLA', 'PPLA2')       # a capital, or an administrative seat of at least SEAT_POP inhabitants
SEAT_POP = 100_000
SEAT_KM = 5.0
POLITY_OWN_KM = 25.0                          # the polity's own GeoNames point: a point there is the polity's, not a city's


def fkey(k: str) -> str:
    """The audit's phonetic fold of a normalised name key: Konstantinopel and Constantinopel are one name, as are Köln and Cölln."""
    return AC.fold(k)


# words that qualify a toponym without being part of its name: 'Wing. parish' is Wing, 'Stift Schildesche' is Schildesche, 'County of Middlesex'
QUALIFIERS = frozenset('parish county of kloster stift schloss castle church kirche abbey abtei bistum hochstift herzogtum furstentum grafschaft '
                       'konigreich kapitanat insel isle island fluss river lake see st saint sankt san santa s the de la le'.split())


def strip_qualifiers(k: str) -> str:
    toks = [t for t in k.split() if t not in QUALIFIERS]
    return ' '.join(toks)


def place_keys(name, variants) -> set:
    """The folded name keys of a place: the audit's ``name_keys`` and the same keys without qualifier words (>= 4 letters left)."""
    ks = AC.name_keys(name, variants)
    ks = ks | {q for q in (strip_qualifiers(k) for k in ks) if len(q) >= 4}
    return {f for f in (fkey(k) for k in ks) if len(f) >= 2}


def geonames_distances(places: pd.DataFrame, zip_path: Path, log=_t) -> pd.DataFrame:
    """For each placed place the distance from its final point to the nearest GeoNames feature (classes P A T H L S) that carries one of its
    names (name, variants, the part before a comma, inside / outside brackets: ``audit_compat.name_keys``, then the audit's phonetic fold on
    both sides). d_km = -1: no feature of that name. Also ``n_polity`` (same-name features that are countries / first-order regions) and
    ``n_settlement`` (same-name populated places): a name that only a country or a region carries is not a town. One pass over the dump."""
    keys = [place_keys(n, v) for n, v in zip(places['name'], places['variants'])]
    primary = [{k for k in (fkey(AC.nrm(n)), fkey(AC.nrm(re.split(r'[,(]', n)[0]))) if len(k) >= 4} for n in places['name']]
    target = set().union(*keys) if keys else set()
    nrm = lru_cache(maxsize=2_000_000)(lambda x: fkey(AC.nrm(x)))
    hits: dict = {}
    seats = []                    # capitals and administrative seats: a polity placed on one of them is a capital for a country
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z, z.open('allCountries.txt') as fh:
        for n, line in enumerate(io.TextIOWrapper(fh, encoding='utf-8', newline='')):
            f = line.split('\t', 15)
            if len(f) < 15 or f[6] not in GN_CLASSES:
                continue
            if f[7] == 'PPLC' or (f[7] in SEAT_FCODES and f[14].isdigit() and int(f[14]) >= SEAT_POP):
                seats.append((float(f[4]), float(f[5]), f[1]))
            nm = {nrm(f[1]), nrm(f[2])} | {nrm(a) for a in f[3].split(',') if a}
            nm &= target
            if nm:
                pt = (float(f[4]), float(f[5]), f[0], f[8], f[7], f[6])
                for k in nm:
                    hits.setdefault(k, []).append(pt)
            if n % 4_000_000 == 0:
                log(f'geonames {n} lines ({time.time() - t0:.0f}s)')
    arr = {k: (np.array([[a[0], a[1]] for a in v]), v) for k, v in hits.items()}
    seat_xy = np.array([[a[0], a[1]] for a in seats]) if seats else np.zeros((0, 2))
    d, gid, cc, npol, nset, dseat, seatn, polhit, dpol = [], [], [], [], [], [], [], [], []
    for ks, pk, la, lo in zip(keys, primary, places['lat'], places['lon']):
        best, bg, bc, found = np.inf, '', '', False
        pol = st = 0
        ph = ''
        own = np.inf
        for k in ks:
            a = arr.get(k)
            if a is None:
                continue
            found = True
            dist, i = haversine_min(la, lo, a[0])
            if dist < best:
                best, bg, bc = dist, a[1][i][2], a[1][i][3]
            if k in pk:
                pol += sum(1 for x in a[1] if x[4] in POLITY_FCODES)
                st += sum(1 for x in a[1] if x[5] == 'P')
                if not ph:
                    ph = next((f'{x[2]}:{x[4]}:{k}' for x in a[1] if x[4] in POLITY_FCODES), '')
                for x in a[1]:
                    if x[4] in POLITY_FCODES:
                        own = min(own, float(haversine_km(la, lo, x[0], x[1])))
        d.append(round(best, 1) if found else -1.0)
        gid.append(bg)
        cc.append(bc)
        npol.append(pol)
        nset.append(st)
        polhit.append(ph)
        dpol.append(round(own, 1) if own < np.inf else np.nan)
        if pol and len(seat_xy):
            dd, i = haversine_min(la, lo, seat_xy)
            dseat.append(round(dd, 1))
            seatn.append(seats[i][2])
        else:
            dseat.append(np.nan)
            seatn.append('')
    return pd.DataFrame({'coll_slug': places['coll_slug'].to_numpy(), 'entity_key': places['entity_key'].to_numpy(),
                         'd_km': d, 'nearest_gid': gid, 'nearest_cc': cc, 'n_polity': npol, 'n_settlement': nset, 'polity_hit': polhit,
                         'd_seat_km': dseat, 'seat_name': seatn, 'd_polity_km': dpol})


def vouched_for(pl: pd.DataFrame) -> pd.Series:
    """Points that another source vouches for. (1) An accepted authority's own point (Wikidata / GeoNames id, which passed check 1): its identity is
    judged there, and the point is the authority's (rivers, buildings, a historical form GeoNames does not carry). (2) A register point within
    50 km of the accepted authority's point: two independent sources agree. (3) The curated exonym table and areas (a coarse point stands for a
    region)."""
    src = pl['coord_source']
    d_auth = pd.Series(haversine_km(pl['lat'].to_numpy(), pl['lon'].to_numpy(), pl['auth_lat'].to_numpy(), pl['auth_lon'].to_numpy()), index=pl.index)
    reg = (src == 'register') & (d_auth <= GN_THRESHOLD_KM) & pl['auth_name_level'].notna()
    tgn_name = pl['gc_tgn'].fillna(False).astype(bool) if 'gc_tgn' in pl else pd.Series(False, index=pl.index)   # v2.1: the Getty TGN gives the place that name
    # v2.3: a Latin name found in the Wikidata Latin labels / the Orbis Latinus / the tables is vouched for by that lexicon (GeoNames carries 'Hagam' in Korea, not the Hague)
    return src.isin(['wikidata', 'geonames', 'tgn']) | reg | src.isin(['name_exonym', 'name_latin']) | (pl['coord_kind'] == 'area') | tgn_name


def base_source(pl: pd.DataFrame, hops: int = 6) -> pd.Series:
    """The coord_source a point really rests on: a name-propagated point is a copy of the entity ``gc_cand`` (``collection:entity_key``); follow
    the copies to the first point that is not itself a copy. -> (coll, entity_key) of the base, or the row's own key."""
    key = list(zip(pl['coll_slug'], pl['entity_key']))
    row = {k: i for i, k in enumerate(key)}
    src = pl['coord_source'].to_numpy(dtype=object)
    cand = pl['gc_cand'].to_numpy(dtype=object) if 'gc_cand' in pl else np.array([None] * len(pl), dtype=object)
    out = np.arange(len(pl))
    for _ in range(hops):
        nxt = out.copy()
        for i, j in enumerate(out):
            if src[j] == 'name_propagation' and isinstance(cand[j], str) and ':' in cand[j]:
                coll, rest = cand[j].split(':', 1)
                t = row.get((coll, rest))
                if t is not None and t != j:
                    nxt[i] = t
        if (nxt == out).all():
            break
        out = nxt
    return pd.Series(out, index=pl.index)


def far_from_every_namesake(pl: pd.DataFrame, dist: pd.DataFrame, threshold: float = GN_THRESHOLD_KM) -> tuple:
    """-> (failures, review) of check 4. ``pl``: places_local_geo rows (coll_slug, entity_key, name, lat, lon, coord_source, coord_kind, gc_cand,
    auth_lat, auth_lon, auth_name_level, n_mentions); ``dist``: output of ``geonames_distances``. A place with no same-name feature at all
    (d = -1) is not judged. A name-propagated point rests on the first point of its chain of copies. Failures: a point the pipeline inferred
    (name-based geocoding, a chain of copies that ends in one), farther than ``threshold`` from every namesake and vouched for by nobody.
    Review (not failed): the editors' own coordinates, alone (a register point, or a copy of one)."""
    pl = pl.reset_index(drop=True)
    j = pl.merge(dist, on=['coll_slug', 'entity_key'], how='left')
    b = base_source(j)
    base = j.iloc[b.to_numpy()].reset_index(drop=True)
    vouched = vouched_for(j).to_numpy() | vouched_for(base).to_numpy()
    editorial = base['coord_source'].isin(['register', 'casebooks_directory']).to_numpy()      # v2.1: the Casebooks directory is a curated source of its own
    far = (j['d_km'] > threshold).to_numpy() & ~vouched
    f = j[far & ~editorial]
    r = j[far & editorial]

    def frame(x):
        return pd.DataFrame({'coll_slug': x['coll_slug'], 'key': x['entity_key'], 'name': x['name'], 'lat': x['lat'], 'lon': x['lon'],
                             'coord_source': x['coord_source'], 'd_km': x['d_km'], 'nearest_cc': x['nearest_cc'], 'n_mentions': x['n_mentions'],
                             'qid': x.get('qid'), 'auth_name_level': x['auth_name_level']}).reset_index(drop=True)
    return frame(f), frame(r)


def run_check4(T: Tables, zip_path: Optional[Path] = None) -> Result:
    res = Result(4, 'far_from_namesakes', 'No point farther than 50 km from every same-name GeoNames feature')
    cols = ['coll_slug', 'entity_key', 'name', 'lat', 'lon', 'coord_source', 'coord_kind', 'auth_lat', 'auth_lon', 'auth_name_level', 'n_mentions',
            'qid', 'scope', 'gc_cand', 'gc_tgn']
    try:
        pl = T.final_t('places_local_geo', cols)
    except Exception:                    # tables of v2 (no gc_tgn column)
        pl = T.final_t('places_local_geo', cols[:-1])
    pl = pl[pl['lat'].notna() & pl['lon'].notna() & (pl['scope'] != 'excluded')]
    cache = (T.out / 'gn_final_dist.tsv') if T.out else None
    if zip_path and Path(zip_path).exists():
        loc = T.all_t('places_local', ['coll_slug', 'entity_key', 'variants'])
        dist = geonames_distances(pl.merge(loc, on=['coll_slug', 'entity_key'], how='left'), Path(zip_path))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            dist.assign(lat=pl['lat'].to_numpy(), lon=pl['lon'].to_numpy()).to_csv(cache, sep='\t', index=False)
    elif cache is not None and cache.exists():
        dist = pd.read_csv(cache, sep='\t', dtype={'nearest_gid': str, 'nearest_cc': str})
        cur = pl[['coll_slug', 'entity_key', 'lat', 'lon']].merge(dist[['coll_slug', 'entity_key', 'lat', 'lon']], on=['coll_slug', 'entity_key'],
                                                                   suffixes=('', '_c'))
        if len(cur) != len(pl) or (np.abs(cur['lat'] - cur['lat_c']) > 1e-6).any() or (np.abs(cur['lon'] - cur['lon_c']) > 1e-6).any():
            res.skipped = 'cached GeoNames distances are older than the tables: rerun with --geonames-zip'
            return res
        dist = dist.drop(columns=['lat', 'lon'])
    else:
        res.skipped = 'no GeoNames dump (allCountries.zip) and no cached distances'
        return res
    res.n_checked = int((dist['d_km'] >= 0).sum())
    res.info['placed_points_checked'] = res.n_checked
    res.info['no_namesake_at_all'] = int((dist['d_km'] < 0).sum())
    res.failures, review = far_from_every_namesake(pl, dist)
    res.extra['check4_review_editorial_points'] = review
    res.info['editorial_points_far_from_namesakes (listed for review, not failed)'] = {'places': int(len(review)), 'mentions': int(review['n_mentions'].sum()),
                                                                                       'by_edition': review.groupby('coll_slug').size().sort_values(ascending=False).head(12).to_dict()}
    T._c['gn_dist'] = dist
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 5: capital for country
# ---------------------------------------------------------------------------------------------------------------------
_STATE_OF_GERMANY = re.compile(r'\bstate of germany\b', re.I)


def _label_head(label) -> str:
    """The class label up to ' of ' / ' in ': 'borough in the United Kingdom' is a borough, 'cadastral area in the Czech Republic' an area (v2.2: the check read the
    country in the label and called Bath, Prague and Cambridge "Wikidata state / country class"; v2.1 built them as areas to match)."""
    return re.sub(r'\s+(?:of|in|within|on|at|from)\s+.*$', '', str(label or ''), flags=re.I)


def capital_for_country(pl: pd.DataFrame, class_labels_of: Callable[[str], list]) -> pd.DataFrame:
    """Failures of check 5 on places_local_geo rows (needs note_type, gn_fcode, qid, coord_kind, coord_source, place_kind, wd classes).
    A country or first-order region (register type country / region, a GeoNames country code, a Wikidata state / country class) that is not
    itself a city must be an area; a country never keeps its register point."""
    rows = []
    for r in pl.itertuples(index=False):
        if r.coord_kind != 'point':
            continue
        nt = str(r.note_type or '').lower()
        fc = str(r.gn_fcode or '')
        labs = class_labels_of(r.qid) if isinstance(r.qid, str) else []
        wd_country = any(COUNTRY_CLASS.search(_label_head(x)) or _STATE_OF_GERMANY.search(x) for x in labs)
        wd_city = any(re.search(r'\b(city|town|village|municipality|commune|human settlement|big city)\b', x, re.I) for x in labs)
        why = None
        if nt in ('country', 'coun', 'countr'):
            why = 'register type country with a point'
        elif fc.startswith('PCL') and not wd_city:
            why = 'GeoNames country feature with a point'
        elif wd_country and not wd_city and r.place_kind != 'city':
            why = 'Wikidata state / country class with a point'
        elif wd_country and not wd_city:
            why = 'Wikidata state / country class, kind city'
        elif fc == 'ADM1' and not wd_city and r.coord_source in ('register', 'geonames') and r.place_kind == 'city':
            why = 'first-order region kept as a city'
        if why:
            rows.append(dict(coll_slug=r.coll_slug, key=r.entity_key, name=r.name, qid=r.qid, note_type=r.note_type, gn_fcode=r.gn_fcode,
                             place_kind=r.place_kind, coord_source=r.coord_source, why=why, n_mentions=r.n_mentions))
    return pd.DataFrame(rows, columns=['coll_slug', 'key', 'name', 'qid', 'note_type', 'gn_fcode', 'place_kind', 'coord_source', 'why',
                                       'n_mentions'])


def polity_name_points(pl: pd.DataFrame, dist: pd.DataFrame) -> tuple:
    """-> (failures, review). Points whose NAME only countries or first-order regions carry in GeoNames (Polen, Portugal, Holstein, Katalonien ...)
    and that have no register type, no accepted id to say so. A polity placed within 5 km of a capital or an administrative seat is the capital for
    the country (Kraków for the Kingdom of Poland): failure. Elsewhere it is a centroid or a coarse point (Sicily, Zeeland): listed for review."""
    empty = pd.DataFrame(columns=['coll_slug', 'key', 'name', 'why', 'n_mentions'])
    if 'n_polity' not in dist.columns or 'd_seat_km' not in dist.columns or 'd_polity_km' not in dist.columns:
        return empty, empty
    j = pl.merge(dist[['coll_slug', 'entity_key', 'n_polity', 'n_settlement', 'd_seat_km', 'seat_name', 'polity_hit', 'd_polity_km']],
                 on=['coll_slug', 'entity_key'])
    nt = j['note_type'].fillna('').str.lower()
    # 'Parish of St. Andrew' folds onto the Caribbean parishes (ADM1 of Barbados, Jamaica ...): a parish of the London register is not a polity
    j = j[(j['coord_kind'] == 'point') & (j['n_polity'] > 0) & (j['n_settlement'] == 0) & ~nt.isin(['city', 'stadt', 'settlement', 'town', 'vill', 'dorf'])
          & ~j['name'].str.contains(r'\bparish\b', case=False, regex=True, na=False)]
    # on a big seat (a capital, or an administrative seat of 100,000) and not on the polity's own GeoNames point: Krakow for the Kingdom of Poland
    seat = (j['d_seat_km'] <= SEAT_KM) & ~(j['d_polity_km'] <= POLITY_OWN_KM)

    def frame(x, why):
        return pd.DataFrame({'coll_slug': x['coll_slug'], 'key': x['entity_key'], 'name': x['name'], 'qid': x['qid'], 'note_type': x['note_type'],
                             'gn_fcode': x['gn_fcode'], 'place_kind': x['place_kind'], 'coord_source': x['coord_source'],
                             'polity_hit': x['polity_hit'], 'd_seat_km': x['d_seat_km'], 'd_polity_km': x['d_polity_km'], 'seat_name': x['seat_name'], 'why': why,
                             'n_mentions': x['n_mentions']}).reset_index(drop=True)
    return (frame(j[seat], 'a name only countries / first-order regions carry, placed on a capital or a big seat away from the polity\'s own point: capital for country'),
            frame(j[~seat], 'a polity name placed as a coarse point (the polity\'s own point, or away from any big seat)'))


def run_check5(T: Tables) -> Result:
    res = Result(5, 'capital_for_country', 'No capital-for-country point')
    auth = T.auth()
    pl = T.final_t('places_local_geo', ['coll_slug', 'entity_key', 'name', 'qid', 'note_type', 'gn_fcode', 'coord_kind', 'coord_source',
                                        'place_kind', 'n_mentions', 'scope'])
    pl = pl[(pl['scope'] != 'excluded') & (pl['n_mentions'] > 0)]

    def labs(q):
        e = auth.wd.get(q)
        return [auth.classlab.get(c, '') for c in e['classes']] if e and not e.get('missing') else []
    res.n_checked = int((pl['coord_kind'] == 'point').sum())
    f = capital_for_country(pl, labs)
    dist = T._c.get('gn_dist')
    if dist is None and T.out is not None and (T.out / 'gn_final_dist.tsv').exists():
        dist = pd.read_csv(T.out / 'gn_final_dist.tsv', sep='\t', dtype={'nearest_gid': str, 'nearest_cc': str})
    if dist is not None:
        f2, rv = polity_name_points(pl, dist)
        res.info['polity_name_on_a_capital'] = int(len(f2))
        res.info['polity_name_centroids (listed for review, not failed)'] = int(len(rv))
        res.extra['check5_review_polity_points'] = rv
        f = pd.concat([f, f2[~f2['key'].isin(f['key'])]], ignore_index=True)
    else:
        res.info['polity_name_signal'] = 'not run (no GeoNames distances: run check 4 first)'
    res.failures = f
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 6: default counts
# ---------------------------------------------------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r'^(?:s\.?\s?l\.?|s\.?\s?n\.?|n\.?\s?n\.?|pl0+|unknown|unbekannt\w*(?:\s+\w+)?|ignot[oa]|inconnu[e]?|onbekend\w*|'
                          r'an[oó]nim[oa]s?[\s_\-]*\d*|anonymus|anonymous|anonyme|anon\.?|auctores\s+varii|\?+|-+|\.+|…+)$', re.I)


def default_mention_failures(m: pd.DataFrame, with_pronouns: bool = False) -> pd.DataFrame:
    """Default mentions (canonical, not excluded) that should have been left out, with the reason. Pronouns (anaphora tagged by the editors:
    Sandrart's ich, the Tagebuch's er) are a policy question, not a rule of the audit: they are reported by ``pronoun_mentions`` and fail only with
    ``with_pronouns``."""
    d = m[m['canonical'].fillna(True).astype(bool) & ~m['excluded'].fillna(False).astype(bool)]
    surf = d['surface'].fillna('').str.strip()
    low = surf.str.lower().str.strip('.,;:!?')
    reasons = {
        'zone': d['zone'].isin(EXCLUDED_ZONES),
        'class': d['person_class'].isin(EXCLUDED_CLASSES),
        'role': d['role'].isin(EXCLUDED_ROLES),
        'placeholder': surf.map(lambda s: bool(s) and bool(_PLACEHOLDER.match(s))),
    }
    if with_pronouns:
        reasons['pronoun'] = (d['kind'] == 'person') & low.isin(PRONOUNS)
    rows = []
    for why, mask in reasons.items():
        x = d[mask.to_numpy()]
        if x.empty:
            continue
        g = x.assign(_s=x['surface'].fillna('').str.strip()).groupby(['coll_slug', 'kind', '_s'], as_index=False).size()
        for r in g.itertuples(index=False):
            rows.append(dict(coll_slug=r.coll_slug, key=f'{r.kind}:{r._2}', kind=r.kind, why=why, surface=r._2, n_mentions=int(r.size)))
    return pd.DataFrame(rows, columns=['coll_slug', 'key', 'kind', 'why', 'surface', 'n_mentions'])


def pronoun_mentions(m: pd.DataFrame) -> pd.DataFrame:
    """Default person mentions whose whole surface is a personal pronoun, by edition (information for the owner's decision)."""
    d = m[m['canonical'].fillna(True).astype(bool) & ~m['excluded'].fillna(False).astype(bool) & (m['kind'] == 'person')]
    low = d['surface'].fillna('').str.strip().str.lower().str.strip('.,;:!?')
    x = d[low.isin(PRONOUNS).to_numpy()]
    return x.groupby('coll_slug').size().rename('n_mentions').reset_index()


def count_failures(m: pd.DataFrame, local: pd.DataFrame, kind: str) -> list:
    """The local table's n_mentions must be the number of default mentions of the kind (summed over the edition's entities)."""
    d = m[(m['kind'] == kind) & m['canonical'].fillna(True).astype(bool) & ~m['excluded'].fillna(False).astype(bool)]
    from .merge import _norm            # the key of an unregistered mention is the pipeline's own normalised surface
    surf = d['surface'].fillna('').map(_norm)
    keyed = d['local_key'].notna() | (surf != '')
    want = int(keyed.sum())
    have = int(local['n_mentions'].sum())
    return [] if want == have else [dict(coll_slug='ALL', key=f'{kind}:n_mentions', why=f'sum of {kind} locals n_mentions {have} != default mentions {want}',
                                         n_mentions=have - want)]


def run_check6(T: Tables) -> Result:
    res = Result(6, 'default_counts', 'No commentary, apparatus, repository, character, placeholder or pronoun row in the default counts')
    m = T.mentions()
    res.n_checked = int((m['canonical'].fillna(True).astype(bool) & ~m['excluded'].fillna(False).astype(bool)).sum())
    f = default_mention_failures(m, with_pronouns=True)       # v2.1: the owner excluded tagged pronouns by default: they now fail the check
    pr = pronoun_mentions(m)
    res.info['pronoun_mentions_in_default_counts (must be 0 since v2.1)'] = {'total': int(pr['n_mentions'].sum()),
                                                                            'by_edition': dict(zip(pr['coll_slug'], map(int, pr['n_mentions'])))}
    extra = []
    for kind, name in (('place', 'places_local_geo'), ('person', 'persons_local_global')):
        loc = T.final_t(name, ['coll_slug', 'entity_key', 'n_mentions'])
        extra += count_failures(m, loc, kind)
    pg = T.final_t('persons_global', ['id', 'person_class_final', 'n_mentions', 'name'])
    bad = pg[pg['person_class_final'].isin(EXCLUDED_CLASSES) & (pg['n_mentions'] > 0)]
    for r in bad.itertuples(index=False):
        extra.append(dict(coll_slug='GLOBAL', key=r.id, why=f'global person of class {r.person_class_final} has default mentions', n_mentions=int(r.n_mentions)))
    pgl = T.final_t('places_global', ['id', 'place_class', 'n_mentions', 'name'])
    bad = pgl[pgl['place_class'].isin(('celestial',)) & (pgl['n_mentions'] > 0)]
    for r in bad.itertuples(index=False):
        extra.append(dict(coll_slug='GLOBAL', key=r.id, why=f'global place of class {r.place_class} has default mentions', n_mentions=int(r.n_mentions)))
    if extra:
        f = pd.concat([f, pd.DataFrame(extra)], ignore_index=True)
    res.failures = f
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 7: duplicated mentions
# ---------------------------------------------------------------------------------------------------------------------
def duplicate_failures(m: pd.DataFrame, share: float = HEADER_DUP_SHARE, strict_header: bool = False) -> tuple:
    """-> (failures, info). Casebooks: more than one person / org mention for the same (case, entity). Every edition: the share of header rows
    that repeat (document, kind, entity, role, tag) is information (a file with many letters repeats its senders legitimately: croala, the
    Sozzini letters); with ``strict_header`` an edition above ``share`` fails."""
    d = m[m['canonical'].fillna(True).astype(bool) & ~m['excluded'].fillna(False).astype(bool)].copy()
    d['_k'] = d['local_key'].fillna('s:' + d['surface'].fillna(''))
    rows = []
    cb = d[(d['coll_slug'] == 'casebooks') & d['kind'].isin(['person', 'org'])]
    dup = cb.duplicated(['doc_id', 'kind', '_k'], keep='first')
    for r in cb[dup].groupby('kind').size().items():
        rows.append(dict(coll_slug='casebooks', key=f'{r[0]}:(case, entity)', why='several mentions of one person / org in one case', n_mentions=int(r[1])))
    h = d[d['zone'] == 'header']
    hd = h.duplicated(['coll_slug', 'source_path', 'doc_id', 'kind', '_k', 'role', 'tag'], keep='first')
    tot = h.groupby('coll_slug').size()
    dup_by = h[hd].groupby('coll_slug').size()
    info = {}
    for c, n in dup_by.items():
        s = n / max(int(tot.get(c, 1)), 1)
        info[c] = round(float(s), 4)
        if strict_header and s > share:
            rows.append(dict(coll_slug=c, key='header rows', why=f'{s:.1%} of the header rows repeat (document, entity, role, tag)', n_mentions=int(n)))
    return pd.DataFrame(rows, columns=['coll_slug', 'key', 'why', 'n_mentions']), info


def run_check7(T: Tables) -> Result:
    res = Result(7, 'duplicates', 'No duplicated mention (Casebooks: one per case and person)')
    m = T.mentions()
    res.n_checked = int((m['coll_slug'] == 'casebooks').sum())
    f, info = duplicate_failures(m)
    res.info['header_duplicate_share'] = info
    cb = m[(m['coll_slug'] == 'casebooks') & (m['kind'] == 'place') & ~m['excluded'].fillna(False).astype(bool)]
    k = cb['local_key'].fillna(cb['surface'])
    res.info['casebooks_place_repeats_in_one_case (two address strings of the text, kept)'] = int(
        cb.assign(_k=k).duplicated(['doc_id', '_k'], keep='first').sum())
    res.failures = f
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 8: dates
# ---------------------------------------------------------------------------------------------------------------------
def date_failures(m: pd.DataFrame, wide: int = WIDE_RANGE, period=PERIOD) -> pd.DataFrame:
    lo, hi = m['doc_year_from'], m['doc_year_to']
    w = hi - lo
    rows = []

    def add(mask, why):
        x = m[mask.fillna(False).to_numpy()]
        if x.empty:
            return
        g = x.groupby(['coll_slug', 'date_source'], dropna=False).size().reset_index(name='n')
        for r in g.itertuples(index=False):
            rows.append(dict(coll_slug=r.coll_slug, key=f'{r.date_source}', why=why, n_mentions=int(r.n)))
    add((w > wide) & m['doc_year'].notna(), f'a range wider than {wide} years has a doc_year (midpoint)')
    add(m['doc_year'].notna() & ((m['doc_year'] < lo) | (m['doc_year'] > hi)), 'doc_year outside its own range')
    inside = (lo >= period[0]) & (hi <= period[1])
    outside = (hi < period[0]) | (lo > period[1])
    add(inside & (m['in_period'] != True), 'range inside the period but in_period is not True')       # noqa: E712
    add(outside & (m['in_period'] != False), 'range outside the period but in_period is not False')   # noqa: E712
    straddle = ~inside & ~outside & (w > wide)
    add(straddle & m['in_period'].notna(), 'wide range straddling the period boundary has an in_period')
    return pd.DataFrame(rows, columns=['coll_slug', 'key', 'why', 'n_mentions'])


def run_check8(T: Tables) -> Result:
    res = Result(8, 'dates', 'No midpoint date for a range wider than 20 years')
    m = T.mentions()
    d = m[m['doc_year_from'].notna() & m['doc_year_to'].notna()]
    res.n_checked = len(d)
    res.info['wide_ranges_without_doc_year'] = int(((d['doc_year_to'] - d['doc_year_from']) > WIDE_RANGE).sum())
    res.failures = date_failures(d)
    # information (not blocking): the share of an edition's dated mentions on one year
    dd = m[m['doc_year'].notna() & ~m['excluded'].fillna(False).astype(bool)]
    top = dd.groupby(['coll_slug', 'doc_year']).size().reset_index(name='n')
    tot = dd.groupby('coll_slug').size()
    mx = top.sort_values('n').groupby('coll_slug').tail(1).set_index('coll_slug')
    res.info['single_year_spikes(>40% of an edition with >500 dated mentions)'] = {
        c: {'year': int(r.doc_year), 'share': round(float(r.n / tot[c]), 3)} for c, r in mx.iterrows() if tot[c] > 500 and r.n / tot[c] > 0.4}
    return res


# ---------------------------------------------------------------------------------------------------------------------
# check 9: sources
# ---------------------------------------------------------------------------------------------------------------------
def source_prefix(path: str) -> str:
    """The harvest folder of a source path: ``originals_2026-10-17/<folder>`` or the first component."""
    p = str(path).split('/')
    return '/'.join(p[:2]) if p[0].startswith('originals_') and len(p) > 1 else p[0]


def sources_failures(prefixes: pd.DataFrame, has_sources: Callable[[str], bool], legacy: pd.DataFrame, fresh: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """``prefixes``: (coll_slug, prefix) of every harvested file of every edition. ``has_sources(prefix)``: the folder has a SOURCES.tsv.
    ``legacy``: slug, source_prefix, ... The rules: an uncovered prefix is listed as legacy; a legacy row covers a real, uncovered prefix; an
    edition that changed upstream after its harvest (``fresh``: slug, changed_since_harvest) has a covered prefix and no legacy row."""
    rows = []
    leg = {(s, p) for s, p in zip(legacy['slug'], legacy['source_prefix'])}
    seen = set()
    for r in prefixes.itertuples(index=False):
        seen.add((r.coll_slug, r.prefix))
        if not has_sources(r.prefix) and (r.coll_slug, r.prefix) not in leg:
            rows.append(dict(coll_slug=r.coll_slug, key=r.prefix, why='no SOURCES.tsv in the folder and not listed as legacy'))
    for s, p in sorted(leg - seen):
        rows.append(dict(coll_slug=s, key=p, why='legacy row for a folder the edition does not read'))
    for s, p in sorted(leg & seen):
        if has_sources(p):
            rows.append(dict(coll_slug=s, key=p, why='listed as legacy but the folder has a SOURCES.tsv'))
    if fresh is not None:
        stale = set(fresh.loc[fresh['changed_since_harvest'] == 'yes', 'slug'])
        for s in sorted(stale):
            if any(ls == s for ls, _ in leg):
                rows.append(dict(coll_slug=s, key='legacy', why='changed upstream after its harvest but still legacy'))
            elif not any(has_sources(p) for cs, p in seen if cs == s):
                rows.append(dict(coll_slug=s, key='sources', why='changed upstream after its harvest and has no new SOURCES.tsv'))
    return pd.DataFrame(rows, columns=['coll_slug', 'key', 'why'])


def manifest_failures(folder: Path, verify_hashes: bool = False) -> list:
    """MANIFEST.tsv (file, bytes, sha256) or MANIFEST.sha256 of a harvest folder: every file exists, has its size and (optional) its hash."""
    import hashlib
    rows = []
    entries = []
    mt = folder / 'MANIFEST.tsv'
    ms = folder / 'MANIFEST.sha256'
    if mt.exists():
        t = pd.read_csv(mt, sep='\t', dtype=str, keep_default_na=False)
        fc = 'file' if 'file' in t.columns else 'path'
        entries = [(a, int(b) if b.isdigit() else None, c) for a, b, c in zip(t[fc], t['bytes'], t['sha256'])]
    elif ms.exists():
        for line in ms.read_text(encoding='utf-8').splitlines():
            if line.strip():
                h, _, f = line.partition('  ')
                entries.append((f.strip(), None, h.strip()))
    for f, size, h in entries:
        base = f.split('!')[0]
        p = folder / base
        if not p.exists():
            rows.append(dict(coll_slug='-', key=f'{folder.name}/{f}', why='listed in the manifest, missing on disk'))
            continue
        if '!' in f:
            continue
        if size is not None and p.stat().st_size != size:
            rows.append(dict(coll_slug='-', key=f'{folder.name}/{f}', why=f'size {p.stat().st_size} != manifest {size}'))
        elif verify_hashes and h and h != '-':
            if hashlib.sha256(p.read_bytes()).hexdigest() != h:
                rows.append(dict(coll_slug='-', key=f'{folder.name}/{f}', why='sha256 differs from the manifest'))
    return rows


def run_check9(T: Tables, verify_hashes: bool = False) -> Result:
    res = Result(9, 'sources', 'Every harvested edition has a SOURCES.tsv row or is listed as legacy')
    if T.raw is None or not T.raw.exists():
        res.skipped = 'no raw root'
        return res
    m = T.mentions()
    pf = m.drop_duplicates(['coll_slug', 'source_path'])[['coll_slug', 'source_path']]
    pf = pf.assign(prefix=pf['source_path'].map(source_prefix)).drop_duplicates(['coll_slug', 'prefix'])[['coll_slug', 'prefix']]
    legacy = pd.read_csv(DATA / 'legacy_sources.tsv', sep='\t', dtype=str, keep_default_na=False)
    fresh_p = DATA / 'freshness_audit.tsv'
    fresh = pd.read_csv(fresh_p, sep='\t', dtype=str, keep_default_na=False) if fresh_p.exists() else None
    cache: dict = {}

    def has(prefix):
        if prefix not in cache:
            cache[prefix] = (T.raw / prefix / 'SOURCES.tsv').exists()
        return cache[prefix]
    f = sources_failures(pf, has, legacy, fresh)
    mf = []
    folders = sorted({p for p in pf['prefix'] if has(p)} | {p.parent.relative_to(T.raw).as_posix() for p in (T.raw / 'originals_2026-10-17').glob('*/SOURCES.tsv')})
    for p in folders:
        mf += manifest_failures(T.raw / p, verify_hashes)
    if mf:
        f = pd.concat([f, pd.DataFrame(mf)], ignore_index=True)
    res.n_checked = int(len(pf))
    res.info['folders_with_sources'] = len(folders)
    res.info['legacy_rows'] = int(len(legacy))
    res.info['hashes_verified'] = bool(verify_hashes)
    res.failures = f
    return res


# ---------------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------------
CHECKS = {1: run_check1, 2: run_check2, 3: run_check3, 4: run_check4, 5: run_check5, 6: run_check6, 7: run_check7, 8: run_check8, 9: run_check9}


def run_all(T: Tables, which=None, zip_path=None, verify_hashes=False, allow_path=None) -> list:
    allow = load_allow(allow_path)
    out = []
    for n in sorted(which or CHECKS):
        t0 = time.time()
        _t(f'check {n} ...')
        fn = CHECKS[n]
        try:
            if n in (1, 4):
                r = fn(T, zip_path)
            elif n == 9:
                r = fn(T, verify_hashes)
            else:
                r = fn(T)
        except Exception as e:      # noqa: BLE001 - a check that crashes is a failed check, not a pass
            import traceback
            traceback.print_exc()
            r = Result(n, fn.__name__, 'crashed', failures=pd.DataFrame([dict(coll_slug='-', key='-', why=f'check crashed: {type(e).__name__}: {e}')]))
        r = apply_allow(r, allow)
        _t(f'check {n}: {len(r.failures)} failures, {r.allowed} allowed, {r.n_checked} checked ({time.time() - t0:.0f}s) {r.skipped}')
        out.append(r)
    return out


def write_results(results: list, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        p = out / f'check{r.n}_{r.name}.tsv'
        if len(r.failures):
            r.failures.to_csv(p, sep='\t', index=False)
        elif p.exists():
            p.unlink()
        for name, df in r.extra.items():
            df.to_csv(out / f'{name}.tsv', sep='\t', index=False)
    (out / 'audit_v2_summary.json').write_text(json.dumps([r.summary() for r in results], indent=1, ensure_ascii=False, default=str), encoding='utf-8')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas audit v2')
    ap.add_argument('--all', type=Path, required=True, help='out_v2/all (mentions, entities, *_local)')
    ap.add_argument('--final', type=Path, required=True, help='the reconcile tables (places_local_geo, persons_global, gate_pairs ...)')
    ap.add_argument('--rdir', type=Path, required=True, help='reconcile_v2 (wd_entities, wd_map, gnd_records, gn_ids ...)')
    ap.add_argument('--raw', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--geonames-zip', type=Path, default=None)
    ap.add_argument('--tgn-dir', type=Path, default=None, help='local Getty TGN index: check 1 reads the accepted TGN ids in it')
    ap.add_argument('--verify-hashes', action='store_true')
    ap.add_argument('--checks', default=None, help='comma-separated check numbers (default all)')
    ap.add_argument('--allow', type=Path, default=None)
    a = ap.parse_args(argv)
    T = Tables(a.all, a.final, a.rdir, a.raw, a.out, a.tgn_dir)
    which = [int(x) for x in a.checks.split(',')] if a.checks else None
    results = run_all(T, which, a.geonames_zip, a.verify_hashes, a.allow)
    write_results(results, a.out)
    for r in results:
        s = r.summary()
        print(f"check {s['check']} {s['name']}: {'OK' if r.ok else 'FAIL'} failures={s['failures']} allowed={s['allowed']} checked={s['checked']} {s['skipped']}")
    return 1 if any((not r.ok) and r.blocking for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
