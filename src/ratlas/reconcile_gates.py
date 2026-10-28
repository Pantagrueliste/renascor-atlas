"""Reconcile stage, part 2c: the gates that decide which authority ids an entity may use (v2; run on Hypatia through Slurm).

    called by ``reconcile build --gates`` (see ``run``); inputs in ``--rdir`` (reconcile_v2): wd_map.tsv, wd_entities.parquet,
    wd_class_labels.tsv, gnd_records.parquet (``ratlas reconcile authority``), gn_ids.parquet, gn_samename.parquet
    (``ratlas reconcile gnscan``).

Unit of decision: a *pair* (entity, scheme, id, qid): an id of an entity and the Wikidata item it leads to (qid None when the id
has no item). Each pair is ``accepted``, ``suggested`` or ``rejected`` (see ``namegate`` for the reasons) and every pair that is not
accepted goes to ``gate_log.tsv``. A pair is accepted when

* the id is not an editor's low / medium ``@cert`` identification (every tag that carries it has cert low / medium and the
  register does not give it): otherwise ``suggested`` (reason ``cert``);
* the item (or GND record) is of the right class: persons need P31 human, except for ``person_class`` dynasty / mythological /
  biblical / character / people (their own classes); no disambiguation pages, given names, lists, offices, organisations, places;
  places must not be people, companies, taxa, works ... ; duplicates are followed to their target (redirect or P2959);
* the authority's names (labels and aliases in ten languages incl. Latin, GND names, GeoNames names) are compatible with the
  entry's names, or a second independent id agrees (another scheme leading to the same item; for places also a same-name GeoNames feature
  within a few km of the authority's coordinate that carries one of the item's names too: the entry name and the item are one feature);
* chronology (persons): not born after the last document that mentions them, register dates within 50 years of Wikidata's;
* homonyms (GeoNames): a settlement with fewer than 100 inhabitants that shares the entry's (unqualified) name with a town of the same
  country more than 50 km away is a *candidate*; v2.1: rejected only when it lies more than 30 km from every other placed place of the
  edition (the editor's territory), suggested (``homonym_unverified``) when the edition has too few placed places, accepted otherwise.
* TGN (v2.1): the names of the subject in the local Getty index (all languages, historical forms) are compared with the entry's names:
  compatible = accepted, not = rejected (``tgn_name_mismatch``); a subject the index does not hold stays suggested (``unverified``).

Wikipedia ids (``lang:Title``) are ids only if the title itself matches the entry (MoEML's links to battles and abbeys).
Ids without any label source (VIAF, HLS, ULAN, a TGN id the local index does not hold) cannot be tested: ``suggested`` / ``unverified``.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import namegate as NG
from . import tgn as TGN
from .merge import _norm
from .reconcile_ids import PROPS, _long_ids, normalise_long

REL = {'wikidata': 3.0, 'geonames': 2.0, 'tgn': 2.0, 'pleiades': 2.0, 'gnd': 2.0, 'hls': 1.5, 'viaf': 1.5, 'ulan': 1.5,
       'isni': 1.5, 'loc': 1.5, 'cerl': 1.5, 'odnb': 1.5, 'bnf': 1.5, 'wikipedia': 1.0}
PLACE_SCHEMES_ON_PERSON = ('geonames', 'tgn', 'pleiades')
LOW_CERT = ('low', 'medium')
# editions whose VIAF column is unreliable (sandrart: GND and VIAF disagree for 45 locals; the VIAF-only QID is the wrong one): a VIAF id
# counts only when a second id leads to the same item
VIAF_UNRELIABLE = frozenset({'sandrart'})
# an id carried by a small minority of the mentions of a bare surface ('Europa': one Pleiades link of the province, 11 mentions of the continent)
# is not the identity of the surface
MINORITY_SHARE = 0.34
ALIAS_NEAR_KM = 10.0


def _t(msg):
    print(f'[gates {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
class AuthData:
    """Everything the gates know about the authorities."""

    def __init__(self, rdir: Path, tgn_dir: Optional[Path] = None):
        rdir = Path(rdir)
        self.tgn_dir = Path(tgn_dir) if tgn_dir else None
        self.tgn: dict = {}               # subject_id -> {'names', 'lat', 'lon', 'type', ...} for the TGN ids the editions give (load_tgn)
        m = pd.read_csv(rdir / 'wd_map.tsv', sep='\t', dtype=str, keep_default_na=False)
        self.mapping: dict = {}
        for s, i, qs in zip(m['scheme'], m['id'], m['qids']):
            q = [x for x in qs.split('|') if x]
            if q:
                self.mapping[(s, i)] = q
        # deutsche-biographie ids are GND numbers
        for (s, i), q in list(self.mapping.items()):
            if s == 'deutsche-biographie':
                self.mapping.setdefault(('gnd', i), q)
        e = pd.read_parquet(rdir / 'wd_entities.parquet')
        self.wd: dict = {}
        for r in e.itertuples(index=False):
            if r.missing:
                self.wd[r.qid] = {'missing': True}
                continue
            lab = json.loads(r.labels) if r.labels else {}
            ali = json.loads(r.aliases) if r.aliases else {}
            names = list(dict.fromkeys(list(lab.values()) + [a for v in ali.values() for a in v]))
            self.wd[r.qid] = {
                'missing': False, 'redirect': r.redirect_to if isinstance(r.redirect_to, str) else None,
                'dup': list(r.dup_of) if r.dup_of is not None and not isinstance(r.dup_of, float) else [],
                'classes': list(r.classes) if r.classes is not None else [], 'names': names,
                'label': lab.get('en') or lab.get('mul') or next(iter(lab.values()), None), 'labels': lab,
                'occupations': list(r.occupations) if getattr(r, 'occupations', None) is not None else [],
                'lat': r.lat if r.lat == r.lat else None, 'lon': r.lon if r.lon == r.lon else None,
                'birth': NG.wd_year(r.birth), 'birth_prec': r.birth_prec, 'death': NG.wd_year(r.death),
                'death_prec': r.death_prec, 'n_claims': int(r.n_claims or 0), 'country': r.country_qid}
        cl = pd.read_csv(rdir / 'wd_class_labels.tsv', sep='\t', dtype=str, keep_default_na=False)
        self.classlab = dict(zip(cl['qid'], cl['label_en']))
        g = pd.read_parquet(rdir / 'gnd_records.parquet') if (rdir / 'gnd_records.parquet').exists() else pd.DataFrame()
        self.gnd: dict = {}
        for r in g.itertuples(index=False):
            names = [r.name] if isinstance(r.name, str) else []
            names += list(r.variants) if r.variants is not None else []
            self.gnd[str(r.gnd)] = {'names': names, 'types': list(r.types) if r.types is not None else [],
                                    'qids': list(r.qids) if r.qids is not None else [], 'birth': r.birth, 'death': r.death}
        self.gn: dict = {}
        p = rdir / 'gn_ids.parquet'
        if p.exists():
            for r in pd.read_parquet(p).itertuples(index=False):
                self.gn[str(r.gid)] = {'names': set(r.names.split('|')) if isinstance(r.names, str) else set(),
                                       'name': r.name, 'lat': float(r.lat), 'lon': float(r.lon), 'pop': int(r.pop),
                                       'cc': r.cc, 'fcode': r.fcode, 'fclass': r.fclass}
        self.same: dict = defaultdict(list)            # normalised name -> [feature]
        p = rdir / 'gn_samename.parquet'
        if p.exists():
            sm = pd.read_parquet(p)
            has_names = 'names' in sm.columns
            for r in sm.itertuples(index=False):
                self.same[r.key].append({'gid': str(r.gid), 'lat': float(r.lat), 'lon': float(r.lon), 'pop': int(r.pop),
                                         'cc': r.cc, 'fcode': r.fcode,
                                         'names': (set(r.names.split('|')) if isinstance(r.names, str) else set()) if has_names else None})
        self.weak: set = set()
        _t(f'authority data: {len(self.mapping)} mapped ids, {len(self.wd)} items, {len(self.gnd)} GND records, '
           f'{len(self.gn)} GeoNames features, {len(self.same)} same-name keys')

    def load_tgn(self, ids, log=None) -> int:
        """Load the local TGN subjects (names, coordinates, type) of ``ids`` (no-op without ``tgn_dir``); returns the number found."""
        if self.tgn_dir is None:
            return 0
        self.tgn.update(TGN.load_subset(self.tgn_dir, ids))
        if log:
            log(f'TGN: {len(self.tgn)} subjects found for {len(set(ids))} ids ({TGN.ATTRIBUTION})')
        return len(self.tgn)

    def canon(self, q: str):
        """-> (qid, note, entry) after following redirects and permanent duplicates; qid None if the item is gone."""
        note = ''
        for _ in range(4):
            e = self.wd.get(q)
            if e is None:
                return q, note, None
            if e.get('missing'):
                return None, 'qid_missing', e
            if e['redirect'] and e['redirect'] != q:
                q, note = e['redirect'], 'redirect'
                continue
            if e['dup']:
                q, note = e['dup'][0], 'duplicate'
                continue
            return q, note, e
        return q, note, self.wd.get(q)

    def person_labels(self) -> list:
        return [e['names'] for e in self.wd.values() if not e.get('missing') and NG.HUMAN in e['classes']]


# ---------------------------------------------------------------------------------------------------------------------
# votes
# ---------------------------------------------------------------------------------------------------------------------
def id_votes(v2: Path, kinds=('person', 'place')) -> pd.DataFrame:
    """(coll_slug, kind, entity_key, scheme, id) -> n, n_ip, n_hi (mentions whose tag has cert high / none), reg (given by the
    register entry), sug (id written as ``viaf-suggested:`` by the extractor: the editor's cert was low / medium)."""
    cols = ['coll_slug', 'kind', 'canonical', 'local_key', 'surface', 'in_period', 'auth', 'cert']
    t = pq.read_table(v2 / 'mentions.parquet', columns=cols)
    t = t.filter(pc.and_(pc.equal(t.column('canonical'), True), pc.is_in(t.column('kind'), pa.array(list(kinds)))))
    _t(f'mentions {t.num_rows}')
    lk = t.column('local_key').to_pandas()
    surf = t.column('surface').to_pandas().fillna('')
    nm = {s: _norm(s) for s in surf.unique()}
    n = surf.map(nm)
    key = np.where(lk.notna(), lk, 'surf:' + n)
    keep = (lk.notna() | (n != '')).to_numpy()
    cert = t.column('cert').to_pandas()
    t = t.append_column('entity_key', pa.array(key, type=pa.string()))
    t = t.append_column('ip', pc.fill_null(pc.equal(t.column('in_period'), True), False))
    t = t.append_column('hi', pa.array((~cert.isin(LOW_CERT)).to_numpy()))
    t = t.filter(pa.array(keep))
    d = _long_ids(t, {'coll_slug': None, 'kind': None, 'entity_key': None, 'ip': None, 'hi': None})
    d = _normalise(d)
    d['n'] = 1
    g = d.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id', 'sug'], as_index=False).agg(
        n=('n', 'sum'), n_ip=('ip', 'sum'), n_hi=('hi', 'sum'))
    g['reg'] = False
    e = pq.read_table(v2 / 'entities.parquet', columns=['coll_slug', 'kind', 'local_key', 'auth'])
    e = e.filter(pc.is_in(e.column('kind'), pa.array(list(kinds))))
    e = e.append_column('entity_key', e.column('local_key'))
    de = _normalise(_long_ids(e, {'coll_slug': None, 'kind': None, 'entity_key': None}))
    ge = de.drop_duplicates().assign(n=0, n_ip=0, n_hi=0, reg=True)
    out = pd.concat([g, ge[g.columns]], ignore_index=True)
    out = out.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id'], as_index=False).agg(
        n=('n', 'sum'), n_ip=('n_ip', 'sum'), n_hi=('n_hi', 'sum'), reg=('reg', 'max'), sug=('sug', 'min'))
    # an id the register gives is the register's, whatever the tag's cert; a suggested id never counts as given
    _t(f'id votes: {len(out)} rows')
    return out


def _normalise(d: pd.DataFrame) -> pd.DataFrame:
    """normalise_long + the ``viaf-suggested`` pseudo-scheme (kept as viaf with sug=True); deutsche-biographie -> gnd."""
    if d.empty:
        d = d.copy()
        d['sug'] = False
        return d
    sg = d['scheme'].astype(str).str.startswith('viaf-suggested')
    d = d.copy()
    d['sug'] = sg.to_numpy()
    d.loc[sg, 'scheme'] = 'viaf'
    d = normalise_long(d.rename(columns={'sug': '_sug'})).rename(columns={'_sug': 'sug'})
    d.loc[d['scheme'] == 'deutsche-biographie', 'scheme'] = 'gnd'
    return d


# ---------------------------------------------------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------------------------------------------------
def _names_person(name, variants) -> list:
    out = [name] if isinstance(name, str) else []
    if variants is not None:
        out += [v for v in list(variants) if isinstance(v, str)]
    return list(dict.fromkeys(out))[:14]


def run(votes: pd.DataFrame, ents: pd.DataFrame, auth: AuthData, kind: str, log=_t) -> dict:
    """Gate every (entity, scheme, id[, qid]) of ``kind``. ``ents``: the local table of the kind (coll_slug, entity_key, name,
    variants, person_class, note_type, birth, death, last_year, lat, lon). Returns dict(pairs, log)."""
    t0 = time.time()
    v = votes[votes['kind'] == kind]
    E = ents.set_index(['coll_slug', 'entity_key'])
    if not auth.weak and kind == 'person':
        auth.weak, _ = NG.build_weak(auth.person_labels(), 40)
        log(f'weak tokens: {len(auth.weak)}')
    ctxf = None
    if kind == 'place':
        if auth.tgn_dir is not None:
            auth.load_tgn(v.loc[v['scheme'] == 'tgn', 'id'].unique(), log)
        ctx = edition_context(v, ents, auth)
        log(f'edition context: {sum(len(c[1]) for c in ctx.values())} placed places in {len(ctx)} editions')
        ctxf = lambda coll, ek: ctx_without(ctx, coll, ek)           # noqa: E731
    ename_cache: dict = {}
    rows = []
    by_ent = v.groupby(['coll_slug', 'entity_key'], sort=False)
    total = len(by_ent)
    for n_ent, ((coll, ek), grp) in enumerate(by_ent):
        try:
            er = E.loc[(coll, ek)]
        except KeyError:
            continue
        if isinstance(er, pd.DataFrame):
            er = er.iloc[0]
        pclass = er.get('person_class') if isinstance(er.get('person_class'), str) else None
        if kind == 'person':
            pclass = NG.infer_person_class(er['name'], pclass)
        names = _names_person(er['name'], er['variants'])
        ekeys = NG.name_keys(er['name'], er['variants']) if kind == 'place' else None
        # ---- 1. evidence per vote
        evs = []          # (vote row, [evidence dicts])
        for r in grp.itertuples(index=False):
            evs.append((r, _evidence(r, kind, auth, names, ekeys, pclass, er, ctxf, (coll, ek))))
        # ---- 2. second independent agreement (different schemes leading to the same canonical qid)
        qid_schemes: dict = defaultdict(set)
        for r, ev in evs:
            if r.sug or (r.n_hi == 0 and not r.reg and r.n > 0):
                continue
            for e in ev:
                if e['qid'] and e['class_ok'] is True:
                    qid_schemes[e['qid']].add(r.scheme)
        for r, ev in evs:
            for e in ev:
                other = qid_schemes.get(e['qid'], set()) - {r.scheme} if e['qid'] else set()
                e['second_id'] = bool(other)
                e['second_ids'] = '+'.join(sorted(other))
        # ---- 3. decisions
        for r, ev in evs:
            rows.extend(_decide(r, ev, kind, er, coll, ek, auth, pclass, names))
        if n_ent and n_ent % 20000 == 0:
            log(f'{kind}: {n_ent}/{total} entities ({time.time() - t0:.0f}s)')
    pairs = pd.DataFrame(rows)
    log(f'{kind}: {len(pairs)} pairs for {total} entities ({time.time() - t0:.0f}s); statuses '
        f'{pairs["status"].value_counts().to_dict() if len(pairs) else {}}')
    return {'pairs': pairs}


def _evidence(r, kind: str, auth: AuthData, names: list, ekeys, pclass, er, ctxf=None, ent=None) -> list:
    """Evidence records for one vote: one per QID the id leads to, one for the id's own record (GND / GeoNames / TGN)."""
    out = _evidence1(r, kind, auth, names, ekeys, pclass, er, ctxf, ent)
    if r.scheme == 'geonames' and any(e.get('homonym') is not None for e in out):
        hv = next(e['homonym'] for e in out if e.get('homonym') is not None)
        for e in out:                                   # the items reached through a homonym hamlet are rejected with it
            e.update(class_ok=False, class_reason='homonym', homonym=hv)
    elif r.scheme == 'geonames' and any(e.get('homonym_suspect') is not None for e in out):
        hv = next(e['homonym_suspect'] for e in out if e.get('homonym_suspect') is not None)
        for e in out:                                   # no context to decide: the pair and the items behind it are only suggested
            e['homonym_suspect'] = hv
    return out


def _evidence1(r, kind: str, auth: AuthData, names: list, ekeys, pclass, er, ctxf=None, ent=None) -> list:
    out = []
    scheme, idv = r.scheme, r.id
    if kind == 'person' and scheme in PLACE_SCHEMES_ON_PERSON:
        return [{'src': 'ignored', 'qid': None, 'class_ok': False, 'class_reason': 'place_id_on_person', 'verifiable': True,
                 'name_ok': False}]
    if scheme == 'wikidata':
        qids = [idv]
    else:
        qids = auth.mapping.get((scheme, idv), [])
    for q0 in qids:
        q, note, e = auth.canon(q0)
        ev = {'src': 'wd', 'qid': q, 'qid_raw': q0, 'note': note, 'class_ok': None, 'class_reason': '', 'name_ok': False,
              'verifiable': False, 'level': '', 'lat': None, 'lon': None, 'cat': '', 'label': None, 'k': len(qids)}
        if q is None:
            ev.update(class_ok=False, class_reason='qid_missing')
        elif e is None:
            ev.update(class_reason='unfetched')
        else:
            ev['label'] = e['label']
            ev['lat'], ev['lon'] = e['lat'], e['lon']
            if kind == 'person':
                cat = NG.person_category(e['classes'], auth.classlab)
                ok, why = NG.person_class_ok(pclass, cat)
                ev['cat'] = cat
                ev['class_ok'], ev['class_reason'] = ok, why
                if scheme == 'wikipedia':
                    ev['name_ok'] = NG.wikipedia_title_ok(idv.split(':', 1)[1], names, 'person', auth.weak)
                    ev['verifiable'] = True
                    ev['level'] = 'title'
                elif e['names']:
                    st, _m = NG.compare_person(names, e['names'], auth.weak)
                    ev['verifiable'] = st not in ('no_label', 'no_name')
                    ev['name_ok'] = st in ('ok', 'ok_weak')
                    ev['level'] = st
            else:
                ok, why = NG.place_class_ok(e['classes'])
                if ok and str(er.get('note_type') or '').lower() in ('country', 'coun') \
                        and NG.settlement_only([auth.classlab.get(c, '') for c in e['classes']]):
                    ok, why = False, 'type_mismatch'          # the register says country, the item is a village
                ev['class_ok'], ev['class_reason'] = ok, why
                if scheme == 'wikipedia':
                    ev['name_ok'] = NG.wikipedia_title_ok(idv.split(':', 1)[1], names, 'place')
                    ev['verifiable'] = True
                    ev['level'] = 'title'
                elif e['names']:
                    lvl, _a, _b = NG.place_relation(ekeys, NG.auth_keys(e['names']))
                    ev['verifiable'] = not lvl.startswith(('noauth', 'noname'))
                    ev['name_ok'] = NG.place_name_ok(lvl)
                    ev['level'] = lvl
                    if not ev['name_ok'] and ev['verifiable']:
                        ev['near'] = _near_evidence(ev, er, ekeys, auth, e['names'])
                        if ev['near']:
                            ev['name_ok'] = True
                            ev['level'] = ev['near']
        out.append(ev)
    # the id's own record
    if scheme == 'gnd' and idv in auth.gnd:
        g = auth.gnd[idv]
        ev = {'src': 'gnd', 'qid': None, 'class_ok': None, 'class_reason': '', 'name_ok': False, 'verifiable': bool(g['names']),
              'level': '', 'lat': None, 'lon': None, 'label': g['names'][0] if g['names'] else None}
        ok, why = NG.gnd_class_ok(g['types'], kind)
        ev['class_ok'], ev['class_reason'] = ok, why
        if g['names']:
            if kind == 'person':
                st, _m = NG.compare_person(names, g['names'], auth.weak)
                ev['name_ok'] = st in ('ok', 'ok_weak')
                ev['verifiable'] = st not in ('no_label', 'no_name')
                ev['level'] = st
            else:
                lvl, _a, _b = NG.place_relation(ekeys, NG.auth_keys(g['names']))
                ev['name_ok'] = NG.place_name_ok(lvl)
                ev['verifiable'] = not lvl.startswith(('noauth', 'noname'))
                ev['level'] = lvl
        out.append(ev)
    if scheme == 'geonames' and kind == 'place' and idv in auth.gn:
        f = auth.gn[idv]
        lvl, _a, _b = NG.place_relation(ekeys, NG.auth_keys(list(f['names']) + [f['name']]))
        ev = {'src': 'geonames', 'qid': None, 'class_ok': True, 'class_reason': '', 'name_ok': NG.place_name_ok(lvl),
              'verifiable': True, 'level': lvl, 'lat': f['lat'], 'lon': f['lon'], 'label': f['name'], 'gn': f}
        hv = _homonym(f, idv, ekeys, auth, er['name'], ctxf(*ent) if ctxf is not None and ent is not None else None)
        if hv is not None and hv.reject:
            ev.update(class_ok=False, class_reason='homonym', homonym=hv)
        elif hv is not None and hv.suspect:
            ev['homonym_suspect'] = hv
        out.append(ev)
    if scheme == 'tgn' and kind == 'place' and auth.tgn_dir is not None:
        out.append(_tgn_evidence(idv, auth, ekeys, er))
    return out


def _tgn_evidence(idv: str, auth: AuthData, ekeys, er) -> dict:
    """The id's own record in the local TGN: its names (all languages and historical forms) against the entry's names.

    v2.2: a fuzzy relation counts from ``NG.TGN_FUZZY_MIN`` (0.90) only; below it the pair needs independent evidence (a GeoNames feature that carries
    both names) and is otherwise only suggested (``tgn_fuzzy``). An entry of the form 'Head, Qualifier' whose TGN match is on the qualifier alone
    (Bad Düben, Alaunwerk: TGN has an Alaunwerk in Brandenburg) is rejected when the head is a place that lies far from the TGN point."""
    t = auth.tgn.get(str(idv))
    ev = {'src': 'tgn', 'qid': None, 'class_ok': True, 'class_reason': '', 'name_ok': False, 'verifiable': False, 'level': '', 'lat': None,
          'lon': None, 'label': None}
    if t is None:
        return dict(ev, class_ok=None, class_reason='tgn_unfound', verifiable=False)
    ev.update(label=t['name'], lat=t['lat'], lon=t['lon'], tgn=t)
    if not t['names']:
        return ev
    lvl, _a, _b = NG.place_relation(ekeys, NG.auth_keys(t['names']))
    ev['verifiable'] = not lvl.startswith(('noauth', 'noname'))
    ev['name_ok'] = NG.place_name_ok(lvl, NG.TGN_FUZZY_MIN)
    ev['level'] = lvl
    ev['fuzzy_low'] = lvl.startswith('fuzzy') and not ev['name_ok']
    if not ev['name_ok'] and ev['verifiable']:
        near = _near_evidence(ev, er, ekeys, auth, t['names'])
        if near:
            ev['name_ok'], ev['level'], ev['fuzzy_low'] = True, near, False
    if ev['name_ok'] and ev['level'] != 'near_geonames_alias':
        far = _head_far(er.get('name'), t, auth)
        if far:
            ev.update(name_ok=False, level='head_far', head_far=far)
    return ev


HEAD_FAR_KM = 100.0                    # the head of 'Head, Qualifier' is a settlement: the TGN point must be within this distance of it
HEAD_FAR_AREA_KM = 300.0               # ... an area (country, region): within this distance


def _head_far(name, t: dict, auth: AuthData) -> str:
    """'' or the reason: the entry name is 'Head, Qualifier', the TGN names match the qualifier but not the head, and the head (GeoNames features of
    that name) lies far from the TGN point."""
    if not isinstance(name, str) or ',' not in name or t.get('lat') is None:
        return ''
    hk = NG.head_key(name)
    if len(hk) < 3:
        return ''
    lvl, _a, _b = NG.place_relation({hk}, NG.auth_keys(t['names']))
    if NG.place_name_ok(lvl, NG.TGN_FUZZY_MIN):
        return ''
    feats = auth.same.get(hk) or []
    if not feats:
        return ''
    best = None
    for f in feats:
        d = float(NG.hav(t['lat'], t['lon'], f['lat'], f['lon']))
        fc = str(f.get('fcode') or '')
        lim = HEAD_FAR_AREA_KM if (fc.startswith(('PCL', 'RGN', 'TERR', 'AREA')) or fc in ('ADM1', 'ADM2', 'ADMD')) else HEAD_FAR_KM      # ADM3 / ADM4 are communes
        if best is None or d - lim < best[0]:
            best = (d - lim, d, lim)
    if best[0] > 0:
        return f"the head {hk!r} lies {best[1]:.0f} km from the TGN point (limit {best[2]:.0f} km) and the TGN names match only the qualifier"
    return ''


def _near_evidence(ev, er, ekeys, auth: AuthData, item_names) -> str:
    """Independent evidence for a place whose names do not match the item's labels: a GeoNames feature that carries an entry name lies within a
    few km of the item's coordinate AND carries one of the item's names too (Venedig and Venezia are names of one feature; Halle and Altstadt, or
    Balingen and Bisingen, are not). A register coordinate next to the item's point is not evidence of the item's identity (v2 audit, check 1:
    Halle (Saale) took the district Altstadt, Lenzburg a Roman vicus, Neapel a street); nor is a feature whose names are unknown (an older scan)."""
    if ev['lat'] is None:
        return ''
    lat, lon = ev['lat'], ev['lon']
    ik = NG.auth_keys(item_names)
    for k in ekeys:
        for f in auth.same.get(k, ()):
            nm = f.get('names')
            if not nm or float(NG.hav(lat, lon, f['lat'], f['lon'])) > ALIAS_NEAR_KM:
                continue
            lvl, _a, _b = NG.place_relation(NG.auth_keys(list(nm)), ik)
            if lvl in ('exact', 'fold', 'stem'):
                return 'near_geonames_alias'
    return ''


def _same_name(ekeys, auth: AuthData) -> list:
    same = []
    seen = set()
    for k in ekeys:
        for s in auth.same.get(k, ()):
            if s['gid'] not in seen:
                seen.add(s['gid'])
                same.append(s)
    return same


def _homonym(f: dict, gid: str, ekeys, auth: AuthData, name, ctx=None) -> Optional[NG.HomonymVerdict]:
    """The verdict of the homonym gate on a GeoNames feature (None when the pair is not even a candidate). ``ctx``: (lats, lons) of the other
    placed places of the edition."""
    if int(f['pop']) >= NG.HOMONYM_MAX_POP:
        return None
    same = _same_name(ekeys, auth)
    if not same:
        return None
    v = NG.homonym_check({'gid': gid, 'lat': f['lat'], 'lon': f['lon'], 'pop': f['pop'], 'cc': f['cc'], 'fcode': f['fcode']}, same, name, ctx)
    return v if (v.reject or v.suspect) else None


def _valid_pt(lat, lon) -> bool:
    return lat is not None and lon is not None and lat == lat and lon == lon and abs(lat) <= 90 and abs(lon) <= 180 and not (lat == 0 and lon == 0)


def edition_context(v: pd.DataFrame, ents: pd.DataFrame, auth: AuthData) -> dict:
    """{coll_slug: (entity keys, lats, lons)}: one point per live place of the edition, for the homonym gate. The point is the register's
    coordinate, else the feature of a GeoNames id, else the point of a Wikidata item the entity's ids lead to. A GeoNames id that is itself a
    homonym candidate (the pair under test) is not a witness."""
    live = ents[ents['n_mentions'] > 0] if 'n_mentions' in ents else ents
    live = live.assign(**{c: (np.nan if c in ('lat', 'lon') else None) for c in ('lat', 'lon', 'variants') if c not in live})
    ids: dict = defaultdict(list)
    for r in v.itertuples(index=False):
        ids[(r.coll_slug, r.entity_key)].append((r.scheme, str(r.id)))
    out: dict = defaultdict(lambda: ([], [], []))
    for r in live.itertuples(index=False):
        lat = lon = None
        if _valid_pt(r.lat, r.lon):
            lat, lon = float(r.lat), float(r.lon)
        else:
            vs = ids.get((r.coll_slug, r.entity_key), ())
            ekeys = None
            for sc, i in vs:
                if sc != 'geonames' or i not in auth.gn:
                    continue
                f = auth.gn[i]
                if ekeys is None:
                    ekeys = NG.name_keys(r.name, r.variants)
                if _homonym(f, i, ekeys, auth, r.name, ((), ())) is None and _valid_pt(f['lat'], f['lon']):
                    lat, lon = f['lat'], f['lon']
                    break
            if lat is None:
                for sc, i in vs:
                    qs = [i] if sc == 'wikidata' else auth.mapping.get((sc, i), [])
                    for q0 in qs:
                        e = auth.wd.get(q0)
                        if e and not e.get('missing') and _valid_pt(e['lat'], e['lon']):
                            lat, lon = float(e['lat']), float(e['lon'])
                            break
                    if lat is not None:
                        break
        if lat is not None:
            k, la, lo = out[r.coll_slug]
            k.append(r.entity_key)
            la.append(lat)
            lo.append(lon)
    return {c: (np.array(k, dtype=object), np.array(la), np.array(lo)) for c, (k, la, lo) in out.items()}


def ctx_without(ctx: dict, coll: str, ek: str):
    """(lats, lons) of the edition's context points, without the entity itself."""
    c = ctx.get(coll)
    if c is None:
        return ((), ())
    keep = c[0] != ek
    return (c[1][keep], c[2][keep])


def _decide(r, evs: list, kind: str, er, coll: str, ek: str, auth: AuthData, pclass, names: list) -> list:
    """Decisions for the pairs of one vote."""
    base = dict(coll_slug=coll, kind=kind, entity_key=ek, name=er['name'], scheme=r.scheme, id=r.id, n_mentions=int(r.n),
                n_in_period=int(r.n_ip))
    out = []
    low_cert = bool(r.sug) or (r.n_hi == 0 and not r.reg and r.n > 0)
    total = er.get('n_mentions_all')
    minority = (ek.startswith('surf:') and not r.reg and total is not None and total == total and int(total) >= 3
                and r.n / max(int(total), 1) < MINORITY_SHARE)
    if minority and not low_cert:
        out.append(dict(base, qid=None, qid_raw=None, status='suggested', reason='minority_id', src='',
                        detail=f'{int(r.n)} of {int(total)} mentions of this surface carry the id (one occurrence is not the whole name)'))
        return out
    if not evs:
        reason = 'cert' if low_cert else 'unverified'
        out.append(dict(base, qid=None, qid_raw=None, status='suggested', reason=reason, detail=_cert_detail(r) if low_cert
                        else 'no label source for this id', src=''))
        return out
    for e in evs:
        d = dict(base, qid=e['qid'], qid_raw=e.get('qid_raw'), src=e['src'], level=e.get('level', ''),
                 label=e.get('label'), second_id=e.get('second_ids', ''), k=e.get('k', 1), authority_lat=e.get('lat'),
                 authority_lon=e.get('lon'))
        status, reason, detail = _status(e, r, kind, er, low_cert, pclass, auth)
        d.update(status=status, reason=reason, detail=detail)
        if e['src'] == 'ignored':
            d['status'] = 'ignored'
        out.append(d)
    return out


def _cert_detail(r) -> str:
    return "the editor's @cert is low / medium" if r.sug else 'every tag that carries this id has @cert low / medium'


def _status(e: dict, r, kind: str, er, low_cert: bool, pclass, auth: AuthData):
    if e['src'] == 'ignored':
        return 'ignored', e['class_reason'], 'a place id on a person row is the birth / death place'
    if e['class_ok'] is False:
        d = ''
        if e['class_reason'] == 'homonym':
            d = e['homonym'].reason
        elif e.get('cat'):
            d = f"item class {e['cat']}"
        return 'rejected', e['class_reason'] or 'class', d
    if low_cert:
        return 'suggested', 'cert', _cert_detail(r)
    if e.get('homonym_suspect') is not None and not e['second_id']:
        return 'suggested', 'homonym_unverified', e['homonym_suspect'].reason
    if r.scheme == 'viaf' and r.coll_slug in VIAF_UNRELIABLE and not e['second_id']:
        return 'suggested', 'viaf_unreliable', 'the VIAF column of this edition disagrees with its GND column'
    if e['class_ok'] is None:
        if e.get('class_reason') == 'tgn_unfound':
            return 'suggested', 'unverified', 'the TGN subject is not in the local index'
        return 'suggested', ('unfetched' if e.get('class_reason') == 'unfetched' else 'no_class'), \
            ('the item was not fetched' if e.get('class_reason') == 'unfetched' else 'the item has no P31')
    if not e['verifiable']:
        if e['second_id']:
            return 'accepted', 'second_id', f"unlabelled item; also given by {e['second_ids']}"
        return 'suggested', 'unverified', 'no labels to compare with'
    if (e['name_ok'] or e['second_id']) and e['src'] == 'wd' and kind == 'person':
        # chronology applies to every acceptance, also the one a second id gives (the second id can lead to the same wrong item)
        b = auth.wd[e['qid']]
        if NG.chronology_flag(b['birth'], b['birth_prec'], er.get('last_year')):
            return 'suggested', 'chronology', f"born {b['birth']}, last document {int(er.get('last_year'))}"
        if e['name_ok'] and NG.register_dates_flag(er.get('birth'), er.get('death'), b['birth'], b['birth_prec'], b['death'], b['death_prec']):
            return 'suggested', 'register_dates', f"register {er.get('birth')}-{er.get('death')}; Wikidata {b['birth']}-{b['death']}"
    if e['name_ok']:
        return 'accepted', ('title' if e.get('level') == 'title' else e.get('level') or 'name'), ''
    if e['src'] == 'tgn' and e.get('fuzzy_low'):
        names = ' | '.join(e['tgn']['names'][:6])
        return 'suggested', 'tgn_fuzzy', f"entry {str(er['name'])[:60]!r} vs TGN {e['tgn'].get('type')} {names[:120]!r} ({e.get('level')} < {NG.TGN_FUZZY_MIN})"
    if e['src'] == 'tgn' and e.get('head_far'):
        return 'rejected', 'tgn_head_far', f"entry {str(er['name'])[:60]!r}: {e['head_far']}"
    if e['src'] == 'tgn':
        names = ' | '.join(e['tgn']['names'][:6])
        return 'rejected', 'tgn_name_mismatch', f"entry {str(er['name'])[:60]!r} vs TGN {e['tgn'].get('type')} {names[:120]!r} ({e.get('level')})"
    if e['second_id']:
        return 'accepted', 'second_id', f"names differ ({e.get('level')}); also given by {e['second_ids']}"
    why = {'given_only': 'given_name_only'}.get(e.get('level'), 'wikipedia_title' if e.get('level') == 'title' else 'name_mismatch')
    return 'suggested', why, f"entry {str(er['name'])[:60]!r} vs {str(e.get('label'))[:60]!r} ({e.get('level')})"


# ---------------------------------------------------------------------------------------------------------------------
# choosing the identity of an entity from the accepted pairs
# ---------------------------------------------------------------------------------------------------------------------
def choose(pairs: pd.DataFrame, votes: pd.DataFrame) -> dict:
    """-> dict(qid=DataFrame per entity, ids=DataFrame of accepted (entity, scheme, id), suggested=DataFrame per entity).

    ``qid``: coll_slug, entity_key, qid, qid_via, qid_n_cand, qid_score. The weight of a QID is the sum over its accepted pairs of
    (mentions + 0.01) x reliability of the scheme / number of items the id leads to (an id that leads to two items counts half)."""
    acc = pairs[pairs['status'] == 'accepted'].copy()
    key = ['coll_slug', 'entity_key']
    a = acc[acc['qid'].notna()].copy()
    a['w'] = (a['n_mentions'] + 0.01) * a['scheme'].map(REL).fillna(1.0) / a['k'].fillna(1).clip(lower=1)
    if len(a):
        a['qnum'] = a['qid'].str[1:].astype('int64')
        g = a.groupby(key + ['qid', 'qnum'], as_index=False).agg(
            qid_score=('w', 'sum'), qid_via=('scheme', lambda s: '+'.join(sorted(set(s)))))
        g['qid_n_cand'] = g.groupby(key)['qid'].transform('size')
        g = g.sort_values(key + ['qid_score', 'qnum'], ascending=[True, True, False, True]).drop_duplicates(key)
        qid = g[key + ['qid', 'qid_via', 'qid_n_cand', 'qid_score']]
    else:
        qid = pd.DataFrame(columns=key + ['qid', 'qid_via', 'qid_n_cand', 'qid_score'])
    # best suggested item for entities without an accepted one
    sg = pairs[(pairs['status'] == 'suggested') & pairs['qid'].notna()]
    if len(sg):
        sg = sg.sort_values(key + ['n_mentions'], ascending=[True, True, False]).drop_duplicates(key)
        sg = sg[key + ['qid', 'reason']].rename(columns={'qid': 'qid_suggested', 'reason': 'qid_suggested_reason'})
    else:
        sg = pd.DataFrame(columns=key + ['qid_suggested', 'qid_suggested_reason'])
    ids = acc.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id'], as_index=False)['n_mentions'].max()
    # an id accepted as an id of an item whose pair was accepted; the id itself (GND / GeoNames record) accepted separately
    return {'qid': qid, 'ids': ids, 'suggested': sg}


def write_log(pairs: pd.DataFrame, path: Path) -> pd.DataFrame:
    lg = pairs[pairs['status'].isin(['suggested', 'rejected', 'ignored'])].copy()
    cols = ['coll_slug', 'kind', 'entity_key', 'name', 'scheme', 'id', 'qid', 'qid_raw', 'src', 'status', 'reason', 'level', 'label',
            'second_id', 'n_mentions', 'n_in_period', 'detail']
    for c in cols:
        if c not in lg:
            lg[c] = None
    lg = lg[cols].sort_values(['status', 'reason', 'coll_slug', 'n_mentions'], ascending=[True, True, True, False])
    lg.to_csv(path, sep='\t', index=False)
    _t(f'gate log: {len(lg)} rows -> {path}')
    return lg
