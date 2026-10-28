"""NER layer v3, after the GPU jobs (CPU, Hypatia): assemble the spans of the TEI editions and of the repaired extra documents, drop the spans that overlap an editor tag,
and prepare the combined input of the geocoding steps.

    python scripts/ner2_layer_v3_merge.py --v3 ~/renascor-atlas-data/ner/layer_v3 [--sets tei extra]

For each set (``tei``, ``extra``) under --v3, the chunk files of ``<set>/spans`` (main run, XLM-R large) and ``<set>/es_mdeberta/spans`` (Spanish run, mDeBERTa) are put
together in ``<set>/spans.parquet`` with these added columns:

  recommended   non-Spanish: conf >= 0.98 and n_agree >= 2 (of the 4 encoders); Spanish: conf >= 0.5 and n_agree >= 2 (the setup of es_v2)
  ed_overlap    the span shares at least one character with a tag of the editors (``layer_tei/editor_tags.parquet``, same text, same offsets). Any tag counts: place, person,
                organisation or other (a name, a title of a work ...). TEI set only; the extra set has no editor tag in its texts.
  ed_tag, ed_type, ed_kind   the overlapping tag with the longest overlap (element, ``type``, kind place / person / org / other)
  rec_final     recommended and not ed_overlap: the spans that go to the geocoder

``<set>/docs.parquet`` is the documents table with the counts per document. ``geo/layer/`` receives the combined input of ner2_layer_geo_prep.py (``spans.parquet`` = rec_final
spans of both sets, ``docs.parquet``, ``extract/texts`` = links to the text shards for the titles). Writes the tables of counts in ``summary/``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MIN_CONF = {'default': 0.98, 'es': 0.5}
MIN_AGREE = 2


def is_recommended(conf: np.ndarray, n_agree: np.ndarray, lang: np.ndarray) -> np.ndarray:
    """The recommended rule of the layer: conf >= 0.98 for every language but Spanish (>= 0.5, mDeBERTa main), agreement of >= 2 encoders."""
    thr = np.where(np.asarray(lang) == 'es', MIN_CONF['es'], MIN_CONF['default'])
    ag = np.nan_to_num(np.asarray(n_agree, dtype='float64'), nan=0.0)
    return (np.asarray(conf, dtype='float64') >= thr) & (ag >= MIN_AGREE)


def flag_overlaps(spans: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """For every span (doc_id, char_start, char_end; end excluded) the editor tag of the same document that overlaps it by at least one character, with the longest overlap
    (ties: the first in the tag table). Returns a frame indexed like ``spans`` with ed_overlap (bool), ed_tag, ed_type, ed_kind (None where no overlap), ed_len (characters)."""
    n = len(spans)
    out = pd.DataFrame({'ed_overlap': np.zeros(n, bool), 'ed_tag': [None] * n, 'ed_type': [None] * n, 'ed_kind': [None] * n, 'ed_len': np.zeros(n, np.int32)}, index=spans.index)
    if n == 0 or len(tags) == 0:
        return out
    tg = {d: g for d, g in tags.groupby('doc_id', sort=False)}
    pos = pd.Series(np.arange(n), index=spans.index)
    hit = np.zeros(n, bool)
    tag_l = np.full(n, None, dtype=object)
    type_l = np.full(n, None, dtype=object)
    kind_l = np.full(n, None, dtype=object)
    len_l = np.zeros(n, np.int32)
    sub = spans[spans['doc_id'].isin(tg.keys())]
    for d, g in sub.groupby('doc_id', sort=False):
        t = tg[d]
        ts, te = t['char_start'].to_numpy(np.int64), t['char_end'].to_numpy(np.int64)
        ss, se = g['char_start'].to_numpy(np.int64)[:, None], g['char_end'].to_numpy(np.int64)[:, None]
        ov = np.minimum(se, te[None, :]) - np.maximum(ss, ts[None, :])        # > 0: at least one character in common
        best = ov.argmax(axis=1)
        bl = ov[np.arange(len(g)), best]
        ok = bl > 0
        if not ok.any():
            continue
        ix = pos[g.index[ok]].to_numpy()
        hit[ix] = True
        len_l[ix] = bl[ok]
        tag_l[ix] = t['tag'].to_numpy(object)[best[ok]]
        type_l[ix] = t['type'].fillna('').to_numpy(object)[best[ok]]
        kind_l[ix] = t['tag_kind'].to_numpy(object)[best[ok]]
    out['ed_overlap'] = hit
    out['ed_tag'] = tag_l
    out['ed_type'] = type_l
    out['ed_kind'] = kind_l
    out['ed_len'] = len_l
    return out


def read_spans(d: Path) -> pd.DataFrame:
    files = sorted((d / 'spans').glob('*.parquet')) if (d / 'spans').exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def done_docs(d: Path) -> set:
    out = set()
    if (d / 'chunks').exists():
        for f in sorted((d / 'chunks').glob('*.docs')):
            out.update(f.read_text().split())
    return out


def build_set(v3: Path, name: str, tags: pd.DataFrame | None, docs_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    S = v3 / name
    main = read_spans(S)
    es = read_spans(S / 'es_mdeberta')
    sp = pd.concat([x for x in (main, es) if len(x)], ignore_index=True)
    sp['n_agree'] = sp['n_agree'].astype('float64')
    sp['recommended'] = is_recommended(sp['conf'].to_numpy(), sp['n_agree'].to_numpy(), sp['lang'].to_numpy())
    if tags is not None:
        f = flag_overlaps(sp, tags)
        for c in f.columns:
            sp[c] = f[c].to_numpy()
    else:
        sp['ed_overlap'] = False
        sp['ed_tag'] = None
        sp['ed_type'] = None
        sp['ed_kind'] = None
        sp['ed_len'] = np.int32(0)
    sp['rec_final'] = sp['recommended'] & ~sp['ed_overlap']
    sp['set'] = name
    sp.to_parquet(S / 'spans.parquet', index=False, compression='zstd')
    docs = pd.read_parquet(docs_path or Path(os.path.expanduser('~/renascor-atlas-data/ner')) / ('layer_tei/docs.parquet' if name == 'tei' else 'layer_extra/extract/docs_read.parquet'))
    if name == 'extra':
        docs = docs[docs['status'] == 'ok']
    ran = done_docs(S) | done_docs(S / 'es_mdeberta')
    g = sp.groupby('doc_id').agg(n_spans=('conf', 'size'), n_recommended=('recommended', 'sum'), n_dropped_ed=('ed_overlap', 'sum'), n_rec_final=('rec_final', 'sum')).reset_index()
    docs = docs.merge(g, on='doc_id', how='left')
    for c in ('n_spans', 'n_recommended', 'n_dropped_ed', 'n_rec_final'):
        docs[c] = docs[c].fillna(0).astype(int)
    dr = sp[sp['recommended'] & sp['ed_overlap']].groupby('doc_id').size()
    docs['n_rec_dropped_ed'] = docs['doc_id'].map(dr).fillna(0).astype(int)
    docs['run'] = docs['doc_id'].isin(ran)
    docs['set'] = name
    docs.to_parquet(S / 'docs.parquet', index=False)
    return sp, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3', type=Path, required=True)
    ap.add_argument('--sets', nargs='*', default=['tei', 'extra', 'seen'])
    ap.add_argument('--geo-name', default='geo', help='folder of v3 that receives the combined geocoder input')
    ap.add_argument('--tei-dir', type=Path, default=Path.home() / 'renascor-atlas-data/ner/layer_tei')
    ap.add_argument('--extra-dir', type=Path, default=Path.home() / 'renascor-atlas-data/ner/layer_extra')
    a = ap.parse_args()
    v3 = a.v3.expanduser()
    tei = a.tei_dir.expanduser()
    tagf = {'tei': tei / 'editor_tags.parquet', 'seen': tei / 'seen' / 'editor_tags.parquet'}
    docf = {'seen': tei / 'seen' / 'docs.parquet'}
    allsp, alldocs = [], []
    for nm in a.sets:
        sp, docs = build_set(v3, nm, pd.read_parquet(tagf[nm]) if nm in tagf else None, docf.get(nm))
        print(nm, len(sp), 'spans', int(sp['recommended'].sum()), 'recommended', int(sp['ed_overlap'].sum()), 'overlap a tag', int(sp['rec_final'].sum()), 'to the geocoder')
        allsp.append(sp[sp['rec_final']])
        alldocs.append(docs)
    # combined input of the geocoding steps (ner2_layer_geo_prep.py reads spans.parquet and docs.parquet of --layer)
    G = v3 / a.geo_name / 'layer'
    G.mkdir(parents=True, exist_ok=True)
    sp = pd.concat(allsp, ignore_index=True)
    keep = ['doc_id', 'passage', 'w_start', 'w_end', 'char_start', 'char_end', 'surface', 'conf', 'p_per', 'n_agree', 'n_overlap', 'lang', 'year', 'collection', 'set']
    sp[keep].to_parquet(G / 'spans.parquet', index=False, compression='zstd')
    dd = pd.concat(alldocs, ignore_index=True)
    dd.to_parquet(G / 'docs.parquet', index=False)
    tx = G / 'extract' / 'texts'
    srcs = {'tei': a.tei_dir.expanduser() / 'chunks', 'seen': a.tei_dir.expanduser() / 'seen' / 'chunks', 'extra': a.extra_dir.expanduser() / 'extract' / 'texts'}
    for nm in a.sets:
        for f in sorted(srcs[nm].glob('*/*.jsonl.gz')):
            lang = f.parent.name
            (tx / lang).mkdir(parents=True, exist_ok=True)
            link = tx / lang / f.name
            if not link.exists():
                os.symlink(f, link)
    print('combined', len(sp), 'spans for the geocoder,', len(dd), 'documents')


if __name__ == '__main__':
    main()
