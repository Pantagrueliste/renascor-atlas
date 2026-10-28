"""NER layer v1, independent check, step 1 (CPU, Hypatia): draw mentions of the layer and cut their passages.

    python scripts/ner2_layer_check_sample.py --layer ~/renascor-atlas-data/ner/layer_v1 --out DIR [--seed 7]

Draws 200 mentions (33-34 per language, half in points with <= 10 mentions, half in points with more), plus 8 mentions for each of the 30 largest points.
Writes DIR/sample.jsonl and DIR/top30.jsonl (one mention per line, with the point and a +-220 character context) and DIR/placed_forms_stats.parquet.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner2_layer_geo_prep import clean_surface, reason  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--per-top', type=int, default=8)
    a = ap.parse_args()
    L = Path(a.layer).expanduser()
    O = Path(a.out).expanduser()
    O.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    s = pq.read_table(L / 'spans.parquet').to_pandas()
    s['row'] = np.arange(len(s))
    s = s[(s['conf'] >= 0.98) & (s['n_agree'] >= 2)].copy()
    cm = {u: clean_surface(u) for u in s['surface'].unique()}
    s['form'] = s['surface'].map(cm)
    s['why'] = s['surface'].map({u: reason(c) for u, c in cm.items()})
    s = s[s['why'] == ''].copy()
    s['key'] = s['form'].str.casefold()
    F = pd.read_parquet(L / 'geo/points/forms_placed.parquet')
    F = F[F['pid'] >= 0]
    m = s.merge(F[['lang', 'key', 'pid', 'gc_source', 'gc_conf', 'gc_cand_name', 'gc_evidence']], on=['lang', 'key'], how='inner')
    print('layer mentions', len(m))
    pts = json.load(open(L / 'geo/points/points.json'))
    P = {int(p['id'].split(':')[1]): p for p in pts}
    assert 'Rome' == P[int(m[m['key'] == 'rome']['pid'].iloc[0])]['name'], 'pid != ml id'
    m['pn'] = m['pid'].map(lambda i: P[i]['n'])
    m['few'] = m['pn'] <= 10
    # per-form statistics for the systematic-error analysis
    st = m.groupby(['lang', 'key', 'pid']).agg(n=('row', 'size'), form=('form', 'first'), p_per=('p_per', 'mean'), conf=('conf', 'mean'), agree=('n_agree', 'mean'),
                                              docs=('doc_id', 'nunique'), cap=('form', lambda x: float(np.mean([f[:1].isupper() for f in x]))),
                                              src=('gc_source', 'first'), cname=('gc_cand_name', 'first')).reset_index()
    st['pname'] = st['pid'].map(lambda i: P[i]['name'])
    st['plat'] = st['pid'].map(lambda i: P[i]['lat'])
    st['plon'] = st['pid'].map(lambda i: P[i]['lon'])
    st['pn'] = st['pid'].map(lambda i: P[i]['n'])
    st.to_parquet(O / 'placed_forms_stats.parquet', index=False)
    picks = []
    sizes = {}
    for lg in ['en', 'it', 'fr', 'la', 'de', 'es']:
        for few in (True, False):
            d = m[(m['lang'] == lg) & (m['few'] == few)]
            k = 17 if (few == (lg != 'en')) or True else 17
            k = min(k, len(d))
            if lg == 'en' and not few:
                k = 17
            idx = rng.choice(d.index.to_numpy(), size=k, replace=False)
            sizes[f'{lg}_{"few" if few else "many"}'] = [int(len(d)), int(k)]
            picks.append(m.loc[idx].assign(stratum=f'{lg}_{"few" if few else "many"}', weight=len(d) / k))
    S = pd.concat(picks)
    # trim to exactly 200: drop 2 random from the en strata (6*2*17 = 204)
    drop = rng.choice(S.index[S['lang'] == 'en'].to_numpy() if False else np.arange(len(S))[S['lang'].to_numpy() == 'en'], size=len(S) - 200, replace=False)
    S = S.iloc[[i for i in range(len(S)) if i not in set(drop.tolist())]]
    print(len(S), sizes)
    top = sorted(P.values(), key=lambda p: -p['n'])[:30]
    T = []
    for p in top:
        d = m[m['pid'] == int(p['id'].split(':')[1])]
        T.append(d.loc[rng.choice(d.index.to_numpy(), size=min(a.per_top, len(d)), replace=False)].assign(stratum='top'))
    T = pd.concat(T)
    need = set(S['doc_id']) | set(T['doc_id'])
    print('docs needed', len(need))
    texts = {}
    for f in sorted((L / 'extract/texts').rglob('*.jsonl*')):
        op = gzip.open if f.suffix == '.gz' else open
        with op(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                if not any(True for _ in ()):    # keep cheap: parse only when the id may be needed
                    i = line.find('"id": "')
                    j = line.find('"', i + 7)
                    did = line[i + 7:j]
                    if did in need and did not in texts:
                        texts[did] = json.loads(line)['t']
        print(f.name, len(texts), flush=True)
        if len(texts) >= len(need):
            break

    def ctx(r, w=220):
        t = texts.get(r['doc_id'])
        if t is None:
            return None
        a_, b_ = int(r['char_start']), int(r['char_end'])
        return (t[max(0, a_ - w):a_] + '[[' + t[a_:b_] + ']]' + t[b_:b_ + w]).replace('\n', ' / ')

    for name, D in (('sample', S), ('top30', T)):
        with open(O / f'{name}.jsonl', 'w', encoding='utf-8') as fh:
            for _, r in D.iterrows():
                p = P[int(r['pid'])]
                o = {'stratum': r['stratum'], 'weight': float(r.get('weight', 0) or 0), 'lang': r['lang'], 'doc_id': r['doc_id'], 'year': int(r['year']) if r['year'] == r['year'] else None,
                     'collection': r['collection'], 'surface': r['surface'], 'p_per': float(r['p_per']), 'conf': float(r['conf']), 'agree': int(r['n_agree']),
                     'pid': int(r['pid']), 'pname': p['name'], 'lat': p['lat'], 'lon': p['lon'], 'k': p['k'], 'r': p.get('r'), 'pn': p['n'], 'gs': r['gc_source'],
                     'ctx': ctx(r)}
                fh.write(json.dumps(o, ensure_ascii=False) + '\n')
    print('done')


if __name__ == '__main__':
    main()
