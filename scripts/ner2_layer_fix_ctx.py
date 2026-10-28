"""NER layer v2 fix, step 1 (CPU, Hypatia, debug partition): per-form facts that need the spans or the texts.

    python scripts/ner2_layer_fix_ctx.py --layer ~/renascor-atlas-data/ner/layer_v1 --out ~/renascor-atlas-data/ner/layer_v1/fix [--peek 15]

Writes in --out:
    form_pper.parquet   (run, lang, key, n, p_per): mean person probability of the recommended spans of each cleaned form
                        (run v1: confidence >= 0.98 and agreement >= 2, all languages but Spanish is also kept; run es: es_mdeberta, confidence >= 0.5 and agreement >= 2)
    ctx.parquet         (run, doc_id, lang, key, rule, n_total, n_match): for the forms of the context rules (see ``ner2_layer_fix.CONTEXT_RULES``), per document, how many
                        mentions have the cue in the text around them
    titles.json         doc_id -> first-line title, for every document of ``doc_forms.parquet`` of both runs
    peek_<rule>.jsonl   (--peek N) N random mentions per rule with +-90 characters of text and the verdict, to tune the cue lists (read on Hypatia)
No text leaves Hypatia except the small peek files read by the owner of the run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner2_layer_geo_prep import clean_surface, reason  # noqa: E402
import ner2_layer_fix as FX  # noqa: E402
import ner2_layer_geo_points as GP  # noqa: E402


def _t(m):
    print(f'[fix-ctx {time.strftime("%H:%M:%S")}] {m}', file=sys.stderr, flush=True)


def load_spans(path: Path, min_conf: float):
    s = pq.read_table(path, columns=['doc_id', 'surface', 'conf', 'n_agree', 'lang', 'p_per', 'char_start', 'char_end']).to_pandas()
    s = s[(s['conf'] >= min_conf) & (s['n_agree'] >= 2)].copy()
    cm = {u: clean_surface(u) for u in s['surface'].unique()}
    s['form'] = s['surface'].map(cm)
    s['why'] = s['surface'].map({u: reason(c) for u, c in cm.items()})
    s = s[s['why'] == ''].copy()
    s['key'] = s['form'].str.casefold()
    return s


def scan_texts(args):
    path, needed = args
    out = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            i = line.find('"id": "')
            if i < 0:
                continue
            j = line.find('"', i + 7)
            did = line[i + 7:j]
            if did in needed and did not in out:
                out[did] = json.loads(line)['t']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--peek', type=int, default=0)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    L = Path(a.layer).expanduser()
    O = Path(a.out).expanduser()
    O.mkdir(parents=True, exist_ok=True)
    runs = {'v1': (L / 'spans.parquet', 0.98, L / 'geo/doc_forms.parquet'),
            'es': (L / 'es_mdeberta/spans.parquet', 0.5, L / 'es_mdeberta/geo/doc_forms.parquet')}
    pp, ctx_rows, all_docs = [], [], set()
    work = {}
    for run, (sp, mc, dfp) in runs.items():
        s = load_spans(sp, mc)
        _t(f'{run}: {len(s)} recommended spans')
        g = s.groupby(['lang', 'key']).agg(n=('p_per', 'size'), p_per=('p_per', 'mean')).reset_index()
        g.insert(0, 'run', run)
        pp.append(g)
        all_docs |= set(pd.read_parquet(dfp, columns=['doc_id'])['doc_id'])
        tg = s[s['key'].map(FX.context_rule_of).notna()].copy()
        tg['rule'] = tg['key'].map(FX.context_rule_of)
        work[run] = tg
        _t(f'{run}: {len(tg)} spans of the context rules {tg["rule"].value_counts().to_dict()}')
    pd.concat(pp).to_parquet(O / 'form_pper.parquet', index=False)
    need = set()
    for tg in work.values():
        need |= set(tg['doc_id'])
    _t(f'{len(need)} documents to read for the context rules')
    shards = sorted((L / 'extract/texts').glob('*/*.jsonl.gz'))
    texts = {}
    with mp.Pool(a.procs) as pool:
        for r in pool.imap_unordered(scan_texts, [(str(p), need) for p in shards]):
            for k, v in r.items():
                texts.setdefault(k, v)
    _t(f'{len(texts)} texts read')
    rng = np.random.default_rng(a.seed)
    peek = {}
    bad = 0
    for run, tg in work.items():
        for r in tg.itertuples():
            t = texts.get(r.doc_id)
            if t is None:
                bad += 1
                ctx_rows.append((run, r.doc_id, r.lang, r.key, r.rule, 1, -1))
                continue
            cs, ce = int(r.char_start), int(r.char_end)
            before, after = t[max(0, cs - 120):cs], t[ce:ce + 120]
            ok = FX.apply_context_rule(r.rule, before, after)
            ctx_rows.append((run, r.doc_id, r.lang, r.key, r.rule, 1, int(ok)))
            if a.peek:
                peek.setdefault(r.rule, []).append((run, r.lang, r.doc_id, t[cs:ce], before[-90:], after[:90], bool(ok)))
    _t(f'mentions without text: {bad}')
    c = pd.DataFrame(ctx_rows, columns=['run', 'doc_id', 'lang', 'key', 'rule', 'n_total', 'n_match'])
    c = c.groupby(['run', 'doc_id', 'lang', 'key', 'rule'], as_index=False).agg(n_total=('n_total', 'sum'), n_match=('n_match', lambda x: int((x == 1).sum())), n_notext=('n_match', lambda x: int((x == -1).sum())))
    c.to_parquet(O / 'ctx.parquet', index=False)
    _t(f'ctx rows {len(c)}')
    for rule, lst in peek.items():
        idx = rng.choice(len(lst), size=min(a.peek, len(lst)), replace=False)
        with open(O / f'peek_{rule}.jsonl', 'w', encoding='utf-8') as fh:
            for i in idx:
                run, lg, did, sf, b, af, ok = lst[i]
                fh.write(json.dumps({'run': run, 'lang': lg, 'doc': did, 'surface': sf, 'ok': ok, 'ctx': (b + '[[' + sf + ']]' + af).replace('\n', ' / ')}, ensure_ascii=False) + '\n')
    # titles
    shards_all = shards
    titles = {}
    with mp.Pool(a.procs) as pool:
        for r in pool.imap_unordered(GP.scan_shard, [(str(p), all_docs) for p in shards_all]):
            for k, v in r.items():
                titles.setdefault(k, v)
    (O / 'titles.json').write_text(json.dumps(titles, ensure_ascii=False), 'utf-8')
    _t(f'titles {len(titles)} / {len(all_docs)} documents')
    print('done')


if __name__ == '__main__':
    main()
