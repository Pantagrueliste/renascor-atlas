"""External comparison: spaCy pipelines as off-the-shelf NER (venv_spacy), same output format as scripts/ner2_zs.py.

    python -I scripts/ner3_extcmp_spacy.py --route es=es_core_news_lg,it=it_core_news_lg,fr=fr_core_news_lg,de=de_core_news_lg,en=en_core_web_lg,la=la_core_web_lg \
        --data passages.jsonl --out pred/zs_spacy_lg.jsonl.gz
    python -I scripts/ner3_extcmp_spacy.py --route '*=de_grand_tour_ner' --data ... --out ...

--route LANG=MODEL,...: the pipeline used for the passages of each language ('*' = every passage). A passage whose language has no pipeline gets no
prediction (reported in the log). The pipeline runs on the passage text; an entity is mapped to every word (tokens / token_starts) it touches,
as for GLiNER in the v1 audit. Labels: LOC, GPE, GEO, LOCATION -> LOC; PER, PERS, PERSON -> PER; the others (ORG, MISC, NORP, FAC ...) -> O.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

LOC = {'LOC', 'GPE', 'GEO', 'LOCATION', 'PLACE'}
PER = {'PER', 'PERS', 'PERSON', 'PRS'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--procs', type=int, default=1)
    a = ap.parse_args()
    import spacy
    route = dict(x.split('=', 1) for x in a.route.split(','))
    recs = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]
    by_model = defaultdict(list)
    skipped = Counter()
    for i, r in enumerate(recs):
        m = route.get(r.get('lang') or '') or route.get('*')
        if m:
            by_model[m].append(i)
        else:
            skipped[r.get('lang')] += 1
    log(len(recs), 'records; per model', {m: len(v) for m, v in by_model.items()}, 'no pipeline', dict(skipped))
    preds = [{'LOC': [], 'PER': []} for _ in recs]
    labels_seen = Counter()
    for m, idx in by_model.items():
        nlp = spacy.load(m, exclude=[p for p in ('parser', 'lemmatizer', 'tagger', 'morphologizer', 'attribute_ruler', 'trainable_lemmatizer', 'senter')])
        log('loaded', m, nlp.pipe_names, nlp.get_pipe('ner').labels if 'ner' in nlp.pipe_names else '')
        if hasattr(nlp, 'max_length'):
            nlp.max_length = 2_000_000
        texts = (recs[i]['text'] for i in idx)
        for i, doc in zip(idx, nlp.pipe(texts, batch_size=a.batch, n_process=a.procs)):
            r = recs[i]
            ts, toks = r['token_starts'], r['tokens']
            te = [s + len(t) for s, t in zip(ts, toks)]
            for e in doc.ents:
                labels_seen[(m, e.label_)] += 1
                task = 'LOC' if e.label_.upper() in LOC else ('PER' if e.label_.upper() in PER else None)
                if not task:
                    continue
                k0 = bisect.bisect_right(te, e.start_char)          # first word that ends after the start
                k1 = bisect.bisect_left(ts, e.end_char)             # words that start before the end
                if k0 < k1:
                    preds[i][task].append((k0, k1))
        log('done', m)
    log('labels', dict(labels_seen))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, 'wt', encoding='utf-8') as out:
        for r, p in zip(recs, preds):
            o = {'id': r['id']}
            ts, toks, text = r['token_starts'], r['tokens'], r.get('text', '')
            for task in ('LOC', 'PER'):
                lst = []
                for s, e in sorted(set(p[task])):
                    cs, ce = ts[s], ts[e - 1] + len(toks[e - 1])
                    lst.append({'s': s, 'e': e, 'conf': 1.0, 'cs': cs, 'ce': ce, 'text': text[cs:ce]})
                o[task] = lst
            out.write(json.dumps(o, ensure_ascii=False) + '\n')
    log('written', a.out)


if __name__ == '__main__':
    main()
