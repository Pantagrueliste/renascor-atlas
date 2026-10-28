"""Merge the per-model prediction files of gold2 into one predictions.jsonl (kept apart from passages.jsonl: annotators never see it).

Input: a directory with ``<model>.jsonl.gz`` (ner_audit_predict.py / ner_audit_zs.py / ner2_predict_llm.py output). Only LOC spans are kept.
Output record per passage:
  {"id", "lang",
   "models": {model: [{"s","e","cs","ce","text","conf"}, ...]},       # word span [s,e), character span [cs,ce), conf = span probability (null for the LLM)
   "consensus": [{"cs","ce","text","models":[...],"n_encoders":k,"max_conf":x}, ...]}   # spans grouped by exact character span, all models
A model that was not run on a passage (flair outside English) has no key in "models". Also writes predictions_meta.json.

    python scripts/ner2_merge_predictions.py --passages passages.jsonl --pred-dir pred --out predictions.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ENCODERS = ['r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_xlmr_large_H', 'r3_hmteams_HM']
OTHERS = ['flair_ner_english_large', 'llm_qwen35_9b']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passages', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--meta', default='')
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.passages, encoding='utf-8') if l.strip()]
    by_id = {r['id']: r for r in recs}
    pred: dict[str, dict[str, list]] = defaultdict(dict)
    counts = {}
    for m in ENCODERS + OTHERS:
        f = Path(a.pred_dir) / f'{m}.jsonl.gz'
        if not f.exists():
            counts[m] = None
            continue
        n = 0
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                d = json.loads(line)
                if d['id'] not in by_id:
                    continue
                spans = []
                for x in d['LOC']:
                    spans.append({'s': x['s'], 'e': x['e'], 'cs': x.get('cs'), 'ce': x.get('ce'), 'text': x['text'], 'conf': x.get('conf')})
                pred[d['id']][m] = spans
                n += 1
        counts[m] = n
    with open(a.out, 'w', encoding='utf-8') as out:
        for r in recs:
            ms = pred.get(r['id'], {})
            groups: dict[tuple, dict] = {}
            for m, spans in ms.items():
                for x in spans:
                    key = (x['cs'], x['ce'])
                    g = groups.setdefault(key, {'cs': x['cs'], 'ce': x['ce'], 'text': x['text'], 'models': [], 'max_conf': None})
                    g['models'].append(m)
                    if x['conf'] is not None and (g['max_conf'] is None or x['conf'] > g['max_conf']):
                        g['max_conf'] = x['conf']
            cons = []
            for g in sorted(groups.values(), key=lambda g: (g['cs'] if g['cs'] is not None else -1, g['ce'] if g['ce'] is not None else -1)):
                g['n_encoders'] = sum(m in ENCODERS for m in g['models'])
                cons.append(g)
            out.write(json.dumps({'id': r['id'], 'lang': r['lang'], 'models': ms, 'consensus': cons}, ensure_ascii=False) + '\n')
    meta = {'n_passages': len(recs), 'records_per_model': counts, 'encoders': ENCODERS, 'others': OTHERS,
            'note': 'LOC spans only. conf = mean softmax probability of the span (encoders), flair span score, null for the zero-shot LLM.'}
    Path(a.meta or str(Path(a.out).with_name('predictions_meta.json'))).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
