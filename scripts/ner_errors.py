"""Error analysis of a run on the held-out test editions, from runs/<run>/pred_test.jsonl.gz.

   python scripts/ner_errors.py RUNDIR TEST_JSONL EDITION|all [LOC|PER] [N] [OUT_JSON]

For one edition: the most frequent false positives / negatives (surface strings). With ``all``: per edition and label the counts, and the share of false positives
whose string is tagged elsewhere in the same edition (a sign that the gold tagging of that edition is incomplete: the model finds a name the editors tagged in other
places), and the share of false negatives whose first token is a lone letter (split initials) -> written to OUT_JSON.
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas.ner.spans import tags_to_spans

run, test, edition = sys.argv[1:4]
task_arg = sys.argv[4] if len(sys.argv) > 4 else 'LOC'
n = int(sys.argv[5]) if len(sys.argv) > 5 else 25
out_json = sys.argv[6] if len(sys.argv) > 6 else None

preds = {}
with gzip.open(f'{run}/pred_test.jsonl.gz', 'rt') as f:
    for l in f:
        d = json.loads(l)
        preds[d['id']] = {t: set(map(tuple, d[t])) for t in ('LOC', 'PER')}

data = defaultdict(lambda: {'LOC': defaultdict(Counter), 'PER': defaultdict(Counter)})   # edition -> task -> kind -> Counter(surface)
gold_vocab = defaultdict(lambda: {'LOC': set(), 'PER': set()})
with open(test) as f:
    for l in f:
        r = json.loads(l)
        t = r['tokens']
        for task, tk, vk in (('LOC', 'ner_tags_loc', 'loc_valid'), ('PER', 'ner_tags_per', 'per_valid')):
            if not r.get(vk):
                continue
            g = set(tags_to_spans(r[tk]))
            p = preds.get(r['id'], {}).get(task, set())
            e = r['edition']
            for s, en in g:
                gold_vocab[e][task].add(' '.join(t[s:en]))
            for s, en in p - g:
                data[e][task]['fp'][' '.join(t[s:en])] += 1
            for s, en in g - p:
                data[e][task]['fn'][' '.join(t[s:en])] += 1
            for s, en in p & g:
                data[e][task]['tp'][' '.join(t[s:en])] += 1

if edition != 'all':
    d = data[edition][task_arg]
    print(edition, task_arg, {k: sum(v.values()) for k, v in d.items()})
    for k in ('fp', 'fn', 'tp'):
        print(k.upper(), d[k].most_common(n))
else:
    summary = {}
    for e in sorted(data):
        for task in ('LOC', 'PER'):
            d = data[e][task]
            fp, fn, tp = (sum(d[k].values()) for k in ('fp', 'fn', 'tp'))
            if fp + fn + tp == 0:
                continue
            fp_known = sum(c for s, c in d['fp'].items() if s in gold_vocab[e][task])
            fn_initial = sum(c for s, c in d['fn'].items() if s.split(' ')[0].__len__() == 1 and len(s.split(' ')) > 1)
            summary.setdefault(e, {})[task] = {'tp': tp, 'fp': fp, 'fn': fn, 'fp_string_tagged_elsewhere_in_edition': fp_known,
                                               'share_fp_tagged_elsewhere': round(fp_known / fp, 3) if fp else None,
                                               'fn_starting_with_lone_letter': fn_initial, 'share_fn_lone_letter': round(fn_initial / fn, 3) if fn else None}
    print(json.dumps(summary, indent=1))
    if out_json:
        Path(out_json).write_text(json.dumps(summary, indent=1))
