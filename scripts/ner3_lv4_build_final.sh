#!/bin/bash
# RenNER Latin verse routing: build silver/final_v4/ on the Mac from the prompt-v4 answers of the Latin verse and drama passages.
# combined/silver_la.jsonl (v3) -> v4 answers for form verse/drama (ner3_silver_combine.py, +pp) -> merge (ner3_lv4_merge.py)
# -> balanced training file with the same leak drops as silver/final (ner3_silver_final.py) -> same gold v3 withdrawal (ids of final/)
# -> es / it / reserve files copied unchanged. Nothing in silver/final/ or silver/combined/ is written. Gold v2 and gold v3 are never read.
#   bash scripts/ner3_lv4_build_final.sh
set -euo pipefail
R=/Users/clem/Projects/Papers/renascor-atlas; S=/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2/renner/silver
W=$S/final_v4_work; F4=$S/final_v4; PY="nice -n 19 $R/.venv/bin/python"
mkdir -p $W/raw $F4
cd $R
rsync -a hypatia:renascor-atlas-data/ner/silver/v4_full/raw/ $W/raw/
$PY scripts/ner3_silver_combine.py --passages $S/combined/silver_la.jsonl --raw "$W/raw/la_vd_v4*.jsonl" --only-done \
    --prompt-file 'scripts/renner_prompt_rules_en_v4.md + scripts/renner_fewshot_v4.jsonl' --out $W/silver_la_v4vd.jsonl --stats $W/stats_la_v4vd.json
$PY scripts/ner3_lv4_merge.py --old $S/combined/silver_la.jsonl --new $W/silver_la_v4vd.jsonl --out $W/silver_la_v4.jsonl --report $W/merge_la_v4.json
$PY scripts/ner3_silver_final.py --silver $W/silver_la_v4.jsonl --subset balanced --drop-ids $S/final/leak_drop_ids.jsonl \
    --out $W/silver_balanced_la.pre_gold3.jsonl --report $W/report_balanced_la_v4.json
# gold v3 withdrawal: keep exactly the ids of final/silver_balanced_la.jsonl, in the same order (checked)
$PY - "$S/final/silver_balanced_la.jsonl" "$W/silver_balanced_la.pre_gold3.jsonl" "$F4/silver_balanced_la.jsonl" "$W/check_final_v4.json" <<'PYEOF'
import json, sys
from collections import Counter
ref, src, out, rep = sys.argv[1:]
keep = [json.loads(l)['id'] for l in open(ref, encoding='utf8')]
ks = set(keep)
c = Counter(); got = []
with open(out, 'w', encoding='utf8') as f:
    for l in open(src, encoding='utf8'):
        d = json.loads(l)
        if d['id'] in ks:
            f.write(l if l.endswith('\n') else l + '\n'); got.append(d['id'])
        else:
            c['withdrawn_gold3'] += 1
assert got == keep, 'id sequence differs from final/'
# compare with final/: same text and tokens everywhere; spans changed only on verse / drama
for a, b in zip(open(ref, encoding='utf8'), open(out, encoding='utf8')):
    x, y = json.loads(a), json.loads(b)
    assert x['id'] == y['id'] and x['text'] == y['text'] and x['tokens'] == y['tokens'], x['id']
    sx = {(s['start'], s['end'], s['label']) for s in x['spans']}; sy = {(s['start'], s['end'], s['label']) for s in y['spans']}
    form = x.get('form')
    c[f'passages_{form}'] += 1
    c[f'changed_{form}'] += sx != sy
    for lab in ('LOC', 'PER'):
        c[f'{lab}_{form}_v3'] += sum(1 for s in sx if s[2] == lab); c[f'{lab}_{form}_v4'] += sum(1 for s in sy if s[2] == lab)
    c['teacher_v4'] += 'v4' in (y.get('teacher') or {}).get('prompt_files', '')
assert c['changed_prose'] == 0, 'a prose passage changed'
c['passages'] = len(keep)
json.dump(dict(sorted(c.items())), open(rep, 'w'), indent=1); print(json.dumps(dict(sorted(c.items()))))
PYEOF
for f in silver_balanced_es.jsonl silver_balanced_it.jsonl silver_reserve_es.jsonl silver_reserve_it.jsonl; do cp -p $S/final/$f $F4/$f; done
( cd $S/final && md5 silver_balanced_es.jsonl silver_balanced_it.jsonl silver_reserve_es.jsonl silver_reserve_it.jsonl silver_balanced_la.jsonl ) > $W/md5_final.txt
( cd $F4 && md5 silver_balanced_es.jsonl silver_balanced_it.jsonl silver_reserve_es.jsonl silver_reserve_it.jsonl silver_balanced_la.jsonl ) > $W/md5_final_v4.txt
diff <(head -4 $W/md5_final.txt) <(head -4 $W/md5_final_v4.txt) && echo "es/it identical"
cat $W/md5_final_v4.txt
