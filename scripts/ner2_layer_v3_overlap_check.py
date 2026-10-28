"""Independent check (Hypatia, CPU): no span of the geocoder input shares a character with an editor tag, and no (doc, start, end) is twice in it.

    python scripts/ner2_layer_v3_overlap_check.py --v3 ~/renascor-atlas-data/ner/layer_v3 --data ~/renascor-atlas-data/ner --out FILE.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner2_layer_v3_merge as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--v3', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
V3, D = Path(a.v3).expanduser(), Path(a.data).expanduser()
s = pd.read_parquet(V3 / 'geo_all/layer/spans.parquet', columns=['doc_id', 'char_start', 'char_end', 'set', 'collection'])
tags = pd.concat([pd.read_parquet(D / 'layer_tei/editor_tags.parquet'), pd.read_parquet(D / 'layer_tei/seen/editor_tags.parquet')], ignore_index=True)
tags = tags.drop_duplicates(['doc_id', 'char_start', 'char_end', 'tag'])
ov = M.flag_overlaps(s, tags)
s2 = s.sort_values(['doc_id', 'char_start'])
inner = int((s2.groupby('doc_id')['char_start'].shift(-1) < s2['char_end']).sum())
res = {'spans': int(len(s)), 'tags': int(len(tags)), 'tag_docs': int(tags['doc_id'].nunique()), 'spans_overlapping_a_tag': int(ov['ed_overlap'].sum()),
       'duplicate_spans': int(s.duplicated(['doc_id', 'char_start', 'char_end']).sum()), 'spans_overlapping_another_span': inner,
       'spans_in_tagged_docs': int(s['doc_id'].isin(set(tags['doc_id'])).sum()), 'by_set': {k: int(v) for k, v in s['set'].value_counts().items()}}
Path(a.out).write_text(json.dumps(res, indent=1), 'utf-8')
print(res)
