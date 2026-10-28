"""Diagnostic: the same fine-tuned GLiNER on 400 dev passages with GLiNER's own word splitter vs whitespace-only (our tokens).
   python scripts/ner_gliner_diag.py MODEL_DIR DEV_JSONL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from gliner import GLiNER
from ratlas.ner import nerdata as nd
from ratlas.ner.zeroshot import SpaceSplitter, filter_threshold, gliner_predict, score_span_preds

recs = nd.read_slim(sys.argv[2], limit=400)
m = GLiNER.from_pretrained(sys.argv[1], map_location='cpu').to('cuda' if torch.cuda.is_available() else 'cpu').eval()
labels = {'LOC': ['place'], 'PER': ['person']}
for name in ('default', 'space'):
    if name == 'space':
        m.data_processor.words_splitter = SpaceSplitter()
    preds = gliner_predict(m, recs, labels, 100, 0.1, 8)
    for t in (0.1, 0.3, 0.5):
        r = score_span_preds(recs, filter_threshold(preds, t)).report()
        print(name, t, 'LOC', r['LOC']['overall']['p'], r['LOC']['overall']['r'], r['LOC']['overall']['f1'], 'PER', r['PER']['overall']['f1'], flush=True)
