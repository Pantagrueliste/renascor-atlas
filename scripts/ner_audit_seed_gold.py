"""Build gold_seed_v1.jsonl (audit, task 2): the 30 passages of untagged texts with the place names found by the reviewer.

Inputs (scratch, produced during the audit): seed_passages.jsonl, seed_annotation_blind.json (first pass, made BEFORE looking at any model output:
{passage index: {core: [strings], amb: [strings]}}, every whole-word occurrence of a string is a mention). The adjudication of the union of the model predictions
added no new place name, so the final gold equals the blind pass. Output: one JSON per passage with character spans, BIO tags over the tokens, and the
status of the annotation (model-assisted reviewer, to be checked by a human).
"""
import argparse, json, re

LANG_FIX = {29: 'la'}      # a passage of an 'Italian' source that is in Latin
ap = argparse.ArgumentParser(); ap.add_argument('--passages'); ap.add_argument('--blind'); ap.add_argument('--out'); a = ap.parse_args()
P = [json.loads(l) for l in open(a.passages, encoding='utf-8')]
A = json.load(open(a.blind, encoding='utf-8'))
LIC = {'EEBO-TCP': 'public domain (EEBO-TCP, text archive T1)', 'EEBO-TCP-Latin': 'public domain (EEBO-TCP, text archive T1)',
       'Wikisource-IT-Renaissance': 'it.wikisource.org, CC BY-SA 3.0 / GFDL (text archive: nas__ws_it_renaissance)', 'Galileo-Favaro-Wikisource': 'it.wikisource.org, public domain text (Favaro edition) / CC BY-SA 3.0 (transcription)'}
with open(a.out, 'w', encoding='utf-8') as f:
    for i, p in enumerate(P):
        t = p['text']
        spans = []
        for cat, certainty in (('core', 'certain'), ('amb', 'ambiguous')):
            for s in sorted(set(A[str(i)][cat])):
                for m in re.finditer(r'(?<!\w)' + re.escape(s) + r'(?!\w)', t):
                    spans.append({'start': m.start(), 'end': m.end(), 'text': s, 'label': 'LOC', 'certainty': certainty})
        spans.sort(key=lambda x: x['start'])
        toks, ts = p['tokens'], p['token_starts']
        tags = [0] * len(toks)
        for sp in spans:
            if sp['certainty'] != 'certain':
                continue
            idx = [k for k in range(len(toks)) if ts[k] < sp['end'] and ts[k] + len(toks[k]) > sp['start']]
            for j, k in enumerate(idx):
                tags[k] = 1 if j == 0 else 2
        rec = {'id': p['id'], 'lang': LANG_FIX.get(i, p['lang']), 'lang_source_label': p['lang'], 'collection': p['collection'], 'doc': p['doc'], 'title': p['title'],
               'licence_note': LIC.get(p['collection'], ''), 'text': t, 'tokens': toks, 'token_starts': ts, 'spans': spans, 'ner_tags_loc': tags,
               'annotation': {'status': 'annotated by a model-assisted reviewer (Claude), blind first pass then adjudication against model predictions; TO BE CHECKED BY A HUMAN',
                              'labels': 'LOC only. certain = clear toponym (settlement, region, country, river, field name); ambiguous = demonym/adjective, place inside a title or epithet, church or building name, polity (not in ner_tags_loc)',
                              'boundaries': 'minimal toponym: lower-case generic head words (castle, house, isle) and articles are not part of the span',
                              'no_markup_in_source': True}}
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(len(P), 'passages written')
