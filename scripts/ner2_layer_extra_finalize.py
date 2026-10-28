"""NER layer, extra documents: decide the duplicates, then lay out the documents to run exactly like layer_v1/extract.

    python scripts/ner2_layer_extra_finalize.py --read READ_DIR --dedup DEDUP_DIR --out LAYER_EXTRA [--threshold 0.3]

READ_DIR   output of ner2_layer_extra_extract.py (docs_parts/extra.jsonl, texts/)
DEDUP_DIR  layer/{matches,union}.tsv (against the layer), tagged_place/{matches,union}.tsv (against the files in which the editors tagged places), intra/matches.tsv
OUT        writes extract/docs_all.parquet (every document of the run with its final status), extract/docs_read.parquet (documents to run, status ok),
           extract/texts/<lang>/extra.<collection>.jsonl.gz, extract/shards.tsv, extract/plan/worklist.txt, dup_report.tsv

Rule (precision first): a document that is read (status ok) is a duplicate when at least ``threshold`` (0.3) of its sampled 8-word shingles occur in one
document of the reference, or in the union of the documents of the reference. References, in this order:
  dup_layer   the documents of the layer (all the documents of status ok of layer_v1, run or not)
  dup_tagged  the files in which the editors tagged places in the text (atlas v2.4 mentions of kind place, zone text / footnote / apparatus / commentary)
  dup_extra   a larger (or earlier, if equal) document of this very run that holds it
Why 0.3 and not 0.5: two editions of one work (spelling normalised against a print) share a third to two thirds of their 8-word shingles, so the test cannot tell them
from partial reuse with certainty; below 0.3 a document is kept. Every skipped document is listed with its containment in dup_report.tsv.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

STOP = {
    'en': 'the of and to in that is was he for it with as his on be at by had not are but from or have an they which one you were her all she there would their we him been has when who will no more if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people mr how too little state good very make world still own see men work long get here between both life being under never day same another know while last might us great old year off come since against go came right used take three'.split(),
    'la': 'et in est non ad cum ut quod qui quae quam sed si ex de per ab pro sunt esse enim autem nec aut vel quo sicut etiam nam ergo igitur hic haec hoc eius eum eos sua suo suis sunt fuit erat neque tamen quia quidem inter ita uel atque'.split(),
    'it': 'di e il la che in un per non è con si le del della da una i dei lo come ma più gli al alla sono nel anche suo sua se ha ho mi ci ne questo quella quello poi tutti tutto così molto dal delle nella degli cui loro sopra et li'.split(),
    'es': 'de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas'.split(),
    'fr': 'de la le et les des en un du que est pour qui dans une au il ne sur se pas plus par avec son sa ses ce cette mais ou comme sont nous vous leur leurs elle ils elles aux été être avait fait très tout tous bien sans sous cet entre aussi dont où si ni y lui même'.split(),
    'de': 'der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat daß dass sie nach wird bei einer um am sind noch wie einem über einen so zum war haben nur oder aber vor zur bis mehr durch man sein wurde sei vnd daß dem jhr'.split(),
}
STOP = {k: frozenset(v) for k, v in STOP.items()}
WORD = re.compile(r'\w+', re.U)


def guess_lang(text: str, n_words: int = 4000) -> dict:
    """Share of function words of each language in the first ``n_words`` words: a rough, cheap language check (not a classifier)."""
    w = [x.lower() for x in WORD.findall(text[:60000])][:n_words]
    if not w:
        return {k: 0.0 for k in STOP}
    return {k: sum(1 for x in w if x in s) / len(w) for k, s in STOP.items()}


def lang_flag(scores: dict, lang: str, margin: float = 1.5) -> str:
    """'' when the catalogue language scores best (or close to it); the other language when it scores ``margin`` times better and at least 0.08."""
    best = max(scores, key=scores.get)
    if best == lang or scores[best] < 0.08:
        return ''
    if scores[lang] * margin > scores[best]:
        return ''
    return best


def decide(ok_names, nsh, layer, tagged, intra, threshold=0.3):
    """-> dict name -> (status, ref, containment, reference-set note). ``layer`` / ``tagged``: dict with 'max' {cand: (contain, ref)} and 'union' {cand: contain};
    ``intra``: {cand: [(other, contain_cand)]}. nsh: {cand: n sampled shingles}."""
    out = {}
    for n in ok_names:
        for key, st in ((layer, 'dup_layer'), (tagged, 'dup_tagged')):
            mx = key['max'].get(n, (0.0, ''))
            un = key['union'].get(n, 0.0)
            if mx[0] >= threshold or un >= threshold:
                out[n] = (st, mx[1] if mx[0] >= threshold else '(union of several documents)', max(mx[0], un))
                break
    alive = [n for n in ok_names if n not in out]
    alive.sort(key=lambda n: (-nsh.get(n, 0), n))
    kept = set()
    for n in alive:
        hit = [(o, c) for o, c in intra.get(n, []) if o in kept and c >= threshold]
        if hit:
            o, c = max(hit, key=lambda x: x[1])
            out[n] = ('dup_extra', o, c)
        else:
            kept.add(n)
    return out


def load_matches(d: Path):
    mx, un = {}, {}
    p = d / 'matches.tsv'
    if p.exists():
        m = pd.read_csv(p, sep='\t')
        if len(m):
            m = m.sort_values('contain_cand', ascending=False).drop_duplicates('cand')
            mx = {r.cand: (float(r.contain_cand), r.ref) for r in m.itertuples()}
    p = d / 'union.tsv'
    if p.exists():
        u = pd.read_csv(p, sep='\t')
        un = dict(zip(u['cand'], u['union_contain'].astype(float)))
    return {'max': mx, 'union': un}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--read', required=True)
    ap.add_argument('--dedup', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--cands', default='', help='cands.npz (for the shingle counts); default DEDUP/cands.npz')
    a = ap.parse_args()
    R, D, O = Path(a.read), Path(a.dedup), Path(a.out)
    rows = [json.loads(l) for l in open(R / 'docs_parts' / 'extra.jsonl', encoding='utf-8')]
    df = pd.DataFrame(rows)
    ok = df[df['status'] == 'ok']
    import numpy as np
    z = np.load(a.cands or str(D / 'cands.npz'), allow_pickle=True)
    nsh = dict(zip(list(z['names']), [int(x) for x in z['nsh']]))
    layer = load_matches(D / 'layer')
    tagged = load_matches(D / 'tagged_place')
    intra = defaultdict(list)
    ip = D / 'intra' / 'matches.tsv'
    if ip.exists():
        for r in pd.read_csv(ip, sep='\t').itertuples():
            intra[r.cand].append((r.other, float(r.contain_cand)))
    dec = decide(list(ok['doc_id']), nsh, layer, tagged, intra, a.threshold)
    df['dup_of'] = df['doc_id'].map(lambda d: dec.get(d, ('', '', 0.0))[1])
    df['dup_contain'] = df['doc_id'].map(lambda d: dec.get(d, ('', '', 0.0))[2])
    df['status'] = [dec[d][0] if (s == 'ok' and d in dec) else s for d, s in zip(df['doc_id'], df['status'])]
    # partial overlaps (kept): the largest containment in the layer or the tagged references
    df['layer_contain'] = df['doc_id'].map(lambda d: max(layer['max'].get(d, (0.0, ''))[0], layer['union'].get(d, 0.0)))
    df['tagged_contain'] = df['doc_id'].map(lambda d: max(tagged['max'].get(d, (0.0, ''))[0], tagged['union'].get(d, 0.0)))
    # texts: keep the documents that stay ok; language check on them
    keep = set(df.loc[df['status'] == 'ok', 'doc_id'])
    guess = {}
    for p in sorted((R / 'texts').rglob('*.jsonl.gz')):
        lang = p.parent.name
        outp = O / 'extract' / 'texts' / lang / p.name
        outp.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, 'rt', encoding='utf-8') as f, gzip.open(outp, 'wt', encoding='utf-8', compresslevel=3) as g:
            for line in f:
                d = json.loads(line)
                if d['id'] in keep:
                    g.write(line)
                    guess[d['id']] = lang_flag(guess_lang(d['t']), lang)
    df['lang_flag'] = df['doc_id'].map(lambda d: guess.get(d, ''))
    ex = O / 'extract'
    df.to_parquet(ex / 'docs_all.parquet', index=False)
    read = df[df['status'] == 'ok'].drop(columns=['dup_of', 'dup_contain'])
    read.to_parquet(ex / 'docs_read.parquet', index=False)
    df[df['status'].str.startswith('dup_')][['doc_id', 'collection', 'status', 'dup_of', 'dup_contain', 'n_words']].sort_values(['status', 'collection', 'doc_id']).to_csv(
        O / 'dup_report.tsv', sep='\t', index=False)
    shards = []
    for p in sorted((ex / 'texts').rglob('*.jsonl.gz')):
        lang = p.parent.name
        coll = p.name[len('extra.'):-len('.jsonl.gz')]
        sub = read[(read['collection'] == coll) & (read['lang'] == lang)]
        if len(sub):
            shards.append((str(p.relative_to(ex)), lang, len(sub), int(sub['n_words'].sum()), int(sub['n_tok'].sum())))
    with open(ex / 'shards.tsv', 'w') as f:
        f.write('file\tlang\tn_docs\tn_words\tn_tok\n')
        for s_ in shards:
            f.write('\t'.join(map(str, s_)) + '\n')
    (ex / 'plan').mkdir(exist_ok=True)
    order = []
    for lang in ('la', 'it', 'fr', 'de', 'en'):                                        # es has its own main model (run_es) and its own list
        order += [s[0] for s in shards if s[1] == lang]
    (ex / 'plan' / 'worklist.txt').write_text('\n'.join(order) + '\n')
    (ex / 'plan' / 'worklist_es.txt').write_text('\n'.join(s[0] for s in shards if s[1] == 'es') + '\n')
    print(df['status'].value_counts().to_dict())
    print(len(read), 'documents to run,', int(read['n_words'].sum()), 'words')


if __name__ == '__main__':
    main()
