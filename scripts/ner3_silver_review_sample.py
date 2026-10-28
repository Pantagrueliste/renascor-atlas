"""Draw a random review sample from a silver file and render it with inline annotations, for a manual quality estimate.

    python scripts/ner3_silver_review_sample.py --silver pilot_es.jsonl --n 40 --seed 7 --out review_es.md [--ids-out ids.txt]

Each passage is printed with its spans as [text|LABEL]; a numbered list of the spans follows, so that the reviewer can count
correct spans, wrong spans (type or not an entity), boundary errors and missed entities per passage.
"""
import argparse, json, random


def render(p):
    toks, starts, text = p['tokens'], p['token_starts'], p['text']
    out, pos = [], 0
    for s in sorted(p['spans'], key=lambda x: x['char_start']):
        out.append(text[pos:s['char_start']])
        out.append(f"[{text[s['char_start']:s['char_end']]}|{s['label']}]")
        pos = s['char_end']
    out.append(text[pos:])
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--silver', required=True)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.silver, encoding='utf8')]
    sample = random.Random(a.seed).sample(recs, min(a.n, len(recs)))
    with open(a.out, 'w', encoding='utf8') as f:
        for k, p in enumerate(sample, 1):
            f.write(f"## {k}. {p['id']} ({p.get('collection')}, {p.get('year')}, {p.get('form')}) status={p.get('teacher_status')}\n\n")
            f.write(render(p) + '\n\n')
            for j, s in enumerate(p['spans'], 1):
                f.write(f"  {j}. {s['label']} «{s['text']}» votes={s.get('votes')}\n")
            f.write('\n')
    print(f'{len(sample)} passages -> {a.out}')


if __name__ == '__main__':
    main()
