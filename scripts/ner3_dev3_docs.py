"""dev3: list every document the dev3 passages must not come from: v1 train/dev/test, test_v2, gold v2 (final + seed + candidates). One name per line."""
import json, sys, glob, gzip
out = sys.argv[1]
names = set()
def stream(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as f:
        for l in f:
            if l.strip():
                yield json.loads(l)
for p in sys.argv[2:]:
    n = 0
    for d in stream(p):
        if d.get('doc'):
            names.add(str(d['doc']))
            n += 1
    print(p, n, flush=True)
open(out, 'w').write('\n'.join(sorted(names)) + '\n')
print(len(names), 'document names')
