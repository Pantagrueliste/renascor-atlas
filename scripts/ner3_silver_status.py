"""Progress of the RenNER silver teacher run: passages done per language and teacher, recent rate, rough time left.

    python3 scripts/ner3_silver_status.py [--dir RQ/derived/atlas/ner/v2/renner/silver]

Reads <dir>/full_raw/<lang>__<teacher>.jsonl (good lines = no transport error), the pool sizes, and the lane log
<dir>/logs/full/full.runs.log (start/end/quota lines).
"""
import argparse, json, os, time
from pathlib import Path

LANES = [('es', 'gemma4'), ('it', 'gemma4'), ('la', 'gemma4'), ('la', 'kimi-k3'), ('la', 'glm-5.3-flash')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2/renner/silver')
    a = ap.parse_args()
    d = Path(a.dir)
    now = time.time()
    sizes = {l: sum(1 for _ in open(d / f'pool_{l}.jsonl', encoding='utf8')) for l in ('es', 'it', 'la')}
    print(f'{time.strftime("%F %T")}  pool: ' + ', '.join(f'{l} {n}' for l, n in sizes.items()))
    left_s = 0.0
    for lang, t in LANES:
        f = d / 'full_raw' / f'{lang}__{t}.jsonl'
        ok, last_h, err = set(), 0, 0
        if f.exists():
            for line in open(f, encoding='utf8'):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('error') in (None, 'invalid_json'):
                    ok.add(r['id'])
                    if r.get('t_end') and now - r['t_end'] < 3600:
                        last_h += 1
                else:
                    err += 1
        n, tot = len(ok), sizes[lang]
        eta = f'{(tot - n) / last_h:.1f} h at the last-hour rate' if last_h else 'no call in the last hour'
        print(f'  {lang} {t:14s} {n:6d}/{tot} ({100 * n / tot:5.1f} %)  last hour {last_h:5d}  transport errors {err}  left: {eta}')
    log = d / 'logs' / 'full' / 'full.runs.log'
    if log.exists():
        print('lane log (last 5 lines):')
        for line in open(log, encoding='utf8').read().splitlines()[-5:]:
            print('  ' + line)
    pid = d / 'logs' / 'full' / 'full.pid'
    if pid.exists():
        p = pid.read_text().strip()
        alive = os.system(f'kill -0 {p} 2>/dev/null') == 0
        print(f'daemon pid {p}: {"running" if alive else "NOT running"}')
    else:
        print('daemon: no pid file (finished or stopped)')


if __name__ == '__main__':
    main()
