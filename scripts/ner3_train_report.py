"""NER v2 (RenNER) training grid: one table of the runs of scripts/ner3_train_multi.sh (steps, time, GPU hours, throughput, memory, best v1 dev).

    python scripts/ner3_train_report.py --tv ~/renascor-atlas-data/ner/train_v2 --runs scripts/ner3_train_v2_runs.tsv --out report.json

GPU hours: the time each run held its GPU, from the task logs of every job (start of the run to its exit, or to the end of the job when it was
killed: the last write in the job directory). Only the v1 dev is read here (dev3 and the other sets of the final evaluation stay in metrics.json); gold v2 is never read.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--tv', required=True)
ap.add_argument('--runs', required=True)
ap.add_argument('--out')
a = ap.parse_args()
tv = Path(a.tv)
F = '%Y-%m-%d %H:%M:%S'

names = []
for line in open(a.runs):
    line = line.strip()
    if line and not line.startswith('#'):
        p = line.split('|')
        names.append((p[0], p[1], p[2]))


def job_end(jd: Path):
    """End of a job, or now while it runs: the last write in its directory (the GPU loggers write once a minute)."""
    return datetime.fromtimestamp(max(f.stat().st_mtime for f in jd.rglob('*') if f.is_file()))


segs = {}   # run -> list of (job, start, end, rc)
for jd in sorted((tv / 'jobs').glob('*')):
    open_runs = {}
    for lg in sorted(jd.glob('task*.log')):
        for line in open(lg):
            m = re.match(r'\[task \d+\] (\S+ \S+) start (\S+)', line)
            if m:
                open_runs[m.group(2)] = datetime.strptime(m.group(1), F)
                continue
            m = re.match(r'\[task \d+\] (\S+ \S+) (\S+) exit (\d+)', line)
            if m and m.group(2) in open_runs:
                segs.setdefault(m.group(2), []).append((jd.name, open_runs.pop(m.group(2)), datetime.strptime(m.group(1), F), int(m.group(3))))
    if open_runs:
        end = job_end(jd)
        for r, t0 in open_runs.items():
            segs.setdefault(r, []).append((jd.name, t0, end, None))

rows = []
for name, mixv, model in names:
    rd = tv / 'runs' / name
    row = {'run': name, 'model': model.rsplit('/', 1)[-1], 'mix': {'MIX_SBAL': 'S-bal', 'MIX_SFULL': 'S-full'}.get(mixv, mixv),
           'done': (rd / 'done.json').exists(), 'jobs': sorted(set(s[0] for s in segs.get(name, [])))}
    row['gpu_h'] = round(sum((s[2] - s[1]).total_seconds() for s in segs.get(name, [])) / 3600, 2)
    row['wall_h'] = round((max(s[2] for s in segs[name]) - min(s[1] for s in segs[name])).total_seconds() / 3600, 2) if name in segs else 0
    perf = json.loads((rd / 'perf.json').read_text()) if (rd / 'perf.json').exists() else {}
    sg = perf.get('segments', [])
    row['steps_total'] = perf.get('steps_per_epoch', 0) * 3 or None
    row['windows_per_epoch'] = perf.get('windows_per_epoch')
    secs = sum(s.get('seconds', 0) for s in sg)
    row['train_h'] = round(secs / 3600, 2)
    row['windows_per_s'] = round(sum(s.get('windows', 0) for s in sg) / secs, 1) if secs else None
    row['tokens_per_s'] = round(sum(s.get('tokens', 0) for s in sg) / secs) if secs else None
    row['pad_share'] = round(1 - sum(s.get('tokens', 0) for s in sg) / max(1, sum(s.get('padded_tokens', 0) for s in sg)), 3) if secs else None
    row['peak_alloc_gib'] = max((s.get('peak_alloc_gb', 0) for s in sg), default=None)
    row['peak_reserved_gib'] = max((s.get('peak_reserved_gb', 0) for s in sg), default=None)
    hist = []
    if (rd / 'metrics.json').exists():
        m = json.loads((rd / 'metrics.json').read_text())
        hist = m.get('history', [])
        row['steps'] = m.get('steps')
        row['stop'] = m.get('stop')
        dv = m.get('sets', {}).get('dev', {})
        row['dev_loc_f1'] = dv.get('LOC', {}).get('overall', {}).get('f1')
        row['dev_per_f1'] = dv.get('PER', {}).get('overall', {}).get('f1')
    else:
        log = tv / 'run_logs' / f'{name}.log'
        steps = re.findall(r'step (\d+)/(\d+)', log.read_text()) if log.exists() else []
        row['steps'] = int(steps[-1][0]) if steps else 0
        row['steps_total'] = int(steps[-1][1]) if steps else row['steps_total']
        for x in re.findall(r'\[dev\] step (\d+) LOC F1 (\S+) PER F1 (\S+)', log.read_text()) if log.exists() else []:
            hist.append({'step': int(x[0]), 'dev_loc_f1': float(x[1]), 'dev_per_f1': float(x[2])})
        if hist:
            b = max(hist, key=lambda h: h['dev_loc_f1'])
            row['dev_loc_f1'], row['dev_per_f1'] = b['dev_loc_f1'], b['dev_per_f1']
    if hist:
        b = max(hist, key=lambda h: h['dev_loc_f1'])
        row['best_step'] = b['step']
    row['dev_history'] = [(h['step'], h['dev_loc_f1'], h.get('dev_per_f1')) for h in hist]
    md = tv / 'models' / name
    row['model_dir'] = str(md) if (md / 'model.pt').exists() else None
    row['ckpt'] = str(rd / 'ckpt.pt') if (rd / 'ckpt.pt').exists() else None
    rows.append(row)

cols = ['run', 'mix', 'done', 'steps', 'steps_total', 'wall_h', 'gpu_h', 'train_h', 'windows_per_s', 'tokens_per_s', 'peak_alloc_gib', 'dev_loc_f1', 'dev_per_f1', 'best_step', 'jobs']
print('\t'.join(cols))
for r in rows:
    print('\t'.join(str(r.get(c, '')) for c in cols))
if a.out:
    Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
