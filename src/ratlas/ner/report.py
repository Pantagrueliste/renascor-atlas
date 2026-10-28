"""Comparison tables of all NER runs (metrics_v1.md) from ``runs/*/metrics.json``.

    python -m ratlas.ner.report --runs RQ/derived/atlas/ner/runs --out RQ/derived/atlas/ner/tables_v1 [--narrative file.md]

Every number is the strict span F1 (exact boundaries, one label at a time, micro over all spans) unless the column says macro.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEST_ED_LOC = ['elan_crenum', 'hainhofer_reisen', 'kaiserin_eleonora_magdalena', 'koenigsfelden', 'reed', 'sozinianische_briefwechsel']
LANGS = ['de', 'en', 'fr', 'it', 'la']


def load_runs(root: Path) -> dict:
    runs = {}
    for d in sorted(root.iterdir()):
        f = d / 'metrics.json'
        if f.exists() and not d.name.startswith('smoke') and not d.name.endswith('_llm') and not d.name.endswith('_1200') and 'zs_llm' not in d.name:
            try:
                runs[d.name] = json.loads(f.read_text())
            except Exception:
                pass
    for name in [n for n in runs if n.endswith('_hi')]:       # the run with the wider threshold grid replaces the one with the narrow grid
        runs.pop(name[:-3], None)
        runs[name[:-3]] = runs.pop(name)
    return dict(sorted(runs.items()))


def get(run, set_name, task, *path, default=None):
    d = run.get('sets', {}).get(set_name, {}).get(task)
    for p in path:
        if d is None:
            return default
        d = d.get(p) if isinstance(d, dict) else None
    return default if d is None else d


def f(x, nd=3, pct=False):
    if x is None:
        return '-'
    return f'{x * 100:.1f}' if pct else f'{x:.{nd}f}'


def kind_of(name, run):
    k = run.get('kind')
    if k:
        return {'zeroshot': 'zero-shot', 'dictionary': 'dictionary', 'gliner_finetuned': 'GLiNER fine-tuned', 'llm_zeroshot': 'LLM zero-shot'}.get(k, k)
    return 'fine-tuned'


def md(header, rows, align=None):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def trained_on(name, r):
    mix = r.get('mix')
    if mix:
        return ' + '.join(f"{m['set']} ({m['share'] * 100:.0f}%)" if len(mix) > 1 else m['set'] for m in mix)
    return ''


def model_of(name, r):
    from .runinfo import INFO
    if name in INFO:
        return INFO[name][0]
    return r.get('model', '').replace('/home/cag30/renascor-atlas-data/ner/local_models/', '')


def main_table(runs, meta):
    rows = []
    for name, r in sorted(runs.items(), key=lambda kv: (-(get(kv[1], 'test', 'LOC', 'overall', 'f1', default=-1)))):
        rows.append([f'`{name}`', kind_of(name, r), model_of(name, r), trained_on(name, r),
                     f(get(r, 'dev', 'LOC', 'overall', 'f1')), f(get(r, 'test', 'LOC', 'overall', 'p')), f(get(r, 'test', 'LOC', 'overall', 'r')),
                     f(get(r, 'test', 'LOC', 'overall', 'f1')), f(get(r, 'test', 'LOC', 'macro_edition_f1')),
                     f(get(r, 'dev', 'PER', 'overall', 'f1')), f(get(r, 'test', 'PER', 'overall', 'f1')),
                     f(get(r, 'ext_hipe', 'LOC', 'overall', 'f1')), f(get(r, 'ext_hipe', 'PER', 'overall', 'f1'))])
    return md(['run', 'type', 'model', 'trained on', 'LOC dev F1', 'LOC test P', 'LOC test R', 'LOC test F1', 'LOC test macro-ed F1', 'PER dev F1', 'PER test F1',
               'HIPE LOC F1', 'HIPE PER F1'], rows)


def edition_table(runs, task='LOC', eds=TEST_ED_LOC, set_name='test'):
    rows = []
    for name, r in sorted(runs.items(), key=lambda kv: -(get(kv[1], set_name, task, 'overall', 'f1', default=-1))):
        rows.append([f'`{name}`'] + [f(get(r, set_name, task, 'by_edition', e, 'f1')) for e in eds] + [f(get(r, set_name, task, 'overall', 'f1'))])
    return md(['run'] + eds + ['all'], rows)


def lang_table(runs, task='LOC', set_name='test'):
    rows = []
    for name, r in sorted(runs.items(), key=lambda kv: -(get(kv[1], set_name, task, 'overall', 'f1', default=-1))):
        rows.append([f'`{name}`'] + [f(get(r, set_name, task, 'by_lang', l, 'f1')) for l in LANGS] + [f(get(r, set_name, task, 'overall', 'f1'))])
    return md(['run'] + LANGS + ['all'], rows)


HIPE_SUBSETS = ['hipe2022/hipe2020/fr', 'hipe2022/hipe2020/de', 'hipe2022/hipe2020/en', 'hipe2022/newseye/fr', 'hipe2022/newseye/de', 'hipe2022/newseye/fi',
                'hipe2022/newseye/sv', 'hipe2022/letemps/fr', 'hipe2022/sonar/de', 'hipe2022/topres19th/en', 'hipe2022/ajmc/fr', 'hipe2022/ajmc/de', 'hipe2022/ajmc/en']


def ext_table(runs, set_name, task, subsets=None):
    """Rows: runs; columns: the sub-corpora of an external test set (by_edition keys), LOC or PER F1."""
    keys = subsets
    if keys is None:
        ks = set()
        for r in runs.values():
            ks |= set(r.get('sets', {}).get(set_name, {}).get(task, {}).get('by_edition', {}))
        keys = sorted(ks)
    rows = []
    for name, r in sorted(runs.items(), key=lambda kv: -(get(kv[1], set_name, task, 'overall', 'f1', default=-1))):
        if set_name not in r.get('sets', {}):
            continue
        rows.append([f'`{name}`'] + [f(get(r, set_name, task, 'by_edition', k, 'f1')) for k in keys] + [f(get(r, set_name, task, 'overall', 'f1'))])
    short = [k.split('/', 1)[1] if k.startswith('hipe2022/') else k for k in keys]
    return md(['run'] + short + ['all'], rows)


def mix_table(runs):
    rows = []
    for name, r in sorted(runs.items()):
        mix = r.get('mix')
        if not mix:
            continue
        tot = sum(m['words'] for m in mix)
        rows.append([f'`{name}`', f'{tot / 1e6:.1f}'] + ['; '.join(f"{m['set']} {m['words'] / 1e6:.2f} M ({m['share'] * 100:.0f}%)" for m in mix)])
    return md(['run', 'train words (M)', 'mix (words and share)'], rows)


def gap_table(runs):
    rows = []
    for name, r in sorted(runs.items(), key=lambda kv: -(get(kv[1], 'test', 'LOC', 'overall', 'f1', default=-1))):
        if r.get('kind') or 'history' not in r:
            continue
        d, t = get(r, 'dev', 'LOC', 'overall', 'f1'), get(r, 'test', 'LOC', 'overall', 'f1')
        dp, tp = get(r, 'dev', 'PER', 'overall', 'f1'), get(r, 'test', 'PER', 'overall', 'f1')
        rows.append([f'`{name}`', f(d), f(t), f(None if d is None or t is None else d - t), f(dp), f(tp), f(None if dp is None or tp is None else dp - tp),
                     r.get('train_minutes', ''), r.get('gpu', '')])
    return md(['run', 'LOC dev', 'LOC test', 'gap', 'PER dev', 'PER test', 'gap', 'train min', 'GPU'], rows)


PUBLISHED_HIPE = [   # HIPE-2022 overview (CEUR-WS 3180, table 7): strict micro F1, all entity types, best system / XLM-R baseline
    ('hipe2020', 'fr', 0.808, 0.757), ('hipe2020', 'de', 0.794, 0.703), ('hipe2020', 'en', 0.620, 0.477), ('letemps', 'fr', 0.661, None), ('sonar', 'de', 0.529, 0.307),
    ('topres19th', 'en', 0.787, 0.764), ('newseye', 'fr', 0.656, 0.654), ('newseye', 'de', 0.477, None), ('newseye', 'fi', 0.644, None), ('newseye', 'sv', 0.651, None),
    ('ajmc', 'fr', 0.842, 0.741), ('ajmc', 'de', 0.934, 0.818), ('ajmc', 'en', 0.854, 0.736)]


def published_hipe_table():
    rows = [[f'{d}/{l}', f'{b:.3f}', '-' if x is None else f'{x:.3f}'] for d, l, b, x in PUBLISHED_HIPE]
    return md(['HIPE-2022 test set', 'best system, strict F1, all types', 'XLM-R base baseline'], rows)


def sample_scores(root: Path, suffix: str):
    """System -> {LOC: prf, PER: prf} on an LLM sample, from runs/<run>/metrics_eval_<suffix>.json (fine-tuned), runs/<run>_<suffix>/metrics.json (off the shelf) and the LLM runs."""
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        fp = d / f'metrics_eval_{suffix}.json'
        if not fp.exists() and suffix == 'llm':
            fp = d / 'metrics_eval_llm_sample.json'
        if fp.exists():
            out[d.name] = json.loads(fp.read_text())['sets']['llm_sample']
        elif d.name.endswith('_' + suffix) and (d / 'metrics.json').exists():
            out[d.name[:-len(suffix) - 1]] = json.loads((d / 'metrics.json').read_text())['sets']['llm_sample']
    return out


def llm_table(root: Path, suffix: str, llm_runs: list):
    sc = sample_scores(root, suffix)
    for r in llm_runs:
        fp = root / r / 'metrics.json'
        if fp.exists():
            sc[r] = json.loads(fp.read_text())['sets']['llm_sample']
    rows = []
    for name, v in sorted(sc.items(), key=lambda kv: -kv[1]['LOC']['overall']['f1']):
        L, P = v['LOC'], v['PER']['overall']
        eds = L['by_edition']
        rows.append([f'`{name}`', f"{v['n_records']}", f(L['overall']['p']), f(L['overall']['r']), f(L['overall']['f1']), f(L['macro_edition_f1']),
                     f(P['p']), f(P['r']), f(P['f1'])])
    return md(['system', 'passages', 'LOC P', 'LOC R', 'LOC F1', 'LOC macro-ed F1', 'PER P', 'PER R', 'PER F1'], rows)


def seeds_table(runs: dict):
    import statistics as st
    groups = {}
    for name, r in runs.items():
        base = name.rsplit('_s', 1)[0] if name.rsplit('_s', 1)[-1].isdigit() and '_s' in name else name
        groups.setdefault(base, []).append(name)
    rows = []
    for base, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        def stat(task, s, k='f1'):
            xs = [get(runs[n], s, task, 'overall', k) for n in names]
            xs = [x for x in xs if x is not None]
            return f'{st.mean(xs):.3f} +- {st.pstdev(xs):.3f}' if len(xs) > 1 else '-'
        rows.append([f'`{base}`', len(names), stat('LOC', 'dev'), stat('LOC', 'test'), stat('PER', 'test'), stat('LOC', 'ext_hipe'), stat('LOC', 'ext_hist')])
    return md(['config', 'seeds', 'LOC dev F1', 'LOC test F1', 'PER test F1', 'HIPE LOC F1', 'hist LOC F1'], rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--meta', help='JSON file: run -> {model, train}')
    args = ap.parse_args(argv)
    runs = load_runs(Path(args.runs))
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'main.md').write_text(main_table(runs, meta))
    (out / 'edition_loc.md').write_text(edition_table(runs, 'LOC'))
    (out / 'edition_per.md').write_text(edition_table(runs, 'PER', ['elan_crenum', 'elan_fnvr', 'hainhofer_reisen', 'kaiserin_eleonora_magdalena', 'koenigsfelden', 'reed',
                                                                     'sozinianische_briefwechsel']))
    (out / 'lang_loc.md').write_text(lang_table(runs, 'LOC'))
    (out / 'lang_per.md').write_text(lang_table(runs, 'PER'))
    (out / 'hipe_loc.md').write_text(ext_table(runs, 'ext_hipe', 'LOC', HIPE_SUBSETS))
    (out / 'hipe_per.md').write_text(ext_table(runs, 'ext_hipe', 'PER', HIPE_SUBSETS))
    (out / 'hist_loc.md').write_text(ext_table(runs, 'ext_hist', 'LOC'))
    (out / 'hist_per.md').write_text(ext_table(runs, 'ext_hist', 'PER'))
    (out / 'charters_loc.md').write_text(ext_table(runs, 'ext_charters', 'LOC'))
    (out / 'mix.md').write_text(mix_table(runs))
    (out / 'gap.md').write_text(gap_table(runs))
    (out / 'published_hipe.md').write_text(published_hipe_table())
    (out / 'seeds.md').write_text(seeds_table(runs))
    root = Path(args.runs)
    for suffix in ('llm', '1200'):
        try:
            (out / f'llm_{suffix}.md').write_text(llm_table(root, suffix, ['zs_llm_qwen35_9b', 'zs_llm_qwen25_7b'] if suffix == 'llm' else ['zs_llm_qwen35_9b_1200', 'zs_llm_qwen25_7b_1200']))
        except Exception as e:
            print('llm table', suffix, e)
    print(f'{len(runs)} runs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
