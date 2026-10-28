"""Precision and recall from a manual silver review (counts_<lang>_<ver>.tsv), with Wilson 95% intervals and a passage
bootstrap (spans of one passage are not independent).

    python scripts/ner3_silver_review_stats.py counts_es_v1.tsv [counts_it_v1.tsv ...] [--json out.json]

Strict counting: precision = correct / (correct + wrong + boundary); recall = correct / (correct + boundary + missed).
A boundary error is a predicted span that overlaps a true entity with wrong limits: it is a false positive and a miss.
"doubt" spans (PER_doubt, LOC_doubt) are left out of both ratios.
"""
import argparse, csv, json, math, random


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def pr(rows, lab):
    c = sum(int(r[f'{lab}_correct']) for r in rows)
    w = sum(int(r[f'{lab}_wrong']) for r in rows)
    b = sum(int(r[f'{lab}_boundary']) for r in rows)
    m = sum(int(r[f'{lab}_missed']) for r in rows)
    return c, w, b, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsv', nargs='+')
    ap.add_argument('--json')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    out = {}
    for path in a.tsv:
        rows = list(csv.DictReader(open(path, encoding='utf8'), delimiter='\t'))
        name = path.rsplit('/', 1)[-1].replace('counts_', '').replace('.tsv', '')
        res = {'passages': len(rows)}
        for lab in ('LOC', 'PER'):
            c, w, b, m = pr(rows, lab)
            np_, nr = c + w + b, c + b + m
            rng = random.Random(1)
            ps, rs = [], []
            for _ in range(a.boot):
                s = [rng.choice(rows) for _ in rows]
                cc, ww, bb, mm = pr(s, lab)
                if cc + ww + bb:
                    ps.append(cc / (cc + ww + bb))
                if cc + bb + mm:
                    rs.append(cc / (cc + bb + mm))
            ps.sort(); rs.sort()
            q = lambda v: (round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)) if v else (None, None)
            res[lab] = {'correct': c, 'wrong': w, 'boundary': b, 'missed': m,
                        'precision': round(c / np_, 3) if np_ else None, 'precision_wilson95': wilson(c, np_), 'precision_boot95': q(ps),
                        'recall': round(c / nr, 3) if nr else None, 'recall_wilson95': wilson(c, nr), 'recall_boot95': q(rs)}
        res['PER']['doubt'] = sum(int(r.get('PER_doubt') or 0) for r in rows)
        res['LOC']['doubt'] = sum(int(r.get('LOC_doubt') or 0) for r in rows)
        out[name] = res
        for lab in ('LOC', 'PER'):
            x = res[lab]
            print(f"{name} {lab}: C {x['correct']} W {x['wrong']} B {x['boundary']} M {x['missed']} | "
                  f"P {x['precision']} Wilson {x['precision_wilson95']} boot {x['precision_boot95']} | "
                  f"R {x['recall']} Wilson {x['recall_wilson95']} boot {x['recall_boot95']}")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
