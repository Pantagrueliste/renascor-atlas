"""Query the Hugging Face Hub API (public, no token) for model/dataset cards: licence, size, tags, gating, custom code.

usage: python3 scripts/ner_external_hfmeta.py OUT.tsv  models|datasets  id1 id2 ...
Only reads public metadata (one GET per id, 0.5 s apart). Nothing is uploaded.
"""
import json, sys, time, urllib.request, urllib.error

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'renascor-atlas-research/0.1'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {'_error': f'HTTP {e.code}'}
    except Exception as e:
        return {'_error': str(e)[:80]}

def main(out, kind, ids):
    rows = []
    for i in ids:
        d = get(f'https://huggingface.co/api/{kind}/{i}')
        time.sleep(0.5)
        if '_error' in d:
            rows.append([i, d['_error']] + [''] * 9); continue
        tags = d.get('tags', [])
        lic = next((t[8:] for t in tags if t.startswith('license:')), '')
        langs = [t for t in tags if len(t) <= 3 and t.isalpha() and t.islower()]
        cd = d.get('cardData') or {}
        sib = d.get('siblings') or []
        has_py = [s['rfilename'] for s in sib if s['rfilename'].endswith('.py')]
        st = d.get('safetensors') or {}
        rows.append([i, 'ok', lic or str(cd.get('license', '')), str(d.get('gated', False)), d.get('createdAt', '')[:10],
                     d.get('lastModified', '')[:10], str(st.get('total', '') or ''), str(d.get('downloads', '')),
                     d.get('pipeline_tag', '') or d.get('library_name', ''), ','.join(has_py), ' '.join(langs)[:120]])
    hdr = ['id', 'status', 'license', 'gated', 'created', 'modified', 'safetensors_params', 'downloads', 'pipeline_or_lib', 'py_files', 'lang_tags']
    with open(out, 'w') as f:
        f.write('\t'.join(hdr) + '\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
