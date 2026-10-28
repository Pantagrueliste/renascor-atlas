"""Check the pre-downloaded models: file integrity, safetensors headers, config + tokenizer load (no weights are loaded, no torch), subword fertility.

    python -I scripts/ner_external_verify_models.py [--cache ~/renascor-atlas-data/ner/hf_cache] [--out VERIFY.tsv] [--no-sha]

Run with ~/renascor-atlas-data/ner/venv_eval (transformers + tokenizers, no torch) on a compute node (slurm/ner_external_verify.sbatch).
  integrity   every file is compared with the Hub: sha256 of the content against the LFS sha256 of the pinned revision (one public API call per repo), or the
              byte size for the small non-LFS files (blob names of the cache are Xet hashes, not sha256, so they cannot be used);
  safetensors the JSON header is parsed and the declared tensor byte ranges must cover the data section exactly;
  config      AutoConfig (trust_remote_code only when config.json has an auto_map, the code having been read: see external_survey.md) and AutoTokenizer;
  fertility   subword tokens per whitespace word on six short samples (la, it, fr, de, en, nl): lower = fewer split words = better for place names.
GLiNER / flair / spaCy / GLiNER2 repos are only integrity-checked (they need their own library, installed later in a throwaway venv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

SAMPLES = {
    'la': 'Caesar in Galliam profectus est et apud Lutetiam Parisiorum concilium Galliae indixit, ubi Treveri et Aedui convenerunt.',
    'it': 'Il governatore di Firenze scrisse da Pisa al duca di Ferrara che le galee genovesi erano entrate nel porto di Livorno.',
    'fr': "Le roi partit de Fontainebleau pour Paris, puis gagna Lyon où l'attendait l'ambassadeur de Venise.",
    'de': 'Der Kurfürst zog von Heidelberg nach Frankfurt am Main und ließ die Truppen bei Mainz über den Rhein setzen.',
    'en': 'The ambassador sailed from Plymouth to Lisbon and then rode overland to Madrid, where the King of Spain received him.',
    'nl': 'De koopman vertrok uit Amsterdam naar Batavia en liet zijn goederen te Middelburg achter.',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def st_header_ok(path: Path) -> str:
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if n > 100_000_000:
            return 'header-too-big'
        hdr = json.loads(f.read(n))
    size = path.stat().st_size - 8 - n
    spans = sorted((v['data_offsets'][0], v['data_offsets'][1]) for k, v in hdr.items() if k != '__metadata__')
    pos = 0
    for a, b in spans:
        if a != pos:
            return f'gap@{a}'
        pos = b
    return 'ok' if pos == size else f'size {pos} != {size}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(Path.home() / 'renascor-atlas-data/ner/hf_cache'))
    ap.add_argument('--out', default='')
    ap.add_argument('--no-sha', action='store_true')
    a = ap.parse_args()
    cache = Path(a.cache)
    os.environ['HF_HOME'] = str(cache / 'home')
    from huggingface_hub import HfApi  # noqa: PLC0415
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415
    api = HfApi()
    rows = []
    for d in sorted(cache.glob('models--*')):
        rid = d.name[len('models--'):].replace('--', '/')
        snaps = sorted((d / 'snapshots').glob('*'))
        if not snaps:
            continue
        snap = snaps[-1]
        t0 = time.time()
        files = [p for p in snap.rglob('*') if p.is_file() or p.is_symlink()]
        try:
            meta = {x.rfilename: x for x in api.model_info(rid, revision=snap.name, files_metadata=True).siblings}
        except Exception as e:  # noqa: BLE001
            meta = {}
            print('   API', rid, str(e)[:100], flush=True)
        bad, nlfs, nst, stbad = [], 0, 0, []
        for p in files:
            tgt = p.resolve()
            rel = str(p.relative_to(snap))
            if not tgt.exists():
                bad.append(rel + ':missing')
                continue
            m = meta.get(rel)
            lfs = getattr(m, 'lfs', None) if m is not None else None
            if lfs is not None and getattr(lfs, 'sha256', None):
                nlfs += 1
                if not a.no_sha and sha256(tgt) != lfs.sha256:
                    bad.append(rel + ':sha256')
            elif m is not None and getattr(m, 'size', None) is not None and tgt.stat().st_size != m.size:
                bad.append(rel + ':size')
            if p.name.endswith('.safetensors'):
                nst += 1
                r = st_header_ok(tgt)
                if r != 'ok':
                    stbad.append(f'{p.name}:{r}')
        cfg = 'n/a'
        tok = 'n/a'
        fert = {}
        vocab = ''
        cj = snap / 'config.json'
        if cj.exists():
            try:
                c = json.loads(cj.read_text())
                trc = bool(c.get('auto_map'))
                try:
                    conf = AutoConfig.from_pretrained(str(snap), local_files_only=True, trust_remote_code=trc)
                    cfg = f'ok{"+remote_code" if trc else ""}:{getattr(conf, "model_type", "")}'
                except Exception as e:  # noqa: BLE001
                    msg = str(e).replace('\n', ' ')
                    cfg = ('needs-library: ' if 'does not recognize' in msg or 'requires the following packages' in msg else 'FAIL ') + msg[:100]
                try:
                    t = AutoTokenizer.from_pretrained(str(snap), local_files_only=True, trust_remote_code=trc)
                    tok = f'ok:{type(t).__name__}'
                    vocab = str(len(t))
                    for lg, s in SAMPLES.items():
                        ids = t(s, add_special_tokens=False)['input_ids']
                        fert[lg] = round(len(ids) / len(s.split()), 2)
                except Exception as e:  # noqa: BLE001
                    msg = str(e).replace('\n', ' ')
                    tok = ('needs-library: ' if 'requires the following packages' in msg else 'FAIL ') + msg[:100]
            except Exception as e:  # noqa: BLE001
                cfg = 'FAIL ' + str(e)[:80]
        gb = sum(p.resolve().stat().st_size for p in files if p.exists()) / 1e9
        rows.append([rid, snap.name, f'{gb:.2f}', str(len(files)), str(nlfs), ';'.join(bad) or 'ok', str(nst), ';'.join(stbad) or ('ok' if nst else 'n/a'), cfg, tok, vocab,
                     ' '.join(f'{k}={v}' for k, v in fert.items()), f'{time.time() - t0:.0f}s'])
        print('\t'.join(rows[-1]), flush=True)
    hdr = ['id', 'revision', 'GB', 'files', 'lfs_files', 'sha256', 'safetensors_files', 'safetensors_header', 'config', 'tokenizer', 'vocab', 'fertility', 'time']
    out = Path(a.out) if a.out else cache.parent / 'VERIFY.tsv'
    with open(out, 'w') as f:
        f.write('\t'.join(hdr) + '\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print('wrote', out)
    return 1 if any(r[5] != 'ok' for r in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
