"""Pre-download the candidate model weights from the Hugging Face Hub (public repos, no token, nothing is uploaded, nothing is run).

    HF_HOME=... python -I scripts/ner_external_models.py [--cache DIR] [--only id1,id2] [--role encoder|offshelf|hist|llm|latin] [--list]

Runs on a Hypatia compute node (slurm/ner_external_models.sbatch) with ~/renascor-atlas-data/ner/venv_hf (huggingface_hub only; no transformers is
imported, so no model code is executed).  For every repo: the revision (commit sha), the licence tag, the size, the files kept, whether the repo
carries custom code (any .py file, or an auto_map in config.json: then trust_remote_code would be needed) -> MODELS.tsv.  The .py files of such repos
are copied to <cache>/../model_code/<id>/ for a manual read BEFORE trust_remote_code is ever enabled.

Kept: everything small, plus safetensors (or, if none, the .bin / .pt weights); skipped: onnx, openvino, gguf, tf, flax, h5, msgpack, coreml, images.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# role: stub = config + tokenizer of a backbone only; encoder = to fine-tune; offshelf = used as is (zero-shot); hist = historical/HIPE-trained, used as is; latin; llm = zero-shot LLM
MODELS = [
    # ---- encoders to fine-tune ---------------------------------------------------------------------------------------------------------
    ('FacebookAI/xlm-roberta-base', 'encoder', ''),
    ('FacebookAI/xlm-roberta-large', 'encoder', ''),
    ('microsoft/mdeberta-v3-base', 'encoder', ''),
    ('jhu-clsp/mmBERT-base', 'encoder', 'multilingual ModernBERT, 1800 languages, 2025'),
    ('jhu-clsp/mmBERT-small', 'encoder', ''),
    ('EuroBERT/EuroBERT-210m', 'encoder', 'custom modeling code (Llama-style encoder): read before trust_remote_code'),
    ('EuroBERT/EuroBERT-610m', 'encoder', ''),
    ('EuroBERT/EuroBERT-2.1B', 'encoder', 'NER v2 large encoders (2026-10-23): custom code, read before trust_remote_code'),
    ('facebook/xlm-roberta-xl', 'encoder', 'NER v2 large encoders (2026-10-23): 3.5B, native transformers class'),
    ('dbmdz/bert-base-historic-multilingual-64k-td-cased', 'encoder', 'hmBERT 64k'),
    ('dbmdz/bert-base-historic-multilingual-cased', 'encoder', 'hmBERT 32k (base of the impresso baseline)'),
    ('hmteams/teams-base-historic-multilingual-discriminator', 'encoder', 'hmTEAMS'),
    ('bowphs/LaBerta', 'encoder', 'Latin RoBERTa'),
    # ---- generic off-the-shelf NER, zero-shot -------------------------------------------------------------------------------------------
    ('urchade/gliner_multi-v2.1', 'offshelf', 'GLiNER multilingual'),
    ('gliner-community/gliner_large-v2.5', 'offshelf', 'GLiNER large v2.5 (English)'),
    ('knowledgator/gliner-x-large', 'offshelf', 'GLiNER-X multilingual 2025'),
    ('knowledgator/gliner-x-base', 'offshelf', ''),
    ('fastino/gliner2-multi-v1', 'offshelf', 'GLiNER2 multilingual'),
    ('fastino/gliner2.5-multi-v1', 'offshelf', 'GLiNER2.5 multilingual, 2026'),
    ('VAGOsolutions/SauerkrautLM-GLiNER', 'offshelf', 'en de fr it es'),
    ('numind/NuNER_Zero', 'offshelf', 'English'),
    ('numind/NuNER-multilingual-v0.1', 'offshelf', ''),
    ('Babelscape/wikineural-multilingual-ner', 'offshelf', 'CC BY-NC-SA: private use only'),
    ('Davlan/xlm-roberta-large-ner-hrl', 'offshelf', 'AFL-3.0'),
    ('flair/ner-multi', 'offshelf', 'flair pickle'),
    ('flair/ner-english-large', 'offshelf', 'flair pickle'),
    ('flair/ner-german-large', 'offshelf', 'flair pickle'),
    # ---- historical NER models, used as is -----------------------------------------------------------------------------------------------
    ('impresso-project/ner-hipe2020-mmbert-base-final', 'hist', 'mmBERT-base fine-tuned on HIPE-2022 (fr de en), 2026, CC BY-NC-SA'),
    ('impresso-project/ner-hipe2020-hist-base', 'hist', 'hmBERT fine-tuned on HIPE-2022'),
    ('impresso-project/ner-stacked-bert-multilingual', 'hist', 'impresso production model, AGPL, custom code'),
    ('emanuelaboros/historical-ner-baseline', 'hist', 'hmBERT HIPE baseline'),
    ('pjox/dalembert-classical-fr-ner', 'hist', 'Early Modern French (FreEM-NER, 16th-18th c.), flair, Apache-2.0: test F1 pers .9317 loc .9288 (COLING 2022 table 4)'),
    ('pjox/camembert-classical-fr-ner', 'hist', 'same corpus, CamemBERT base, flair: pers .9304 loc .9254'),
    ('globalise/globalise-NER', 'hist', 'early modern Dutch (VOC), MIT'),
    ('aendriu/bert-ner-italian-historical', 'hist', 'Italian'),
    ('NikosKprl/en_deberta_v3_base_ner_historical_place', 'hist', 'English places'),
    ('NikosKprl/en_deberta_v3_base_ner_historical_people', 'hist', 'English people'),
    ('dell-research-harvard/historical_newspaper_ner', 'hist', 'English newspapers'),
    ('ERCDiDip/medieval-latin-ner', 'latin', 'medieval Latin charters, XLM-R base student'),
    ('GhentCDH/Latin-hagiography-NER', 'latin', 'Latin hagiography'),
    ('latincy/la_core_web_lg', 'latin', 'spaCy Latin pipeline (wheel), MIT; needs spacy: a wheel is code, install only in a throwaway venv'),
] + [(f'hmbert-64k/flair-hipe-2022-{d}', 'hist', 'Schweter hmBench flair model (hmBERT 64k)') for d in
     ('hipe2020-fr', 'hipe2020-de', 'newseye-fr', 'newseye-de', 'newseye-fi', 'newseye-sv', 'topres19th-en', 'letemps-fr')] + [
    ('hmbert-64k/flair-icdar-nl', 'hist', 'Schweter hmBench flair model, Europeana Dutch'),
    ('hmbert-64k/flair-icdar-fr', 'hist', 'Schweter hmBench flair model, Europeana French'),
    # ---- backbones that GLiNER configs name (config + tokenizer only, no weights: GLiNER builds the encoder from the config, the weights come from the GLiNER repo)
    ('microsoft/deberta-v3-large', 'stub', 'backbone of gliner_large-v2.5 and NuNER_Zero: config + tokenizer only'),
    ('google/mt5-large', 'stub', 'backbone of gliner-x-large: config + tokenizer only'),
    ('google/mt5-base', 'stub', 'backbone of gliner-x-base: config + tokenizer only'),
    # ---- LLMs, zero-shot, Hypatia GPU only ----------------------------------------------------------------------------------------------------
    ('Qwen/Qwen2.5-7B-Instruct', 'llm', 'Apache-2.0'),
    ('Qwen/Qwen3.5-9B', 'llm', 'Apache-2.0, 2026, multimodal wrapper: use text only'),
]

SKIP_DIRS = ('onnx/', 'openvino/', 'tf/', 'flax/', 'coreml/', 'gguf/', 'runs/', 'logs/', 'assets/', 'images/', 'checkpoint-', 'checkpoints/', 'last-checkpoint/')
SKIP_EXT = ('.zip', '.tar', '.tar.gz', '/optimizer.pt', '/scheduler.pt', '/rng_state.pth', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin', '.onnx', '.gguf', '.h5', '.msgpack', '.ot', '.tflite', '.mlmodel', '.png', '.jpg', '.jpeg', '.gif', '.mp4', '.safetensors.index.json.bak')


WEIGHT_EXT = ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.h5', '.msgpack', '.onnx', '.gguf', '.ot', '.tflite', '.whl')


def pick(files: list[dict], stub: bool = False) -> list[dict]:
    if stub:
        return [f for f in files if not f.rfilename.lower().endswith(WEIGHT_EXT) and (f.size or 0) < 20_000_000 and '/' not in f.rfilename]
    names = [f.rfilename for f in files]
    has_st = any(n.endswith('.safetensors') for n in names)
    keep = []
    for f in files:
        n = f.rfilename
        if any(('/' + n).find('/' + d) >= 0 for d in SKIP_DIRS) or n.lower().endswith(SKIP_EXT):
            continue
        if has_st and (n.endswith('.bin') and 'training_args' not in n or n.endswith('.pt') or n.endswith('.pth') or n.endswith('.ckpt')):
            continue
        keep.append(f)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(Path.home() / 'renascor-atlas-data/ner/hf_cache'))
    ap.add_argument('--only', default='')
    ap.add_argument('--role', default='')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for i, r, n in MODELS:
            print(i, r, n, sep='\t')
        return 0
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    code_dir = cache.parent / 'model_code'
    only = {x for x in a.only.split(',') if x}
    rows = []
    for rid, role, note in MODELS:
        if (only and rid not in only) or (a.role and role != a.role):
            continue
        t0 = time.time()
        try:
            info = api.model_info(rid, files_metadata=True)
            files = pick(info.siblings or [], stub=(role == 'stub'))
            size = sum(f.size or 0 for f in files)
            print(f'[{time.strftime("%H:%M:%S")}] {rid} {len(files)} files {size / 1e9:.2f} GB', flush=True)
            for f in files:
                hf_hub_download(rid, f.rfilename, revision=info.sha, cache_dir=str(cache))
            pyfiles = [f.rfilename for f in files if f.rfilename.endswith('.py')]
            automap = ''
            cfg = next((f for f in files if f.rfilename == 'config.json'), None)
            if cfg:
                try:
                    c = json.load(open(hf_hub_download(rid, 'config.json', revision=info.sha, cache_dir=str(cache))))
                    automap = json.dumps(c.get('auto_map', '')) if c.get('auto_map') else ''
                except Exception:  # noqa: BLE001
                    pass
            if pyfiles:
                for p in pyfiles:
                    dst = code_dir / rid / p
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(hf_hub_download(rid, p, revision=info.sha, cache_dir=str(cache)), dst)
            lic = next((t[8:] for t in (info.tags or []) if t.startswith('license:')), '')
            params = (info.safetensors.total if getattr(info, 'safetensors', None) else '') or ''
            weights = ';'.join(sorted({Path(f.rfilename).suffix for f in files if Path(f.rfilename).suffix in ('.safetensors', '.bin', '.pt', '.whl', '.pth', '.ckpt')}))
            rows.append([rid, role, info.sha, lic, str(params), f'{size / 1e9:.2f}', weights, 'yes' if (pyfiles or automap) else 'no', ','.join(pyfiles),
                         automap, str(info.created_at)[:10] if getattr(info, 'created_at', None) else '', ','.join(info.tags[:0]), note, 'ok', f'{time.time() - t0:.0f}s'])
        except Exception as e:  # noqa: BLE001
            print(f'   FAILED {rid}: {str(e)[:200]}', file=sys.stderr, flush=True)
            rows.append([rid, role, '', '', '', '', '', '', '', '', '', '', note, 'FAILED ' + str(e)[:100].replace('\t', ' '), ''])
    hdr = ['id', 'role', 'revision', 'license', 'params', 'GB', 'weights', 'custom_code', 'py_files', 'auto_map', 'created', '_', 'note', 'status', 'time']
    out = cache.parent / ('MODELS.tsv' if not (only or a.role) else f'MODELS_{a.role or "part"}.tsv')
    with open(out, 'w') as f:
        f.write('\t'.join(hdr) + '\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
