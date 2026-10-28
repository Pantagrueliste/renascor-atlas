#!/bin/bash
# External comparison (N/v2/external_comparison.md): downloads and environment, network only, on the debug partition.
#   sbatch -p debug -c 4 --mem=16G -t 01:30:00 -J extcmp-setup -o ~/renascor-atlas-data/ner/logs/%x-%j.out scripts/ner3_extcmp_setup.sh
# 1. Grand Tour NER (Zenodo 23056599, MIT): early modern German travel reports, test data + spaCy model.
# 2. Hugging Face weights (public repos, no token): mhvdr/neolatiner-model, FacebookAI/xlm-roberta-large-finetuned-conll03-english,
#    spacy/{es,it,fr,de}_core_news_lg, spacy/en_core_web_lg.
# 3. venv_spacy: spaCy and the model wheels (LatinCy from the cache, Grand Tour from its zip). Nothing is uploaded.
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
X="$D/external_v3"; mkdir -p "$X/grandtour/raw"
export HF_HOME="$D/hf_cache/home" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
echo "[setup] $(date +%T) $(hostname)"
# 1. Grand Tour
Z="$X/grandtour/raw/grand-tour-ner-main.zip"
if [ ! -s "$Z" ]; then
  curl -sfL --retry 3 -o "$Z.part" "https://zenodo.org/api/records/23056599/files/grand-tour-ner-main.zip/content" && mv "$Z.part" "$Z"
fi
ls -la "$Z"; sha256sum "$Z"
( cd "$X/grandtour/raw" && unzip -o -q "$Z" 'grand-tour-ner-main/data/*' 'grand-tour-ner-main/README.md' 'grand-tour-ner-main/LICENSE' \
    'grand-tour-ner-main/model/de_grand_tour_ner-0.0.1/dist/*' 'grand-tour-ner-main/model/de_grand_tour_ner-0.0.1/meta.json' 'grand-tour-ner-main/spacy_config/*' )
find "$X/grandtour/raw/grand-tour-ner-main" -maxdepth 4 -type f | head -30
# 2. HF weights
"$D/venv_hf/bin/python" -I - "$D/hf_cache" <<'EOF'
import sys
from huggingface_hub import snapshot_download
cache = sys.argv[1]
for rid in ['mhvdr/neolatiner-model', 'FacebookAI/xlm-roberta-large-finetuned-conll03-english',
            'spacy/es_core_news_lg', 'spacy/it_core_news_lg', 'spacy/fr_core_news_lg', 'spacy/de_core_news_lg', 'spacy/en_core_web_lg']:
    allow = ['*.whl', '*.json', 'README.md'] if rid.startswith('spacy/') else ['*.json', '*.safetensors', '*.model', '*.txt', 'README.md']
    p = snapshot_download(rid, cache_dir=cache, allow_patterns=allow)
    print(rid, p, flush=True)
EOF
# 3. venv_spacy
V="$D/venv_spacy"
if [ ! -x "$V/bin/python" ]; then
  /software/Anaconda/build/bin/python3 -m venv "$V"
fi
"$V/bin/pip" install -q --upgrade pip
"$V/bin/pip" install -q "spacy>=3.8,<3.9" pyarrow
for w in $(ls "$D"/hf_cache/models--spacy--*/snapshots/*/*.whl); do "$V/bin/pip" install -q --no-deps "$w"; done
"$V/bin/pip" install -q --no-deps "$D"/hf_cache/models--latincy--la_core_web_lg/snapshots/*/la_core_web_lg-3.9.8-py3-none-any.whl || echo "latincy 3.9.8 failed"
"$V/bin/pip" install -q --no-deps "$X"/grandtour/raw/grand-tour-ner-main/model/de_grand_tour_ner-0.0.1/dist/*.whl || echo "grand tour wheel failed"
"$V/bin/pip" list 2>/dev/null | grep -i -E "spacy|core_|grand|^la"
"$V/bin/python" -I - <<'EOF'
import spacy
for m in ['es_core_news_lg', 'it_core_news_lg', 'fr_core_news_lg', 'de_core_news_lg', 'en_core_web_lg', 'la_core_web_lg', 'de_grand_tour_ner']:
    try:
        nlp = spacy.load(m)
        print(m, 'ok', nlp.meta.get('version'), nlp.meta.get('spacy_version'), nlp.get_pipe('ner').labels if 'ner' in nlp.pipe_names else nlp.pipe_names)
    except Exception as e:
        print(m, 'FAILED', repr(e)[:300])
EOF
echo "[setup] done $(date +%T)"
