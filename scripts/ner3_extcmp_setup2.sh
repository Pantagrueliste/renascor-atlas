#!/bin/bash
# External comparison, setup part 2 (debug partition, network): spaCy 3.8 model wheels from the GitHub releases of explosion/spacy-models
# (the Hugging Face copies are 3.7 and need spaCy < 3.8), and the LatinCy dependencies (PyPI).
#   sbatch -p debug -c 4 --mem=16G -t 00:45:00 -J extcmp-setup2 -o ~/renascor-atlas-data/ner/logs/%x-%j.out scripts/ner3_extcmp_setup2.sh
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; V="$D/venv_spacy"; W="$D/external_v3/spacy_wheels"; mkdir -p "$W"
for m in es_core_news_lg it_core_news_lg fr_core_news_lg de_core_news_lg en_core_web_lg; do
  f="$m-3.8.0-py3-none-any.whl"
  [ -s "$W/$f" ] || curl -sfL --retry 3 -o "$W/$f" "https://github.com/explosion/spacy-models/releases/download/$m-3.8.0/$f"
  sha256sum "$W/$f"
  "$V/bin/pip" install -q --no-deps "$W/$f"
done
"$V/bin/pip" install -q "latincy-preprocess>=0.2.0,<0.6.0" "la-latincy-lookups>=1.0.0,<1.1.0"
"$V/bin/pip" install -q --no-deps "$D"/hf_cache/models--latincy--la_core_web_lg/snapshots/*/la_core_web_lg-3.9.8-py3-none-any.whl
"$V/bin/pip" list 2>/dev/null | grep -i -E "spacy|core_|grand|latincy|^la"
"$V/bin/python" -I - <<'PY'
import spacy
for m in ['es_core_news_lg', 'it_core_news_lg', 'fr_core_news_lg', 'de_core_news_lg', 'en_core_web_lg', 'la_core_web_lg', 'de_grand_tour_ner']:
    try:
        nlp = spacy.load(m)
        print(m, 'ok', nlp.meta.get('version'), nlp.meta.get('spacy_version'), nlp.get_pipe('ner').labels if 'ner' in nlp.pipe_names else nlp.pipe_names)
        print('  ', [(e.text, e.label_) for e in nlp('Cartesius in Gallia natus est; Carolus V Romam venit, et Philippus in Hispaniam.').ents])
    except Exception as e:
        print(m, 'FAILED', repr(e)[:400])
PY
echo done
