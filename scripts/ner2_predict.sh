#!/bin/bash
# NER v2: LOC / PER predictions of every system scored in v1 on the repaired test passages (test_v2_loc.jsonl), one file per system in $ND/v2/test2/pred/.
# Same models, label maps, thresholds and windows as the v1 audit (scripts/ner_audit_predict.py, ner2_zs.py = ner_audit_zs.py + gliner2). Resumable: a system with a
# non-empty output file is skipped; a failing system does not stop the others.
#   MODE=enc (venv_train)  our fine-tuned encoders
#   MODE=zs  (venv_train)  Hub token classifiers and GLiNER models
#   MODE=flair (venv_flair), MODE=gl2 (venv_gl2), MODE=llm (venv_llm, A100: Qwen3.5-9B on 600 passages per edition, the v1 sample first)
set -u
O="$ND/v2/test2"; P="$O/pred"; mkdir -p "$P"
DATA="$O/test_v2_loc.jsonl"
[ -s "$DATA" ] || { echo "missing $DATA"; exit 2; }
LOCAL="$ND/local_models"
run() {         # run NAME cmd...  (the command writes $P/NAME.jsonl.gz)
  local name="$1"; shift
  if [ -s "$P/$name.jsonl.gz" ]; then echo "=== skip $name"; return; fi
  echo "=== $name $(date +%T)"
  "$@"; echo "=== $name rc=$? $(date +%T)"
}
zs() { local name="$1"; shift; run "$name" python scripts/ner2_zs.py "$@" --data "$DATA" --out "$P/$name.jsonl.gz"; }
case "${MODE:?}" in
  enc)
    for M in r3_mdeberta_HM r3_mmbert_base_HM r3_xlmr_large_H r3_hmteams_HM ft_xlmr_large_ours ft_hmteams_ours ft_mdeberta_ours ft_mmbert_base_ours ft_xlmr_base_ours \
             ft_hmbert64k_ours ft_eurobert210_ours ft_eurobert610_ours r3_mdeberta_H r3_mmbert_base_H r3_hmteams_H r3_xlmr_base_H r3_xlmr_base_HM r3_hmteams_twostage; do
      run "$M" python scripts/ner_audit_predict.py --model-dir "$ND/models/$M" --data "$DATA" --out "$P/$M.jsonl.gz"
    done ;;
  zs)
    zs zs_davlan_xlmr_large --backend hf --model Davlan/xlm-roberta-large-ner-hrl --map loc=LOC,per=PER
    zs zs_impresso_mmbert --backend hf --model impresso-project/ner-hipe2020-mmbert-base-final --map loc=LOC,pers=PER
    zs zs_impresso_hmbert --backend hf --model impresso-project/ner-hipe2020-hist-base --map loc=LOC,pers=PER
    zs zs_wikineural --backend hf --model Babelscape/wikineural-multilingual-ner --map loc=LOC,per=PER
    zs zs_medieval_latin --backend hf --model ERCDiDip/medieval-latin-ner --map loc=LOC,per=PER
    zs zs_aendriu_it --backend hf --model aendriu/bert-ner-italian-historical --map loc=LOC,per=PER
    zs zs_dell_en --backend hf --model dell-research-harvard/historical_newspaper_ner --map loc=LOC,per=PER
    zs zs_gliner_x_large --backend gliner --model "$LOCAL/gliner-x-large" --labels LOC=place,PER=person --threshold 0.7
    zs zs_gliner_x_base --backend gliner --model "$LOCAL/gliner-x-base" --labels LOC=place,PER=person --threshold 0.7
    zs zs_gliner_multi_v21_location --backend gliner --model urchade/gliner_multi-v2.1 --labels LOC=location,PER=person --threshold 0.5
    zs zs_gliner_large_v25 --backend gliner --model gliner-community/gliner_large-v2.5 --variant bf16 --labels LOC=place,PER=person --threshold 0.3
    zs zs_sauerkraut_gliner --backend gliner --model VAGOsolutions/SauerkrautLM-GLiNER --labels LOC=place,PER=person --threshold 0.5
    zs zs_nuner_zero --backend gliner --model numind/NuNER_Zero --labels LOC=location,PER=person --threshold 0.7
    zs ft_gliner_multi_ours_v2 --backend gliner --model "$ND/models/ft_gliner_multi_ours_v2" --labels LOC=place,PER=person --threshold 0.1 --split-ws ;;
  flair)
    zs zs_flair_ner_en_large --backend flair --model flair/ner-english-large --map loc=LOC,per=PER
    zs zs_flair_ner_de_large --backend flair --model flair/ner-german-large --map loc=LOC,per=PER
    zs zs_flair_ner_multi --backend flair --model flair/ner-multi --map loc=LOC,per=PER
    zs zs_hmbench_hipe2020_fr --backend flair --model hmbert-64k/flair-hipe-2022-hipe2020-fr --map loc=LOC,per=PER,pers=PER
    zs zs_hmbench_hipe2020_de --backend flair --model hmbert-64k/flair-hipe-2022-hipe2020-de --map loc=LOC,per=PER,pers=PER
    zs zs_hmbench_newseye_fr --backend flair --model hmbert-64k/flair-hipe-2022-newseye-fr --map loc=LOC,per=PER,pers=PER
    zs zs_hmbench_topres_en --backend flair --model hmbert-64k/flair-hipe-2022-topres19th-en --map loc=LOC,per=PER,pers=PER ;;
  gl2)
    zs zs_gliner2_multi --backend gliner2 --model fastino/gliner2-multi-v1 --labels LOC=place,PER=person --threshold 0.7
    zs zs_gliner25_multi --backend gliner2 --model fastino/gliner2.5-multi-v1 --labels LOC=place,PER=person --threshold 0.7 ;;
  llm)
    run zs_llm_qwen35_9b python scripts/ner2_llm.py --data "$DATA" --v1-sample "$ND/samples/llm_sample_1200.jsonl" --per-edition 600 --out "$P/zs_llm_qwen35_9b.jsonl.gz" ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "systems done: $(ls "$P" | wc -l)"
