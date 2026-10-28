#!/bin/bash
# External comparison, CPU part (debug partition): spaCy pipelines (venv_spacy) on dev3, the editor bench, test_v2 and pub.jsonl, then the
# 8-gram overlap of pub.jsonl with all the training data of the v2 students. Gold v2 and gold v3 are never read. Resumable (outputs skipped).
#   sbatch -p debug -c 16 --mem=64G -t 02:00:00 -J extcmp-cpu -o ~/renascor-atlas-data/ner/logs/%x-%j.out scripts/ner3_extcmp_cpu.sh
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; X="$D/v2/extcmp"
mkdir -p "$X/pred"/{dev3,bench,tv2,pub} "$X/logs"
cd "$HOME/renascor-atlas"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
ROUTE="es=es_core_news_lg,it=it_core_news_lg,fr=fr_core_news_lg,de=de_core_news_lg,en=en_core_web_lg,la=la_core_web_lg"
for SET in pub dev3 tv2 bench; do
  case "$SET" in
    dev3)  IN="$D/v3/dev3/passages.jsonl" ;;
    bench) IN="$D/v2/editor_bench/bench.jsonl" ;;
    tv2)   IN="$D/v2/test2/test_v2_loc.jsonl" ;;
    pub)   IN="$X/data/pub.jsonl" ;;
  esac
  for NAME in zs_spacy_lg zs_grandtour; do
    OUT="$X/pred/$SET/$NAME.jsonl.gz"; [ -s "$OUT" ] && continue
    [ "$NAME" = zs_spacy_lg ] && R="$ROUTE" || R="*=de_grand_tour_ner"
    t0=$(date +%s)
    nice -n 5 "$D/venv_spacy/bin/python" -I scripts/ner3_extcmp_spacy.py --route "$R" --data "$IN" --out "$OUT.part" --procs 12 > "$X/logs/$NAME.$SET.log" 2>&1 && mv "$OUT.part" "$OUT"
    echo "$(date +%T) $NAME $SET rc=$? seconds=$(( $(date +%s) - t0 ))"
  done
done
if [ ! -s "$X/leak_pub.json" ]; then
  "$D/venv_train/bin/python" -I scripts/ner3_extcmp_leak.py --pub "$X/data/pub_new.jsonl" --data "$D/data_v1" --agg "$D/external/aggregate" \
     --silver "$D/silver/final" --out "$X/leak_pub.json" > "$X/logs/leak.log" 2>&1
  echo "$(date +%T) leak rc=$?"
fi
echo done
