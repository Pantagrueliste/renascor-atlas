#!/bin/bash
# External comparison (N/v2/external_comparison.md), GPU part, ONE Slurm job, one task per GPU: off-the-shelf NER models on dev3 (selection set),
# the editor-tag bench and test_v2 (repaired text), and our three models plus the off-the-shelf ones on the public test sets that are not in our
# training mixes (pub.jsonl). Gold v2 and gold v3 are never read. Units (system x set) are claimed in order (mkdir is atomic); an output that
# exists is skipped (resumable); a failing unit does not stop the others.
#   sbatch -J extcmp-gpu -p gpu.L40S --exclude=gpu03 -N 1 --ntasks-per-node=4 --gres=gpu:4 -c 8 --mem=200G -t 03:00:00 scripts/ner3_extcmp_gpu.sh
#SBATCH -o /home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export X="$D/v2/extcmp" JD="$D/v2/extcmp/jobs/$SLURM_JOB_ID"
mkdir -p "$X/pred"/{dev3,bench,tv2,pub} "$X/logs" "$JD/claims"
cat > "$JD/units.txt" <<'EOF'
zs_gliner_x_large|bench|venv_train|zs|--backend gliner --model LOCAL/gliner-x-large --labels LOC=place,PER=person --threshold 0.3
zs_gliner_multi_v21|bench|venv_train|zs|--backend gliner --model urchade/gliner_multi-v2.1 --labels LOC=location,PER=person --threshold 0.3
zs_gliner25_multi|bench|venv_gl2|zs|--backend gliner2 --model fastino/gliner2.5-multi-v1 --labels LOC=place,PER=person --threshold 0.3
zs_ghent_latin|bench|venv_gl2|zs|--backend gliner2 --model GhentCDH/Latin-hagiography-NER --labels LOC=place,PER=person --threshold 0.3
zs_flair_ner_en_large|bench|venv_flair|zs|--backend flair --model flair/ner-english-large --map loc=LOC,per=PER
zs_flair_ner_de_large|bench|venv_flair|zs|--backend flair --model flair/ner-german-large --map loc=LOC,per=PER
zs_davlan_xlmr_large|bench|venv_train|zs|--backend hf --model Davlan/xlm-roberta-large-ner-hrl --map loc=LOC,per=PER
zs_xlmr_conll03|bench|venv_train|zs|--backend hf --model FacebookAI/xlm-roberta-large-finetuned-conll03-english --map loc=LOC,per=PER --iob1
zs_neolatiner|bench|venv_train|zs|--backend hf --model mhvdr/neolatiner-model --map geo=LOC,prs=PER
zs_impresso_mmbert|bench|venv_train|zs|--backend hf --model impresso-project/ner-hipe2020-mmbert-base-final --map loc=LOC,pers=PER
zs_wikineural|bench|venv_train|zs|--backend hf --model Babelscape/wikineural-multilingual-ner --map loc=LOC,per=PER
zs_medieval_latin|bench|venv_train|zs|--backend hf --model ERCDiDip/medieval-latin-ner --map loc=LOC,per=PER
zs_aendriu_it|bench|venv_train|zs|--backend hf --model aendriu/bert-ner-italian-historical --map loc=LOC,per=PER
zs_flair_ner_multi|bench|venv_flair|zs|--backend flair --model flair/ner-multi --map loc=LOC,per=PER
zs_hmbench_hipe2020_fr|bench|venv_flair|zs|--backend flair --model hmbert-64k/flair-hipe-2022-hipe2020-fr --map loc=LOC,per=PER,pers=PER
zs_hmbench_hipe2020_de|bench|venv_flair|zs|--backend flair --model hmbert-64k/flair-hipe-2022-hipe2020-de --map loc=LOC,per=PER,pers=PER
zs_gliner_x_large|tv2|venv_train|zs|--backend gliner --model LOCAL/gliner-x-large --labels LOC=place,PER=person --threshold 0.3
zs_gliner_multi_v21|tv2|venv_train|zs|--backend gliner --model urchade/gliner_multi-v2.1 --labels LOC=location,PER=person --threshold 0.3
zs_gliner25_multi|tv2|venv_gl2|zs|--backend gliner2 --model fastino/gliner2.5-multi-v1 --labels LOC=place,PER=person --threshold 0.3
zs_ghent_latin|tv2|venv_gl2|zs|--backend gliner2 --model GhentCDH/Latin-hagiography-NER --labels LOC=place,PER=person --threshold 0.3
zs_xlmr_conll03|tv2|venv_train|zs|--backend hf --model FacebookAI/xlm-roberta-large-finetuned-conll03-english --map loc=LOC,per=PER --iob1
zs_neolatiner|tv2|venv_train|zs|--backend hf --model mhvdr/neolatiner-model --map geo=LOC,prs=PER
EOF
for S in dev3 pub; do
  grep '|bench|' "$JD/units.txt" | sed "s/|bench|/|$S|/" >> "$JD/units.txt"
done
for M in v2_xlmrxl_Sfull v2_mdeberta_Sfull r3_mdeberta_HM; do echo "$M|pub|venv_train|enc|" >> "$JD/units.txt"; done
echo "[extcmp] $(date +%T) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST tasks $SLURM_NTASKS units $(wc -l < "$JD/units.txt")"
srun --kill-on-bad-exit=0 bash -c '
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
s=$(printf "%02d" "$SLURM_PROCID"); LOG="$JD/task$s.log"
export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
cd "$HOME/renascor-atlas"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")" >> "$LOG"
while IFS="|" read -r NAME SET VENV KIND ARGS; do
  [ -n "$NAME" ] || continue
  mkdir "$JD/claims/$NAME.$SET" 2>/dev/null || continue
  case "$SET" in
    dev3)  IN="$D/v3/dev3/passages.jsonl" ;;
    bench) IN="$D/v2/editor_bench/bench.jsonl" ;;
    tv2)   IN="$D/v2/test2/test_v2_loc.jsonl" ;;
    pub)   IN="$X/data/pub.jsonl" ;;
  esac
  OUT="$X/pred/$SET/$NAME.jsonl.gz"; [ -s "$OUT" ] && continue
  ARGS="${ARGS//LOCAL\//$D/local_models/}"
  t0=$(date +%s)
  if [ "$KIND" = enc ]; then
    case "$NAME" in r3_*) MD="$D/models/$NAME" ;; *) MD="$D/train_v2/models/$NAME" ;; esac
    timeout 60m "$D/venv_train/bin/python" scripts/ner_audit_predict.py --model-dir "$MD" --data "$IN" --out "$OUT.part" --words-from tokens > "$X/logs/$NAME.$SET.log" 2>&1
  else
    timeout 60m "$D/$VENV/bin/python" scripts/ner3_extcmp_zs.py $ARGS --data "$IN" --out "$OUT.part" --tmp "$X/tmp" > "$X/logs/$NAME.$SET.log" 2>&1
  fi
  rc=$?; [ $rc -eq 0 ] && mv "$OUT.part" "$OUT"
  echo "[task $s] $(date "+%F %T") $NAME $SET rc=$rc seconds=$(( $(date +%s) - t0 ))" >> "$LOG"
done < "$JD/units.txt"
echo "[task $s] $(date "+%F %T") done" >> "$LOG"
'
echo "[extcmp] $(date +%T) all tasks finished"
cat "$JD"/task*.log
