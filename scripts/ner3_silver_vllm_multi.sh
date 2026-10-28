#!/bin/bash
# Data-parallel RenNER silver run: one vLLM process per GPU (task i of a node takes the node's GPU i), over one or several nodes, in a single Slurm job.
#   sbatch -J r3s-multi -N 3 --ntasks-per-node=2 --gres=gpu:2 -t 03:30:00 scripts/ner3_silver_vllm_multi.sh \
#     "GUIDE|PASSAGES|OUT_PREFIX" ...
# The remaining passages of every source are split across all tasks (scripts/ner3_silver_shard.py); task i writes
# OUT_PREFIX.sNN.jsonl. Resubmitting the same command finishes what a time limit cut (done = id in any OUT_PREFIX*.jsonl).
# R3_FEWSHOT=scripts/renner_fewshot_v4.jsonl (exported before sbatch) replaces the default worked examples (v4 Latin verse run).
#SBATCH -p gpu.A100
#SBATCH -c 8
#SBATCH --mem=400G
#SBATCH -t 03:30:00
#SBATCH -o /home/cag30/renascor-atlas-data/ner/silver/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
source "$HOME/compendo-v3-170/vllm-venv-cu129/bin/activate"   # torch cu129 runs on the 12.5 driver of the A100 nodes
export CUDA_HOME="$HOME/cuda-12.9" PATH="$HOME/cuda-12.9/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
cd "$HOME/renascor-atlas"
SH="$D/silver/shards/${SLURM_JOB_ID}"
echo "[r3m] $(date +%T) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST tasks $SLURM_NTASKS"
python scripts/ner3_silver_shard.py "$SLURM_NTASKS" "$SH" "$@" || exit 1
srun --kill-on-bad-exit=0 bash -c '
  s=$(printf "%02d" $SLURM_PROCID)
  export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))   # one GPU per task
  echo "[r3m] $(date +%T) task $s on $(hostname) gpu $CUDA_VISIBLE_DEVICES"
  while IFS=$'"'"'\t'"'"' read -r guide passages out; do
    [ -z "$guide" ] && continue
    python scripts/renner_vllm_teacher.py --tp 1 --model google/gemma-4-31B-it --name gemma4_31b --text-only \
      --max-model-len 8192 --prompts fewshot --batch 500 --guide "$guide" ${R3_FEWSHOT:+--fewshot "$R3_FEWSHOT"} --passages "$passages" --out "$out" \
      >> "'"$SH"'/task$s.log" 2>&1 || echo "[r3m] task $s FAILED on $passages"
    echo "[r3m] $(date +%T) task $s done $passages"
  done < "'"$SH"'/slot$s.tsv"
'
echo "[r3m] $(date +%T) all tasks finished"
