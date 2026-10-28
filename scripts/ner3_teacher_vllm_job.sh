#!/bin/bash
# Open teachers on dev3 with vLLM 0.31 (venv vllm-venv-cu129 of compendo-v3-170, used read-only; vLLM 0.15 in ~/.local lacks Gemma 4 and Qwen 3.5/3.6).
#   sbatch -p gpu.A100 --gres=gpu:N -J r3h-<tag> scripts/ner3_teacher_vllm_job.sh "<hf id>|<name>|<extra args>" ["..."]
# Models run one after the other in the same job. Outputs in $D/v3/teacher_vllm/.
#SBATCH -p gpu.A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o /home/cag30/renascor-atlas-data/ner/v3/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
source "$HOME/compendo-v3-170/vllm-venv-cu129/bin/activate"   # torch cu129 runs on the 12.5 driver of the A100 nodes; the cu130 venv does not
export CUDA_HOME="$HOME/cuda-12.9" PATH="$HOME/cuda-12.9/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export R3_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l)"
echo "[r3h] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpus: $R3_GPU vllm $(python -c 'import vllm; print(vllm.__version__)')"
cd "$HOME/renascor-atlas"
TP=$(nvidia-smi -L | wc -l)
rc=0
for spec in "$@"; do
  IFS='|' read -r MODEL NAME EXTRA <<< "$spec"
  if [ -s "$D/v3/teacher_vllm/${NAME}__fewshot.jsonl" ] && [ -s "$D/v3/teacher_vllm/${NAME}__rules.jsonl" ]; then echo "skip $NAME"; continue; fi
  echo "=== $NAME ($MODEL) tp=$TP $(date +%T)"
  python scripts/renner_vllm_teacher.py --model "$MODEL" --name "$NAME" --tp "$TP" --passages "$D/v3/dev3/passages.jsonl" \
      --outdir "$D/v3/teacher_vllm" $EXTRA || { rc=1; echo "=== $NAME FAILED"; }
  echo "=== $NAME end $(date +%T)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
exit $rc
