#!/bin/bash
# RenNER teacher runs on Hypatia with vLLM 0.31 (venv vllm-venv-cu129 of compendo-v3-170, used read-only).
# Each argument is one full argument list for scripts/renner_vllm_teacher.py; the runs go one after the other in the same job.
#   dev3 check:  sbatch -J r3s-dev3 -t 01:00:00 scripts/ner3_silver_vllm_job.sh \
#                  "--model google/gemma-4-31B-it --name gemma4_31b --text-only --max-model-len 8192 --prompts fewshot \
#                   --passages $D/v3/dev3/passages.jsonl --outdir $D/silver/dev3_check --guide v2=scripts/renner_prompt_rules_en_v2.md ..."
#   silver chunk: sbatch -J r3s-la_c01 -t 03:30:00 scripts/ner3_silver_vllm_job.sh \
#                  "--model google/gemma-4-31B-it --name gemma4_31b --text-only --max-model-len 8192 --prompts fewshot \
#                   --guide scripts/renner_prompt_rules_en_v3.md --passages $D/silver/chunks/la_c01.jsonl \
#                   --out $D/silver/raw/la_c01__gemma4_31b.jsonl --batch 1000"
# --out runs are resumable: a chunk stopped by the time limit is finished by resubmitting the same command.
#SBATCH -p gpu.A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /home/cag30/renascor-atlas-data/ner/silver/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
source "$HOME/compendo-v3-170/vllm-venv-cu129/bin/activate"   # torch cu129 runs on the 12.5 driver of the A100 nodes
export CUDA_HOME="$HOME/cuda-12.9" PATH="$HOME/cuda-12.9/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export R3_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l)"
echo "[r3s] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpus: $R3_GPU vllm $(python -c 'import vllm; print(vllm.__version__)')"
cd "$HOME/renascor-atlas"
TP=$(nvidia-smi -L | wc -l)
rc=0
for spec in "$@"; do
  echo "=== $(date +%T) $spec"
  # shellcheck disable=SC2086
  python scripts/renner_vllm_teacher.py --tp "$TP" $spec || { rc=1; echo "=== FAILED $(date +%T)"; }
  echo "=== end $(date +%T)"
done
exit $rc
