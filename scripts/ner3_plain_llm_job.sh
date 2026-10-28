#!/bin/bash
# RenNER plain-LLM baseline (reviewer question): two open instruction models with a plain prompt (renner_vllm_teacher.py --prompts plain:
# no RenNER guidelines, no examples, JSON answer) on dev3 and the 2,000-passage teacher sample of the editor bench (editor_bench/
# teacher_sample, the sample the teacher was scored on). One model per A100, both at once; vLLM 0.31 of compendo-v3-170 (read-only).
# Gold v2 and gold v3 are never read.
#   sbatch -J r6-plain -p gpu.A100 --exclude=gpu03 -N 1 --gres=gpu:2 -c 16 --mem=240G -t 01:00:00 \
#     -o ~/renascor-atlas-data/ner/plain_llm/logs/%x-%j.out scripts/ner3_plain_llm_job.sh
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; O="$D/plain_llm"; mkdir -p "$O/pred" "$O/logs"
P="$O/passages_dev3_benchsample.jsonl"
if [ ! -s "$P" ]; then
  cat "$D/v3/dev3/passages.jsonl" "$D/v2/editor_bench/teacher_sample/passages_la.jsonl" "$D/v2/editor_bench/teacher_sample/passages_other.jsonl" > "$P.tmp" && mv "$P.tmp" "$P"
fi
echo "[r6p] $(date +%T) $(hostname) job $SLURM_JOB_ID passages $(wc -l < "$P")"
source "$HOME/compendo-v3-170/vllm-venv-cu129/bin/activate"
export CUDA_HOME="$HOME/cuda-12.9" PATH="$HOME/cuda-12.9/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
cd "$HOME/renascor-atlas"
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
run() {   # gpu model name
  CUDA_VISIBLE_DEVICES="$1" R3_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$1") x1" \
    python scripts/renner_vllm_teacher.py --tp 1 --model "$2" --name "$3" --text-only --max-model-len 8192 --prompts plain \
      --passages "$P" --outdir "$O/pred" > "$O/logs/$3-$SLURM_JOB_ID.log" 2>&1
  echo "[r6p] $(date +%T) $3 exit $?"
}
run "${GPUS[0]}" google/gemma-4-31B-it gemma4_31b &
run "${GPUS[1]}" Qwen/Qwen3.8-27B qwen38_27b &
wait
ls -la "$O/pred"
