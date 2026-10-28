#!/bin/bash
# NER v2, large encoders: smoke tests on ONE A100 (run through slurm/ner_audit_gpu.sbatch with JOB_SH=scripts/ner3_enc_smoke_job.sh).
# For each model: two short probes (60 steps, with and without activation checkpointing) for throughput and memory, then the smoke run proper:
# 300 optimizer steps on the HM mix of v1 (R3), grad checkpointing + 2 micro-batches, stopped at step 150 (checkpoint, exit 3) and resumed to 300,
# then the final evaluation on the full v1 dev set. Gold v2 is never read.
#   MODELS="xlxl eb2b eb610" (default: all three, in this order)
set -u
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
S="$ND/v3/enc_smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$S/runs" "$S/models"
COMMON="$MIX_HM --dev $OURS/dev.jsonl --out $S/runs --models $S/models --cache $ND/cache --batch 32 --max-tokens 12000 --max-len 256 --seed 1 --warmup 0.06"

spec() {   # name -> model path, extra flags, lr, no-ckpt accumulation
  case "$1" in
    xlxl)  echo "facebook/xlm-roberta-xl||1e-5|4" ;;
    eb2b)  echo "$ND/local_models/EuroBERT-2.1B|--trust-remote-code --force-bos|2e-5|2" ;;
    eb610) echo "$ND/local_models/EuroBERT-610m|--trust-remote-code --force-bos|2.5e-5|1" ;;
  esac
}

for m in ${MODELS:-xlxl eb2b eb610}; do
  IFS='|' read -r MP EXTRA LR ACC <<< "$(spec "$m")"
  echo "================ $m $MP lr $LR $(date +%T)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  # probe A: activation checkpointing, whole batch at once
  python -m ratlas.ner.train --name "probe_${m}_ckpt" --model "$MP" $EXTRA --lr "$LR" $COMMON --grad-ckpt --grad-accum 1 \
      --max-steps 60 --limit-dev 300 --skip-final-eval
  echo "[probe ckpt] exit $?"
  # probe B: no checkpointing, the batch split into ACC micro-batches
  python -m ratlas.ner.train --name "probe_${m}_nockpt" --model "$MP" $EXTRA --lr "$LR" $COMMON --grad-accum "$ACC" \
      --max-steps 60 --limit-dev 300 --skip-final-eval
  echo "[probe nockpt] exit $?"
  # smoke run: 150 steps, checkpoint, exit 3; resume to 300; final evaluation on the full v1 dev
  for leg in 1 2; do
    python -m ratlas.ner.train --name "smoke_${m}" --model "$MP" $EXTRA --lr "$LR" $COMMON --grad-ckpt --grad-accum 2 \
        --max-steps 300 --limit-dev 1000 --stop-at-step $([ $leg = 1 ] && echo 150 || echo 0) --save-preds none
    echo "[smoke leg $leg] exit $?"
  done
  rm -f "$S"/models/probe_"$m"_*/model.pt "$S/models/smoke_$m/model.pt"   # smoke weights are not kept
done
echo "all done $(date +%T)"
