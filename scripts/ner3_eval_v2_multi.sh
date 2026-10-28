#!/bin/bash
# RenNER v2 evaluation in ONE Slurm job, one task per GPU (any number of nodes): predictions of the v2 students (and r3_mdeberta_HM for speed)
# on dev3 and on test_v2 (repaired text), with the predictor of the v1 audit (scripts/ner_audit_predict.py), plus a GPU speed run on dev3 x50.
# Gold v2 is never read here.
#   sbatch -J r4-eval -p gpu.A100 -N 3 --ntasks-per-node=2 --gres=gpu:2 -c 8 --mem=120G -t 01:30:00 scripts/ner3_eval_v2_multi.sh
#SBATCH -o /home/cag30/renascor-atlas-data/ner/train_v2/eval/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export EV="$D/train_v2/eval" JD="$D/train_v2/eval/jobs/$SLURM_JOB_ID"
mkdir -p "$EV"/{pred_dev3,pred_testv2,speed,logs} "$JD/claims"
# dev3 x50 (ids made unique) for the speed run: same passages as the teacher's throughput measure, repeated
if [ ! -s "$EV/speed/dev3_x50.jsonl" ]; then
  python3 - "$D/v3/dev3/passages.jsonl" "$EV/speed/dev3_x50.jsonl" <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
with open(sys.argv[2] + '.tmp', 'w') as f:
    for k in range(50):
        for r in recs:
            f.write(json.dumps({**r, 'id': f"{r['id']}#{k}"}, ensure_ascii=False) + '\n')
import os; os.replace(sys.argv[2] + '.tmp', sys.argv[2])
EOF
fi
echo "[r4e] $(date +%T) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST tasks $SLURM_NTASKS"
srun --kill-on-bad-exit=0 bash "$HOME/renascor-atlas/scripts/ner3_eval_v2_task.sh"
echo "[r4e] $(date +%T) all tasks finished"
cat "$JD"/task*.log
