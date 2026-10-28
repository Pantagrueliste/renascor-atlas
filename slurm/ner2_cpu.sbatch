#!/bin/bash
# NER v2, CPU job on the debug partition (<= 24 cpus, 2 h): JOB_SH (a bash script, path relative to ~/renascor-atlas) runs with the extraction environment.
#   sbatch --job-name=ner2_s2_rebuild --export=ALL,JOB_SH=scripts/ner2_rebuild_test.sh slurm/ner2_cpu.sbatch
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data"
export RATLAS_DATA="$D" RATLAS_RAW="$D/raw" RATLAS_SCAN="$D/scan.jsonl.gz" RATLAS_SCAN_EXTRA="$D/scan_extra.jsonl.gz" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export ND="$D/ner" PY="$HOME/renascor-atlas/.venv/bin/python"
cd "$HOME/renascor-atlas"
echo "[ner2-cpu] $(date +%T) $(hostname) JOB_SH=$JOB_SH cpus=${SLURM_CPUS_PER_TASK:-1}"
J="$(mktemp /tmp/ner2job.XXXXXX.sh)"; cp "$JOB_SH" "$J"      # a copy: the script on disk may be edited while the job runs
nice -n 10 bash "$J"
rc=$?
echo "[ner2-cpu] exit $rc $(date +%T)"
exit $rc
