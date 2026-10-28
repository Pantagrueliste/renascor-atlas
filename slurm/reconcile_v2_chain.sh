#!/bin/bash
# Submit the v2 reconcile chain (after pass 1 and geocode_index_v2 exist):
#   namecheck -> pass 2 (arbitration) -> geocode run -> build with geocode (pre_verify) -> verify rules -> final build
# usage: slurm/reconcile_v2_chain.sh [<job id to wait for>]
set -euo pipefail
D="$HOME/renascor-atlas-data"
L="--output=$D/logs/%x-%j.out"
R="$D/out_v2/reconcile"
dep=""
[ -n "${1:-}" ] && dep="--dependency=afterok:$1"
j1=$(sbatch --parsable $L $dep slurm/namecheck_v2.sbatch)
j2=$(sbatch --parsable $L --dependency=afterok:$j1 slurm/build2.sbatch --dest $R/pass2 --namecheck $D/reconcile_v2/namecheck_local.tsv)
j3=$(sbatch --parsable $L --dependency=afterok:$j2 slurm/geocode_run_v2.sbatch --eval-n 1000 --eval-seeds 1,2)
j4=$(sbatch --parsable $L --dependency=afterok:$j3 slurm/build2.sbatch --dest $R/pre_verify --namecheck $D/reconcile_v2/namecheck_local.tsv \
  --geocode $D/geocode_v2/geocode_places.parquet)
j5=$(sbatch --parsable $L --dependency=afterok:$j4 slurm/verify_v2.sbatch)
j6=$(sbatch --parsable $L --dependency=afterok:$j5 slurm/build2.sbatch --dest $R/final --namecheck $D/reconcile_v2/namecheck_local.tsv \
  --geocode $D/geocode_v2/geocode_places.parquet --geocode-verify $D/geocode_v2/geocode_verify.tsv)
echo "namecheck $j1, pass2 $j2, geocode $j3, pre_verify $j4, verify $j5, final $j6"
