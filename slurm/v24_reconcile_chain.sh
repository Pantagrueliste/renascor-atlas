#!/bin/bash
# (DEP=<job id>: wait for it, e.g. a re-run of merge + gnscan)
# v2.4: the reconcile half of the chain only (inputs out_v2_4/all, geocode_v2_4/gn_index.parquet and reconcile_v2_4/ exist; the code changed):
#   build2 pass1 -> namecheck -> pass2 -> gcrun -> pre_verify -> verify -> final -> audit (+ editions) -> whole test suite.  Everything is rebuilt in this one pass.
set -euo pipefail
D="$HOME/renascor-atlas-data"; L="--output=$D/logs/%x-%j.out"; R="$D/reconcile_v2_4"; G="$D/geocode_v2_4"
cd "$HOME/renascor-atlas"
cp data/id_overrides.tsv data/coord_overrides.tsv data/geocode_rejects.tsv data/casebooks_place_map.tsv data/kind_overrides.tsv "$R/"
rm -rf "$D/out_v2_4/reconcile" "$D/out_v2_4/audit_v2"; mkdir -p "$D/out_v2_4/reconcile"
S="sbatch --parsable $L"
j3=$($S ${DEP:+--dependency=afterok:$DEP} slurm/v24.sbatch build2 pass1)
j4=$($S --dependency=afterok:$j3 --cpus-per-task=1 --mem=32G slurm/v24.sbatch namecheck)
j5=$($S --dependency=afterok:$j4 slurm/v24.sbatch build2 pass2 --namecheck "$R/namecheck_local.tsv")
j7=$($S --dependency=afterok:$j5 --cpus-per-task=2 --mem=64G --time=02:00:00 slurm/v24.sbatch gcrun --eval-n 1000 --eval-seeds 1,2 --eval-latin 1500)
j8=$($S --dependency=afterok:$j7 slurm/v24.sbatch build2 pre_verify --namecheck "$R/namecheck_local.tsv" --geocode "$G/geocode_places.parquet")
j9=$($S --dependency=afterok:$j8 --cpus-per-task=1 --mem=16G --time=00:30:00 slurm/v24.sbatch verify)
j10=$($S --dependency=afterok:$j9 slurm/v24.sbatch build2 final --namecheck "$R/namecheck_local.tsv" --geocode "$G/geocode_places.parquet" --geocode-verify "$G/geocode_verify.tsv")
j11=$($S --dependency=afterok:$j10 --cpus-per-task=2 --mem=64G slurm/v24.sbatch audit)
j12=$($S --dependency=afterany:$j11 --cpus-per-task=1 --mem=24G --time=01:00:00 slurm/v24.sbatch audit_editions)
j13=$($S --dependency=afterany:$j11 --cpus-per-task=2 --mem=64G --time=01:30:00 slurm/v24.sbatch pytest)
j14=$($S --dependency=afterok:$j10 slurm/cards_globe_v24.sbatch)
echo "pass1 $j3, namecheck $j4, pass2 $j5, gcrun $j7, pre_verify $j8, verify $j9, final $j10, audit $j11, audit_editions $j12, pytest $j13, cards $j14"
