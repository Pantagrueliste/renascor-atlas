#!/bin/bash
# v2.3 final build, ONE clean chain from scratch on Hypatia (Slurm only; the login node just prepares small files and submits):
#   extract (all collections) -> merge -> gnscan -> build2 pass1 -> namecheck -> build2 pass2 -> gcindex (+ TGN names) -> gcrun -> build2 pre_verify
#   -> verify rules -> build2 final -> the nine audit checks -> the whole test suite.
# Nothing under out_v2, out, v1, v1.1, v2 or raw is written. usage: slurm/v23_chain.sh   (after scripts/push_repo.sh)
set -euo pipefail
D="$HOME/renascor-atlas-data"
L="--output=$D/logs/%x-%j.out"
OUT="$D/out_v2_3"; R="$D/reconcile_v2_3"; G="$D/geocode_v2_3"
mkdir -p "$D/logs" "$OUT" "$G"
# the authority tables of v2 (Wikidata, GND, GeoNames features: fetched on 2026-10-17, no network now) + the curated corrections of the repository
rm -rf "$R"; mkdir -p "$R"
cp "$D"/reconcile_v2/{ids.tsv,wd_map.tsv,wd_class_labels.tsv,wd_entities.parquet,gnd_records.parquet,gn_ids.parquet,gn_samename.parquet} "$R/"
cd "$HOME/renascor-atlas"
cp data/id_overrides.tsv data/coord_overrides.tsv data/geocode_rejects.tsv data/casebooks_place_map.tsv data/kind_overrides.tsv "$R/"
SLUGS=$(ls src/ratlas/collections/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -v -e '^__init__$' -e '^generic_tei$')
ex=$(OUTNAME=out_v2_3 CONCURRENT=3 slurm/submit.sh $SLUGS | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*$')
S="sbatch --parsable $L"
j1=$($S --dependency=afterok:$ex --cpus-per-task=4 --mem=96G slurm/v23.sbatch merge)
j2=$($S --dependency=afterok:$j1 --cpus-per-task=1 --mem=24G slurm/v23.sbatch gnscan)
j3=$($S --dependency=afterok:$j2 slurm/v23.sbatch build2 pass1)
j4=$($S --dependency=afterok:$j3 --cpus-per-task=1 --mem=32G slurm/v23.sbatch namecheck)
j5=$($S --dependency=afterok:$j4 slurm/v23.sbatch build2 pass2 --namecheck "$R/namecheck_local.tsv")
j6=$($S --dependency=afterok:$j1 --cpus-per-task=2 --mem=96G --time=02:00:00 slurm/v23.sbatch gcindex)
j7=$($S --dependency=afterok:$j5:$j6 --cpus-per-task=2 --mem=64G --time=02:00:00 slurm/v23.sbatch gcrun --eval-n 1000 --eval-seeds 1,2 --eval-latin 1500)
j8=$($S --dependency=afterok:$j7 slurm/v23.sbatch build2 pre_verify --namecheck "$R/namecheck_local.tsv" --geocode "$G/geocode_places.parquet")
j9=$($S --dependency=afterok:$j8 --cpus-per-task=1 --mem=16G --time=00:30:00 slurm/v23.sbatch verify)
j10=$($S --dependency=afterok:$j9 slurm/v23.sbatch build2 final --namecheck "$R/namecheck_local.tsv" --geocode "$G/geocode_places.parquet" --geocode-verify "$G/geocode_verify.tsv")
j11=$($S --dependency=afterok:$j10 --cpus-per-task=2 --mem=64G slurm/v23.sbatch audit)
j12=$($S --dependency=afterany:$j11 --cpus-per-task=1 --mem=24G --time=01:00:00 slurm/v23.sbatch audit_editions)
j13=$($S --dependency=afterany:$j11 --cpus-per-task=2 --mem=64G --time=01:30:00 slurm/v23.sbatch pytest)
# the data shards of the private globe (core.json, wd.json, docs/, img/, letters.json, letters_docs.json, letters_coverage.tsv, build_report.json)
j14=$($S --dependency=afterok:$j10 slurm/cards_globe_v23.sbatch)
echo "extract $ex, merge $j1, gnscan $j2, pass1 $j3, namecheck $j4, pass2 $j5, gcindex $j6, gcrun $j7, pre_verify $j8, verify $j9, final $j10, audit $j11, audit_editions $j12, pytest $j13, cards $j14"
