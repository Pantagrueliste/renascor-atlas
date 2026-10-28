#!/bin/bash
# v2 reconcile build (gates, arbitration, kinds, globe). Extra arguments are passed to `ratlas reconcile build2`:
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/build2.sbatch --dest $HOME/renascor-atlas-data/out_v2/reconcile/pass1
#   ... --namecheck $D/reconcile_v2/namecheck_local.tsv --geocode $D/geocode_v2/geocode_places.parquet --geocode-verify $D/geocode_v2/geocode_verify.tsv
# inputs: out_v2/all (mentions, entities, *_local, collections.tsv) and reconcile_v2/ (ids, wd_map, wd_entities, wd_class_labels,
# gnd_records, gn_ids, gn_samename, coord_overrides)
#SBATCH --job-name=ratlas-build2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile build2 --v2 "$D/out_v2/all" --rdir "$D/reconcile_v2" "$@"
