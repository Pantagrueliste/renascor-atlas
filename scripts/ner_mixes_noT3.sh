# Source on Hypatia AFTER ner_sets.sh and ner_mixes.sh: RenNER variant without the T3 collections CELT and CIDTC.
# Same sets, names and order as MIX_SFULL_LC25_LV4 (ner_mixes_lv4.sh); every file is a filtered copy (scripts/ner3_noT3_filter.py: by edition,
# plus any 8-gram shared with a CELT / CIDTC passage) written to train_noT3/data, except silver_bal_es and silver_bal_it (nothing dropped:
# the original files). modern_silver: per-seed copy of the very sample of the lc25-lv4 run (scripts/ner3_noT3_modern_presample.py), read
# without a cap. Early-stopping dev: DEV_NOT3 (R6_DEV). Nothing of data_v1, silver/ or external/ is modified.
export T3D="$ND/train_noT3/data"
export DEV_NOT3="$T3D/dev_noT3.jsonl"
_noT3_common_a="--train ours=$T3D/train_nomap_noT3.jsonl --train hist_gold=$T3D/hist_gold_train_noT3.jsonl"
_noT3_common_b="--train silver_bal_es=$ND/silver/final/silver_balanced_es.jsonl --train silver_bal_it=$ND/silver/final/silver_balanced_it.jsonl --train silver_bal_la=$T3D/silver_balanced_la_v4_noT3.jsonl --train silver_res_es=$T3D/silver_reserve_es_noT3.jsonl --train silver_res_it=$T3D/silver_reserve_it_noT3.jsonl --train aug_lc=$T3D/aug_lc25_lv4_noT3.jsonl"
for _k in 1 2 3; do
  export MIX_SFULL_LC25_LV4_NOT3_S$_k="$_noT3_common_a --train modern_silver=$T3D/modern_silver_s${_k}_noT3.jsonl $_noT3_common_b"
done
unset _k
