# Source on Hypatia AFTER ner_sets.sh and ner_mixes.sh: RenNER Latin v4 mixes (latin silver verse and drama re-annotated with prompt v4).
# The v4 Latin silver lives in its own folder, so that the files of the running v2.1 runs (and their cache keys) do not change.
export SIL_LV4="$ND/silver/final_v4"
export MIX_SFULL_V21_LV4="${MIX_SFULL_V21/$SIL\/silver_balanced_la.jsonl/$SIL_LV4/silver_balanced_la.jsonl}"
export MIX_SFULL_LC25_LV4="$MIX_SFULL_V21_LV4 --train aug_lc=$V21/aug/aug_lc25_lv4.jsonl"
