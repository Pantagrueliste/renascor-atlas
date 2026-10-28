# Cavriana transfer test: the S-full and HM mixes with Cavriana removed (scripts/ner3_cav_filter.py --write). Source after ner_sets.sh.
# A source with no removal keeps its original path (same file, same sample, same cache entry).
C="$ND/cav_transfer/data"
_p() { if [ -s "$C/$1/$1.jsonl" ]; then echo "$C/$1/$1.jsonl"; else echo "$2"; fi; }
_A="$ND/external/aggregate"; _S="$ND/silver/final"
export CAV_OURS="$(_p ours "$OURS/train.jsonl")"
export CAV_DEV="$(_p dev "$OURS/dev.jsonl")"
export MIX_HM_NOCAV="--train ours=$CAV_OURS --train hist_gold=$(_p hist_gold "$_A/hist_gold/train.jsonl") --train modern_silver=$(_p modern_silver "$_A/modern_silver/train.jsonl"):6000000"
export MIX_SFULL_NOCAV="$MIX_HM_NOCAV --train silver_bal_es=$(_p silver_bal_es "$_S/silver_balanced_es.jsonl") --train silver_bal_it=$(_p silver_bal_it "$_S/silver_balanced_it.jsonl") --train silver_bal_la=$(_p silver_bal_la "$_S/silver_balanced_la.jsonl") --train silver_res_es=$(_p silver_res_es "$_S/silver_reserve_es.jsonl") --train silver_res_it=$(_p silver_res_it "$_S/silver_reserve_it.jsonl")"
