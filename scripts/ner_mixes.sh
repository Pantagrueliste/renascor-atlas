# Source on Hypatia after ner_sets.sh: the training mixes of R3 (ours + external data).
export AGG="$ND/external/aggregate"
export MIX_OURS="--train ours=$OURS/train.jsonl"
# H : ours + all the historical gold data of hist_gold (train splits: HIPE-2022, Europeana, GeoEDdA, Globalise, KIND/De Gasperi, Herodotos, LatinCy, ...), 2.45 M words
export MIX_H="--train ours=$OURS/train.jsonl --train hist_gold=$AGG/hist_gold/train.jsonl"
# HM: H + at most 6 M words of the modern silver data (WikiNEuRal, MultiNERD, WikiANN, Pile-NER, GLiNER synthetic), sampled with the seed
export MIX_HM="$MIX_H --train modern_silver=$AGG/modern_silver/train.jsonl:6000000"
# A : ours + the all_private view (every source capped at 25 % of its words, 13.1 M words)
export MIX_A="--train ours=$OURS/train.jsonl --train all_private=$AGG/all_private/train.jsonl"
# stage 1 of the two-stage variant: external data only
export MIX_EXT="--train hist_gold=$AGG/hist_gold/train.jsonl --train modern_silver=$AGG/modern_silver/train.jsonl:6000000"
# NER v2 (RenNER): + the silver of the LLM teacher (gemma-4-31B-it +pp, R/silver/final; same O/B/I LOC/PER scheme as v1, every passage valid for both labels)
export SIL="$ND/silver/final"
# S-bal: HM + the balanced pool (es, it, la; at most ~25 % per collection)
export MIX_SBAL="$MIX_HM --train silver_bal_es=$SIL/silver_balanced_es.jsonl --train silver_bal_it=$SIL/silver_balanced_it.jsonl --train silver_bal_la=$SIL/silver_balanced_la.jsonl"
# S-full: S-bal + the es/it reserves (no cap per collection)
export MIX_SFULL="$MIX_SBAL --train silver_res_es=$SIL/silver_reserve_es.jsonl --train silver_res_it=$SIL/silver_reserve_it.jsonl"
# NER v2.1 (RenNER): S-full + lowercase-name copies (scripts/ner3_v21_lc_augment.py; train_v21/aug/). lc12 / lc25: 12 % / 25 % of the
# S-full passage count, 30 % of the copies lowercased in full; lc12n: 12 %, entity names only
export V21="$ND/train_v21"
# v2.1: data_v1/train.jsonl holds 11 passages of map_of_early_modern_london (MAP: never in training); the v2.1 runs read a copy without them
export OURS_V21="$V21/data/train_nomap.jsonl"
export MIX_SFULL_V21="${MIX_SFULL/$OURS\/train.jsonl/$OURS_V21}"
export MIX_SFULL_LC12="$MIX_SFULL_V21 --train aug_lc=$V21/aug/aug_lc12.jsonl"
export MIX_SFULL_LC25="$MIX_SFULL_V21 --train aug_lc=$V21/aug/aug_lc25.jsonl"
export MIX_SFULL_LC12N="$MIX_SFULL_V21 --train aug_lc=$V21/aug/aug_lc12n.jsonl"
