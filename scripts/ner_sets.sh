# Source this on Hypatia: the evaluation sets shared by every run.
export ND="$HOME/renascor-atlas-data/ner"
export OURS="$ND/data_v1"
export EXT="$ND/external/aggregate/hist_gold/test"
export SET_ARGS="--set dev=$OURS/dev.jsonl --set test=$OURS/test.jsonl --set ext_hipe=$EXT/hipe2022__*.jsonl --set ext_hist=$EXT/europeana__*.jsonl;$EXT/reichsanzeiger__*.jsonl;$EXT/geoedda__*.jsonl;$EXT/globalise__*.jsonl;$EXT/earlymodern_german__*.jsonl;$EXT/sdhk_old_swedish__*.jsonl;$EXT/kind__*.jsonl;$EXT/latin_*.jsonl --set ext_charters=$EXT/medieval_charters__*.jsonl"
# the same as --test / --eval arguments of ratlas.ner.train (name=path)
export TRAIN_EVAL_ARGS="--test test=$OURS/test.jsonl --eval ext_hipe=$EXT/hipe2022__*.jsonl --eval ext_hist=$EXT/europeana__*.jsonl;$EXT/reichsanzeiger__*.jsonl;$EXT/geoedda__*.jsonl;$EXT/globalise__*.jsonl;$EXT/earlymodern_german__*.jsonl;$EXT/sdhk_old_swedish__*.jsonl;$EXT/kind__*.jsonl;$EXT/latin_*.jsonl --eval ext_charters=$EXT/medieval_charters__*.jsonl"
