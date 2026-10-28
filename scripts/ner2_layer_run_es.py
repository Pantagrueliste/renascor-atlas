"""NER layer, Spanish rerun (GPU): same as ner2_layer_run.py but with r3_mdeberta_HM as the main model.

    python scripts/ner2_layer_run_es.py <same arguments as ner2_layer_run.py>

The main model is the first of MODELS; the three other encoders (xlmr large, mmbert, hmteams) are run on the passages where the main model has a span of
confidence >= --select-conf (use 0 to run them on every passage that has a span: Spanish confidences are low). Output schema is unchanged: ``n_agree`` counts
the encoders (the main one included) with exactly the same span.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner2_layer_run as R

R.MODELS = ['r3_mdeberta_HM', 'r3_xlmr_large_H', 'r3_mmbert_base_HM', 'r3_hmteams_HM']

if __name__ == '__main__':
    R.main()
