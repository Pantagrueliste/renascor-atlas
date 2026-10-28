#!/bin/bash
# flair taggers (venv_flair): seed passages and held-out test editions.
export MODE=flair
bash scripts/ner_audit_run_seed.sh
bash scripts/ner_audit_run_zs_test.sh
