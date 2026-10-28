#!/usr/bin/env bash
# Copy the audit scripts to Hypatia (new files only; nothing else is touched).
set -euo pipefail
cd "$(dirname "$0")/.."
ssh hypatia 'mkdir -p ~/renascor-atlas/scripts ~/renascor-atlas/slurm ~/renascor-atlas-data/ner/audit'
rsync -a scripts/ner_audit_*.py scripts/ner_audit_*.sh hypatia:renascor-atlas/scripts/ 2>/dev/null || rsync -a scripts/ner_audit_*.py hypatia:renascor-atlas/scripts/
rsync -a slurm/ner_audit_*.sbatch hypatia:renascor-atlas/slurm/
