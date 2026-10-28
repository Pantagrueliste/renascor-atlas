"""Training data for a place-name (LOC) and person-name (PER) NER, built from the spans the editors tagged in TEI editions.

Modules
  tei         one TEI document -> passages of <= ~200 words with character spans (deterministic, lxml)
  langs       language codes and a stop-word language guess
  editions    the editions (= collection slugs) of the atlas: files, licences, languages
  extract     Slurm job: all editions -> passages/<slug>.jsonl.gz
  complete    annotation completeness per edition and per passage
  splits      train / dev / test, 15% cap, HF JSONL, DATASHEET.md
"""
READING = 'diplomatic'   # see tei.py: the reading of choice/orig/reg/abbr/expan/sic/corr used everywhere
