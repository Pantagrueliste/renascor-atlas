"""Descriptions of the compared systems for the tables: model, size, licence, what it was trained on (and what that implies for the comparison)."""
from __future__ import annotations

# name -> (model, params, licence, trained on / caveat)
INFO = {
    'gazetteer_ours': ('exact match of the training gazetteer', '-', '-', 'our train split; case-sensitive longest match, entries with >= 50% tagged occurrences'),
    'zs_wikineural': ('Babelscape/wikineural-multilingual-ner (mBERT)', '177M', 'CC BY-NC-SA 4.0', 'WikiNEuRal silver (Wikipedia), 9 languages; no Latin'),
    'zs_davlan_xlmr_large': ('Davlan/xlm-roberta-large-ner-hrl', '559M', 'AFL-3.0', 'WikiANN-like data of 10 high-resource languages (no Latin)'),
    'zs_impresso_mmbert': ('impresso-project/ner-hipe2020-mmbert-base-final (mmBERT)', '308M', 'CC BY-NC-SA 4.0', 'HIPE-2022 train+dev (fr de en): in-domain for HIPE, so HIPE test is not zero-shot for it'),
    'zs_impresso_hmbert': ('impresso-project/ner-hipe2020-hist-base (hmBERT)', '110M', 'CC BY-NC-SA 4.0', 'HIPE-2022 train+dev (fr de en)'),
    'zs_globalise_nl': ('globalise/globalise-NER (RoBERTa)', '125M', 'MIT', 'VOC Dutch (17th-18th c.), its own label set (LOC_NAME, PER_NAME)'),
    'zs_aendriu_it': ('aendriu/bert-ner-italian-historical', '109M', 'Apache-2.0', 'Italian historical'),
    'zs_dell_en': ('dell-research-harvard/historical_newspaper_ner', '125M', 'CC BY 2.0', 'English historical newspapers'),
    'zs_medieval_latin': ('ERCDiDip/medieval-latin-ner (XLM-R base)', '277M', 'MIT', 'medieval Latin charters (CBMA, ...): in-domain for ext_charters'),
    'zs_gliner_multi_v21_place': ('urchade/gliner_multi-v2.1, prompts place / person', '289M', 'Apache-2.0', 'Pile-NER silver labels (Mistral-7B)'),
    'zs_gliner_multi_v21_location': ('urchade/gliner_multi-v2.1, prompts location / person', '289M', 'Apache-2.0', 'same'),
    'zs_gliner_multi_v21_multiling': ('urchade/gliner_multi-v2.1, prompts in 5 languages', '289M', 'Apache-2.0', 'same'),
    'zs_gliner_large_v25': ('gliner-community/gliner_large-v2.5 (English)', '435M', 'Apache-2.0', 'English synthetic + supervised'),
    'zs_gliner_x_large': ('knowledgator/gliner-x-large (2025)', '865M', 'Apache-2.0', 'European synthetic multilingual data'),
    'zs_gliner_x_base': ('knowledgator/gliner-x-base (2025)', '494M', 'Apache-2.0', 'same'),
    'zs_sauerkraut_gliner': ('VAGOsolutions/SauerkrautLM-GLiNER', '~300M', 'MIT', 'en de fr it es'),
    'zs_nuner_zero': ('numind/NuNER_Zero', '449M', 'MIT', 'English'),
    'zs_gliner2_multi': ('fastino/gliner2-multi-v1 (2025)', '307M', 'Apache-2.0', 'multilingual'),
    'zs_gliner25_multi': ('fastino/gliner2.5-multi-v1 (2026)', '287M', 'Apache-2.0', 'multilingual'),
    'zs_flair_ner_multi': ('flair/ner-multi', '~400M', 'MIT', 'CoNLL-03 en de nl es'),
    'zs_flair_ner_en_large': ('flair/ner-english-large', '~560M', 'MIT', 'CoNLL-03 English'),
    'zs_flair_ner_de_large': ('flair/ner-german-large', '~560M', 'MIT', 'CoNLL-03/GermEval German'),
    'zs_dalembert_fr': ("pjox/dalembert-classical-fr-ner (FreEM-NER)", '~100M', 'Apache-2.0', 'early modern French 16th-18th c.: in-domain for elan_*'),
    'zs_hmbench_hipe2020_fr': ('hmBERT-64k flair, HIPE-2020 fr', '136M', 'MIT', 'HIPE-2020 fr train'),
    'zs_hmbench_hipe2020_de': ('hmBERT-64k flair, HIPE-2020 de', '136M', 'MIT', 'HIPE-2020 de train'),
    'zs_hmbench_newseye_fr': ('hmBERT-64k flair, NewsEye fr', '136M', 'MIT', 'NewsEye fr train'),
    'zs_hmbench_newseye_de': ('hmBERT-64k flair, NewsEye de', '136M', 'MIT', 'NewsEye de train'),
    'zs_hmbench_topres_en': ('hmBERT-64k flair, TopRes19th en', '136M', 'MIT', 'TopRes19th train'),
    'zs_llm_qwen25_7b': ('Qwen/Qwen2.5-7B-Instruct, JSON prompt', '7.6B', 'Apache-2.0', 'general instruction LLM, 300 passages only (llm_sample)'),
}

ENCODERS = {
    'xlmr_base': ('FacebookAI/xlm-roberta-base', '279M', 'MIT'),
    'xlmr_large': ('FacebookAI/xlm-roberta-large', '561M', 'MIT'),
    'hmbert64k': ('dbmdz/bert-base-historic-multilingual-64k-td-cased', '136M', 'MIT'),
    'hmteams': ('hmteams/teams-base-historic-multilingual-discriminator', '111M', 'Apache-2.0'),
    'mmbert_base': ('jhu-clsp/mmBERT-base', '307M', 'MIT'),
    'eurobert210': ('EuroBERT/EuroBERT-210m', '210M', 'Apache-2.0'),
    'eurobert610': ('EuroBERT/EuroBERT-610m', '610M', 'Apache-2.0'),
    'mdeberta': ('microsoft/mdeberta-v3-base', '279M', 'MIT'),
}
