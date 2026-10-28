"""NER v2: sanity check of XLM-R XL as loaded by transformers in venv_train (CPU): fill-mask with the MLM head (fp32, sdpa and eager), and the scale
of the last hidden states of AutoModel next to XLM-R large."""
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

torch.set_num_threads(16)
S = ['Paris is the <mask> of France.', 'Roma è la <mask> d\'Italia.', 'Il re di <mask> mandò un ambasciatore a Venezia.']
for mid in ('FacebookAI/xlm-roberta-large', 'facebook/xlm-roberta-xl'):
    tok = AutoTokenizer.from_pretrained(mid)
    for attn in ('sdpa', 'eager'):
        m = AutoModelForMaskedLM.from_pretrained(mid, attn_implementation=attn).eval()
        with torch.no_grad():
            for s in S:
                b = tok(s, return_tensors='pt')
                lo = m(**b).logits[0]
                i = (b['input_ids'][0] == tok.mask_token_id).nonzero()[0, 0]
                print(mid, attn, s, '->', tok.convert_ids_to_tokens(lo[i].topk(5).indices.tolist()), flush=True)
            b = tok(['Il re di Francia mandò un ambasciatore a Venezia.', 'Roma.'], return_tensors='pt', padding=True)
            hs = m.base_model(**b).last_hidden_state if hasattr(m, 'base_model') else None
            with torch.autocast('cpu', dtype=torch.bfloat16):
                hb = m.base_model(**b).last_hidden_state
            print(mid, attn, 'last hidden: mean |h|', float(hs.abs().mean()), 'std', float(hs.std()), 'max', float(hs.abs().max()),
                  'bf16 rel diff', float((hb.float() - hs).abs().mean() / hs.abs().mean()), flush=True)
        del m
