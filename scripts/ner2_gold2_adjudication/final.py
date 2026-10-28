import json,collections,importlib,unicodedata
N=lambda x:unicodedata.normalize('NFC',x)
from merge import *
from view import status0
LANGS=['en','la','it','es','fr','de']
NAMES={'en':'anglais','la':'latin','it':'italien','es':'espagnol','fr':'français','de':'allemand'}
DEC={}
for l in LANGS:
    DEC[l]=importlib.import_module('dec_'+l).D
ST={'P':'proposed','R':'rejected','C':'to_check'}
SHORT={'r3_mdeberta_HM':'mDeBERTa','r3_mmbert_base_HM':'mmBERT','r3_xlmr_large_H':'XLM-R large','r3_hmteams_HM':'hmTEAMS','flair_ner_english_large':'flair EN','llm_qwen35_9b':'Qwen3.5-9B (LLM)'}
spans_all=build()
# map decisions
def find_pid(lang,key):
    c=[pid for pid in P if P[pid]['lang']==lang and (N(pid.split('/',2)[2])==N(key) or (len(key)>=60 and N(pid.split('/',2)[2]).startswith(N(key))))]
    assert len(c)==1,(lang,key,c)
    return c[0]
decmap={}
for l in LANGS:
    for key,lst in DEC[l].items():
        pid=find_pid(l,key)
        for a,b,s,r in lst:
            assert (pid,a,b) not in decmap,(pid,a,b)
            decmap[(pid,a,b)]=(ST[s],r)
used=set(); out=[]; stats=collections.Counter()
for pid,p in P.items():
    rec={k:v for k,v in p.items()}
    rec['span_convention']='start inclusive, end exclusive, token indices into tokens; char offsets from token_starts'
    sp=[]
    for i,d in enumerate(spans_all[pid]):
        a,b=d['start'],d['end']
        s0=status0(d)
        src=d['sources']
        nsys=[k for k in SYS if k in src]
        if s0=='agreed':
            status='agreed'; reason=f"annotateur sûr et {len(nsys)} système(s) d'accord"
        else:
            assert (pid,a,b) in decmap,(pid,a,b,p['tokens'][a:b])
            status,reason=decmap[(pid,a,b)]; used.add((pid,a,b))
        cs=p['token_starts'][a]; ce=p['token_starts'][b-1]+len(p['tokens'][b-1])
        an=src.get('annotator')
        sources={'annotator':(an['certainty'] if an else None)}
        for m in SYS:
            sources[m]= (True if m in src else (False if (m!='flair_ner_english_large' or p['lang']=='en') else None))
        sp.append({'span_id':f"{pid}#{i}",'start':a,'end':b,'text':' '.join(p['tokens'][a:b]),'surface':p['text'][cs:ce],
                   'char_start':cs,'char_end':ce,'status':status,'reason':reason,'sources':sources,
                   'annotator_reason':(an['reason'] if an else None),
                   'confs':{m:src[m]['conf'] for m in SYS if m in src and src[m].get('conf') is not None}})
        stats[(p['lang'],status)]+=1
    rec['spans']=sp
    out.append(rec)
assert used==set(decmap),set(decmap)-used
with open(D+'proposed_gold_v2.jsonl','w') as f:
    for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
json.dump({'stats':{f"{k[0]}/{k[1]}":v for k,v in stats.items()}},open('stats.json','w'))
print(sorted(stats.items()))
# ---------- scoring
def pred_sets(pid,m):
    return {(s['s'],s['e']) for s in PR[pid]['models'].get(m,[])}
def enc_vote(pid,minn,minconf):
    c=collections.defaultdict(list)
    for m in SYS[:4]:
        for s in PR[pid]['models'][m]:
            c[(s['s'],s['e'])].append(s['conf'] or 0)
    return {k for k,v in c.items() if len(v)>=minn and max(v)>=minconf}
SYSROWS=[(m,lambda pid,m=m:pred_sets(pid,m)) for m in SYS]
SYSROWS+= [('vote_ge2',lambda pid:enc_vote(pid,2,0)),('vote_ge2_conf98',lambda pid:enc_vote(pid,2,0.98))]
LABEL=dict(SHORT); LABEL['vote_ge2']='vote ≥ 2 encodeurs'; LABEL['vote_ge2_conf98']='vote ≥ 2 encodeurs et conf ≥ 0.98'
def ov(a,b): return a[0]<b[1] and b[0]<a[1]
def score(lang,fn,mode):
    tp=fp=fn_=0; npred=0
    for pid in P:
        if P[pid]['lang']!=lang and lang!='all': continue
        ref={(s['start'],s['end']) for s in next(r for r in out if r['id']==pid)['spans'] if s['status'] in('agreed','proposed')}
        chk={(s['start'],s['end']) for s in next(r for r in out if r['id']==pid)['spans'] if s['status']=='to_check'}
        pr=fn(pid)
        if mode=='strict':
            tp+=len(pr&ref); fp+=len(pr-ref); fn_+=len(ref-pr)
        elif mode=='strict_ignoreC':
            tp+=len(pr&ref); fp+=len(pr-ref-chk); fn_+=len(ref-pr)
        else: # lenient overlap
            tp+=sum(1 for x in pr if any(ov(x,r) for r in ref))
            fp+=sum(1 for x in pr if not any(ov(x,r) for r in ref) and not any(ov(x,c) for c in chk))
            fn_+=sum(1 for r in ref if not any(ov(x,r) for x in pr))
    p_=tp/(tp+fp) if tp+fp else 0; r_=tp/(tp+fn_) if tp+fn_ else 0
    f=2*p_*r_/(p_+r_) if p_+r_ else 0
    return p_,r_,f,tp,fp,fn_
res={}
for mode in ['strict','strict_ignoreC','lenient']:
    for l in LANGS+['all']:
        for m,fn in SYSROWS:
            if m=='flair_ner_english_large' and l not in('en',): continue
            res[(mode,l,m)]=score(l,fn,mode)
json.dump({f"{k[0]}|{k[1]}|{k[2]}":v for k,v in res.items()},open('scores.json','w'))
def table(mode,title):
    L=[f"### {title}",'',"| Système | "+" | ".join(f"{NAMES[l]}" for l in LANGS)+" | toutes langues |","|---|"+"---|"*(len(LANGS)+1)]
    for m,_ in SYSROWS:
        cells=[]
        for l in LANGS+['all']:
            r=res.get((mode,l,m))
            cells.append('—' if r is None else f"{r[2]:.2f} ({r[0]:.2f}/{r[1]:.2f})")
        L.append(f"| {LABEL[m]} | "+" | ".join(cells)+" |")
    return '\n'.join(L)
tabs={'strict':table('strict',"A. Correspondance exacte, spans «à vérifier» comptés comme faux positifs"),
      'strict_ignoreC':table('strict_ignoreC',"B. Correspondance exacte, spans «à vérifier» ignorés"),
      'lenient':table('lenient',"C. Correspondance par recouvrement (frontières tolérées), «à vérifier» ignorés")}
open('tabs.json','w').write(json.dumps(tabs,ensure_ascii=False))
for k,v in tabs.items(): print(v);print()
# counts per language
cnt=collections.defaultdict(lambda:collections.Counter())
for r in out:
    cnt[r['lang']]['passages']+=1; cnt[r['lang']]['tokens']+=len(r['tokens'])
    for s in r['spans']: cnt[r['lang']][s['status']]+=1; cnt[r['lang']]['spans']+=1
    if any(s['status'] in('agreed','proposed') for s in r['spans']) is False: cnt[r['lang']]['no_place']+=1
json.dump({l:dict(c) for l,c in cnt.items()},open('counts.json','w'))
print({l:dict(c) for l,c in cnt.items()})
