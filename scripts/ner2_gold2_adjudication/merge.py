import json,collections
D='/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2/gold2/'
P={}
for x in open(D+'passages.jsonl'):
    r=json.loads(x); P[r['id']]=r
ANN={}
for l in ['en','la','it','es','fr','de']:
    for x in open(D+f'ann_{l}.jsonl'):
        r=json.loads(x); ANN[r['id']]=r
PR={}
for x in open(D+'predictions.jsonl'):
    r=json.loads(x); PR[r['id']]=r
SYS=['r3_mdeberta_HM','r3_mmbert_base_HM','r3_xlmr_large_H','r3_hmteams_HM','flair_ner_english_large','llm_qwen35_9b']
def build():
    out={}
    for pid,p in P.items():
        sp=collections.OrderedDict()
        def add(a,b,src,info=None):
            d=sp.setdefault((a,b),{'start':a,'end':b,'sources':{}})
            d['sources'][src]=info if info is not None else True
        for s in ANN[pid]['spans']:
            a=s.get('start',s.get('token_start'));b=s.get('end',s.get('token_end'))
            add(a,b,'annotator',{'certainty':s['certainty'],'reason':s.get('reason',''),'kind':s.get('kind')})
        for m in SYS:
            for s in PR[pid]['models'].get(m,[]):
                add(s['s'],s['e'],m,{'conf':s.get('conf')})
        out[pid]=sorted(sp.values(),key=lambda d:(d['start'],d['end']))
    return out
if __name__=='__main__':
    o=build()
    c=collections.Counter()
    for pid,L in o.items():
        lang=P[pid]['lang']
        for d in L:
            ann='annotator' in d['sources']
            sure= ann and d['sources']['annotator']['certainty']=='sure'
            ns=sum(1 for k in d['sources'] if k!='annotator')
            c[(lang,'agreed' if sure and ns>=2 else 'rest')]+=1
    print(sorted(c.items()))
