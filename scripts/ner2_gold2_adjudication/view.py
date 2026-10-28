import sys,json,collections
from merge import *
AB={'r3_mdeberta_HM':'D','r3_mmbert_base_HM':'M','r3_xlmr_large_H':'X','r3_hmteams_HM':'H','flair_ner_english_large':'F','llm_qwen35_9b':'Q'}
def status0(d):
    ann='annotator' in d['sources']
    sure=ann and d['sources']['annotator']['certainty']=='sure'
    ns=sum(1 for k in d['sources'] if k!='annotator')
    return 'agreed' if sure and ns>=2 else 'rest'
def srcs(d):
    s=''
    a=d['sources'].get('annotator')
    if a: s+='A'+('!' if a['certainty']=='sure' else '?')
    s+=''.join(AB[k] for k in SYS if k in d['sources'])
    return s
if __name__=='__main__':
    lang=sys.argv[1]; mode=sys.argv[2]
    o=build()
    if mode=='agreed':
        c=collections.Counter()
        for pid,L in o.items():
            if P[pid]['lang']!=lang: continue
            t=P[pid]['tokens']
            for d in L:
                if status0(d)=='agreed': c[' '.join(t[d['start']:d['end']])]+=1
        print(dict(c))
    else:
        for pid,L in o.items():
            if P[pid]['lang']!=lang: continue
            t=P[pid]['tokens']
            rest=[d for d in L if status0(d)=='rest']
            if not rest: continue
            print('##',pid.split('/',2)[2][:60],'|',P[pid]['year'],P[pid]['collection'])
            for d in rest:
                a,b=d['start'],d['end']
                ov=[(x['start'],x['end']) for x in L if x is not d and x['start']<b and a<x['end']]
                an=d['sources'].get('annotator')
                print(f"  [{a},{b}] {srcs(d):9} {' '.join(t[max(0,a-9):a])} «{' '.join(t[a:b])}» {' '.join(t[b:b+6])}"+(f"   OV{ov}" if ov else '')+(f"   ANN:{an['reason']}" if an and an['reason'] else ''))
