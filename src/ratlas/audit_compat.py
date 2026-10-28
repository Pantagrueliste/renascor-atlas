"""FROZEN copy of the PLACE name matcher the audit of 2026-10-16 wrote (``place-authorities/compat.py``).

Used by ``audit_v2.py`` as the yardstick for the accepted place ids (``namegate.place_relation`` is the evolving port). Changes from the
original, none of them in behaviour: the ``sys.path`` hack and the monkey patch of ``reconcile_namecheck.nrm`` are replaced by a local
``name_keys`` that uses the audit's ``nrm``, and the unused output directory constant is gone.
"""
import difflib
import re
import unicodedata

import numpy as np

STOP = {'stadt', 'kloster', 'bistum', 'fluss', 'insel', 'kirche', 'schloss', 'burg', 'haus', 'republik', 'furstentum',
        'konigreich', 'herzogtum', 'grafschaft', 'church', 'castle', 'river', 'island', 'abbey', 'palace', 'street',
        'bridge', 'gate', 'parish', 'hall', 'manor', 'hospital', 'house', 'inn', 'school', 'college', 'market'}


def name_keys(name, variants) -> set:
    out = set()
    for n in [name] + list(variants if variants is not None else []):
        if not isinstance(n, str) or not n:
            continue
        for c in [n, re.sub(r'\([^)]*\)', ' ', n)] + re.split(r'[,/()]', n):
            k = nrm(c)
            if len(k) > 1 and k not in STOP:
                out.add(k)
    return out


def nrm(s):
    s=str(s).lower().replace('\u00e6','ae').replace('\u0153','oe').replace('\u017f','s').replace('\u00df','ss').replace('\u00f8','o').replace('\u0142','l').replace('\u0111','d')
    s=unicodedata.normalize('NFKD',s)
    s=''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9 ]+',' ',s).split())
GENERIC=set('''saint sainte sankt santa santo santi sant stadt city town village kingdom republic county province district region island isola insel lake river fluss mount monte berg
castle kloster abbey church kirche diocese bistum duchy duche herzogtum grafschaft roman empire reich holy saints sancti sanctus sancta oppidum civitas urbs regnum comitatus ducatus episcopus
nova novum novi neu neue neuen new alt alte altstadt nuova nuovo nueva nuevo nouvelle nouveau grosse grosser gross klein kleine upper lower oberen unteren ober unter under super inferior superior
municipality commune comune gemeinde region provinz provincia province departement department kreis landkreis bezirk canton kanton
north south east west nord sud ost west'''.split())
SMALL=set('st saint sainte san santa santo sankt sant de del della delle di du des la le les el al the of von vom zu zum zur am an in bei im ob upon on sur sous ad apud et y e da do dos das ten ter van den der het aan op te'.split())
def fold(k):
    toks=[t for t in k.split() if t not in SMALL]
    k=' '.join(toks) if toks else k
    k=k.replace('sch','s').replace('ph','f').replace('th','t').replace('ck','k').replace('qu','k').replace('ae','e').replace('oe','e').replace('ij','i').replace('ss','s').replace('tz','z')
    k=re.sub(r'c(?=[aouklrt]|$)','k',k).replace('c','s')
    k=k.translate(str.maketrans({'j':'i','v':'u','y':'i','w':'u','q':'k','x':'s','z':'s','h':'','k':'k'}))
    k=re.sub(r'(.)\1+',r'\1',k)
    return ' '.join(k.split())
LAT=('orum','arum','iae','ium','ius','ia','um','us','is','ae','os','es','en','er','a','e','i','o','s')
def stem(k):
    out=[]
    for t in k.split():
        for e in LAT:
            if t.endswith(e) and len(t)-len(e)>=3:
                t=t[:-len(e)]; break
        out.append(t)
    return ' '.join(out)
def tri(s):
    s=f'  {s} '
    return {s[i:i+3] for i in range(len(s)-2)}
_tri={}
def trig(s):
    v=_tri.get(s)
    if v is None: v=_tri[s]=tri(s)
    return v
def keys_of(name,variants):
    return name_keys(name,variants)
def auth_keys(labels):
    out=set()
    for n in labels:
        if not isinstance(n,str) or not n: continue
        for c in [n, re.sub(r'\([^)]*\)',' ',n)]+re.split(r'[,/()]',n):
            k=nrm(c)
            if len(k)>1: out.add(k)
    return out
def relation(ek, ak):
    """best name relation between entry keys ek and authority keys ak -> (level, e, a)"""
    if not ek or not ak: return ('noname' if not ek else 'noauthname',None,None)
    inter=ek&ak
    if inter:
        k=sorted(inter)[0]; return ('exact',k,k)
    ef={fold(k):k for k in ek}; af={fold(k):k for k in ak}
    common=set(ef)&set(af)
    common={c for c in common if len(c)>=2}
    if common:
        c=sorted(common)[0]; return ('fold',ef[c],af[c])
    es={stem(f):k for f,k in ef.items() if len(stem(f))>=4}; as_={stem(f):k for f,k in af.items() if len(stem(f))>=4}
    c=set(es)&set(as_)
    if c:
        c=sorted(c)[0]; return ('stem',es[c],as_[c])
    # whole-token containment / significant shared token
    for e in sorted(ek):
        et=[t for t in e.split() if len(t)>=4 and t not in GENERIC and t not in SMALL]
        ef_=fold(e)
        for a in ak:
            if len(e)>=4 and len(a)>=4:
                if re.search(r'(?<![a-z0-9])'+re.escape(e)+r'(?![a-z0-9])',a) or re.search(r'(?<![a-z0-9])'+re.escape(a)+r'(?![a-z0-9])',e):
                    if e not in STOP and a not in STOP: return ('substring',e,a)
            at=a.split()
            if et and any(t in at for t in et): return ('token',e,a)
    # fuzzy
    best=(0,None,None)
    for f,e in ef.items():
        if len(f)<4: continue
        te=trig(f)
        for fa,a in af.items():
            if len(fa)<4 or abs(len(fa)-len(f))>max(4,0.5*len(f)): continue
            ta=trig(fa)
            if len(te&ta)*2/(len(te)+len(ta))<0.4: continue
            ra=difflib.SequenceMatcher(None,f,fa).ratio()
            if ra>best[0]: best=(ra,e,a)
    if best[0]>=0.8: return ('fuzzy%.2f'%best[0],best[1],best[2])
    return ('none%.2f'%best[0],best[1],best[2])

def hav(la1,lo1,la2,lo2):
    if any(v is None or v!=v for v in (la1,lo1,la2,lo2)): return np.nan
    p1,p2=np.radians(la1),np.radians(la2)
    a=np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(np.radians(lo2-lo1)/2)**2
    return float(2*6371.0088*np.arcsin(np.sqrt(min(1,a))))

