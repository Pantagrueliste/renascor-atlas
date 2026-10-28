"""FROZEN copy of the PERSON name matcher the audit of 2026-10-16 wrote (``person-authorities/namecmp.py``), byte for byte below this docstring.

``namegate.py`` is a ported, evolving version of it. The blocking tests of the v2 tables (``audit_v2.py``) judge the accepted
authority ids with THIS code, so that a change in the gates cannot change the yardstick. Do not change its behaviour.
"""
import re, unicodedata, difflib
from collections import Counter

PART = set('''von van vom zu zur zum de der den di da del della dei degli dello delle de' d l' du des le la les el al lo y e et und and the of ibn bin ben abu aus op ten ter te dal dalla dallo dai da' do dos das af av von' t' 't la' l del' a ad in sub
ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx jr jun sr sen
sir mr mrs dr lord lady saint st sainte san santo santa sanctus sancta sanctae sancti beatus beata king queen duke duchess earl count countess prince princess emperor empress pope bishop archbishop cardinal captain colonel general
herr frau graf grafen grafin herzog herzogs herzogin furst fursten furstin churfurst churfursten kurfurst kurfursten kaiser kaiserin konig konigin bischof erzbischof papst pabst sankt hl heilige heiliger pfalzgraf markgraf landgraf freiherr freiin ritter
don dona fra frate maestro messer mons monsieur madame mademoiselle sieur sr sra roi reine duc duchesse comte comtesse princesse pape eveque seigneur sire
rex regina dux ducis comes comitis princeps imperator imperatrix papa episcopus archiepiscopus doctor dominus domina magister frater pater mater cardinalis
re regno regina duca duchessa conte contessa principe principessa imperatore papa vescovo cardinale sig signor signora
rey reina conde condesa principe emperador
koning koningin hertog graaf prins keizer paus
the elder younger junior senior genannt gen called alias dit sive seu vel aka also known as of
mister master mistress lady dame sieur madame
dem das ein eine eines''' .split())
PART |= {'s','st','ste','th','nd','rd','1','2','3','4','5','6','7','8','9'}

def fold(s):
    s=unicodedata.normalize('NFKD',str(s).casefold())
    s=''.join(c for c in s if not unicodedata.combining(c))
    s=s.replace('ß','ss').replace('æ','ae').replace('œ','oe').replace('ø','o').replace('đ','d').replace('ł','l').replace('ı','i').replace('þ','th').replace('ð','d')
    return s

def toks(s, keep_all=False):
    s=fold(s)
    s=re.sub(r'\([^)]*\)',' ',s) if not keep_all else s
    s=re.sub(r"[’'`]",' ',s)
    out=[t for t in re.split(r'[^a-z0-9]+',s) if t]
    return out

def sig(s):
    return [t for t in toks(s) if t not in PART and len(t)>=2 and not t.isdigit()]

def phon(t):
    t=t.replace('ae','e').replace('oe','e').replace('ue','u').replace('ph','f').replace('th','t').replace('ck','k').replace('ch','k').replace('sch','s').replace('tz','z').replace('cz','z').replace('w','v').replace('y','i').replace('j','i').replace('c','k').replace('q','k')
    t=re.sub(r'(.)\1+',r'\1',t)
    t=t.replace('h','')
    return t
ENDS=('orum','arum','ius','ium','us','um','is','os','es','ae','as','ii','i','o','a','e','s')
def stem(t):
    p=phon(t)
    for e in ENDS:
        if len(p)>len(e)+3 and p.endswith(phon(e)): return p[:-len(phon(e))]
    return p

def tmatch(a,b):
    if a==b: return True
    if len(a)<3 or len(b)<3: return False
    pa,pb=phon(a),phon(b)
    if pa==pb: return True
    sa,sb=stem(a),stem(b)
    if sa==sb and len(sa)>=3: return True
    m=min(len(sa),len(sb))
    if m>=4:
        # prefix rule
        k=0
        while k<m and sa[k]==sb[k]: k+=1
        if k>=m-1 and k>=4: return True
        if k>=5 and k>=0.75*m: return True
    ka,kb=re.sub('[aeiou]','',sa),re.sub('[aeiou]','',sb)
    if len(ka)>=3 and ka==kb and sa[:1]==sb[:1]: return True
    if len(ka)>=5 and len(kb)>=5 and sa[:1]==sb[:1] and difflib.SequenceMatcher(None,ka,kb).ratio()>=0.8: return True
    if min(len(a),len(b))>=5 and difflib.SequenceMatcher(None,a,b).ratio()>=0.84: return True
    if m>=4 and difflib.SequenceMatcher(None,sa,sb).ratio()>=0.86: return True
    return False

def build_weak(label_lists, thresh=40):
    c=Counter()
    for ls in label_lists:
        seen=set()
        for l in ls:
            seen|=set(sig(l))
        c.update(seen)
    return {t for t,n in c.items() if n>=thresh}, c

def compare(entry_names, labels, weak):
    """returns (status, matched_tokens, entry_sig, label_sig)
    status: ok | given_only | mismatch | no_label | no_name"""
    ent=[]
    for n in entry_names:
        ent.append(sig(n))
    lab=set()
    for l in labels: lab|=set(sig(l))
    if not labels or not lab: return 'no_label',[],[],[]
    ent_all=[t for ts in ent for t in ts]
    if not ent_all: return 'no_name',[],[],sorted(lab)
    strong=[];weakm=[]
    for t in set(ent_all):
        hit=any(tmatch(t,l) for l in lab)
        if hit:
            (weakm if t in weak else strong).append(t)
    if strong: return 'ok',strong+weakm,sorted(set(ent_all)),sorted(lab)
    if weakm:
        # all primary-name tokens given names? ok if the primary name is made only of weak tokens that matched
        prim=ent[0] if ent else []
        if prim and all(any(tmatch(t,l) for l in lab) for t in prim): return 'ok_weak',weakm,sorted(set(ent_all)),sorted(lab)
        return 'given_only',weakm,sorted(set(ent_all)),sorted(lab)
    return 'mismatch',[],sorted(set(ent_all)),sorted(lab)

ROMAN=set('titus marcus gaius caius lucius publius quintus sextus gnaeus aulus decimus tiberius appius manius servius flavius julius iulius claudius cornelius valerius aurelius antonius marc marcos lucio gaio tito'.split())
