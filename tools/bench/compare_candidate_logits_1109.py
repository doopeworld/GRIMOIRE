"""Bounded-RAM logit comparison for the combined group64 integration check."""
import json,sys
from pathlib import Path
import numpy as np
ref,test,out=map(Path,sys.argv[1:4]);shape=(512,248320)
assert ref.stat().st_size==test.stat().st_size==512*248320*4
r=np.memmap(ref,dtype=np.float32,mode='r',shape=shape)
t=np.memmap(test,dtype=np.float32,mode='r',shape=shape)
cosines=[];kls=[];same=[]
for start in range(0,512,16):
    a=r[start:start+16].astype(np.float64);b=t[start:start+16].astype(np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    def ls(x):
        x=x-x.max(axis=1,keepdims=True)
        return x-np.log(np.exp(x).sum(axis=1,keepdims=True))
    la,lb=ls(a),ls(b)
    cosines.extend((a*b).sum(axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))
    kls.extend((np.exp(la)*(la-lb)).sum(axis=1))
    same.extend(a.argmax(axis=1)==b.argmax(axis=1))
m={'positions':512,'cosine_mean':float(np.mean(cosines)),'kl_mean':float(np.mean(kls)),'top1':float(np.mean(same))}
out.write_text(json.dumps(m,indent=2)+'\n');print(m)
assert m['cosine_mean']>=.973 and m['kl_mean']<=.07 and m['top1']>=.975,m
print('COMBINED_GROUP64_LOGITS_PASS')
