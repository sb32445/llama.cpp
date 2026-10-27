#!/usr/bin/env python3
"""Check that every tensor of the built file is byte-identical to its source (SHA-256, type and shape).

usage: verify_merge.py <built.gguf> <ptq1_0.gguf> <pq2_0-mtp.gguf>
Trunk tensors are compared with the PTQ1_0 file, blk.64.* (head) with the PQ2_0+MTP file.
Needs the `gguf` Python package (or GGUF_PY=<llama.cpp/gguf-py>).
"""
import sys, os, hashlib, numpy as np
if os.environ.get('GGUF_PY'):
    sys.path.insert(0, os.environ['GGUF_PY'])
from gguf import GGUFReader
if len(sys.argv) != 4:
    sys.exit(__doc__)
out, ptq, mtp = (GGUFReader(p) for p in sys.argv[1:4])
def h(t): return hashlib.sha256(np.ascontiguousarray(t.data).tobytes()).hexdigest()
O={t.name:t for t in out.tensors}
bad=[]; n=0
for src,label in ((ptq,'PTQ'),(mtp,'MTP')):
    for t in src.tensors:
        if label=='MTP' and not t.name.startswith('blk.64.'): continue
        o=O.get(t.name)
        if o is None or int(o.tensor_type)!=int(t.tensor_type) or list(o.shape)!=list(t.shape) or h(o)!=h(t): bad.append((label,t.name))
        n+=1
print('geprueft:',n,'Tensoren, Abweichungen:',bad[:5], '| Ausgabe hat',len(O),'Tensoren')
def kv(r):
    d={}
    for k,f in r.fields.items():
        if k.startswith('GGUF.'): continue
        try: v=f.contents()
        except Exception: v='?'
        d[k]=str(v)[:200] if not hasattr(v,'__len__') or isinstance(v,str) else 'len%d:%s'%(len(v),hash(tuple(map(str,v))))
    return d
a,b=kv(ptq),kv(out)
print('KV-Unterschiede PTQ -> Ausgabe:')
for k in sorted(set(a)|set(b)):
    if a.get(k)!=b.get(k): print('  ',k,':',a.get(k,'<fehlt>')[:60],'->',b.get(k,'<fehlt>')[:60])
