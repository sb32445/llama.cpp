#!/usr/bin/env python3
"""Build a PTQ1_0 GGUF with the community MTP head (blk.64.*) taken from the PQ2_0+MTP file.

usage: make_ptq1_mtp.py <ptq1_0.gguf> <pq2_0-mtp.gguf> <out.gguf>
The trunk tensors come unchanged from the PTQ1_0 file (decoded weights are bit-identical to PQ2_0).
Only the 15 head tensors, block_count 65, nextn_predict_layers 1 and the model name change.
Needs the `gguf` Python package (pip install gguf, or set GGUF_PY to llama.cpp/gguf-py).
"""
import sys, os
if os.environ.get('GGUF_PY'):
    sys.path.insert(0, os.environ['GGUF_PY'])
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

if len(sys.argv) != 4:
    sys.exit(__doc__)
ptq_path, mtp_path, out_path = sys.argv[1:4]
rp, rm = GGUFReader(ptq_path), GGUFReader(mtp_path)
arch = rp.fields['general.architecture'].contents()
assert arch == rm.fields['general.architecture'].contents() == 'qwen35'

# the head tensors are exactly those the MTP file has beyond the trunk
trunk = {t.name for t in rp.tensors}
head = [t for t in rm.tensors if t.name not in trunk]
assert len(head) == 15 and all(t.name.startswith('blk.64.') for t in head), [t.name for t in head]
assert rp.fields['qwen35.block_count'].contents() == 64 and rm.fields['qwen35.block_count'].contents() == 65
assert rm.fields['qwen35.nextn_predict_layers'].contents() == 1

w = GGUFWriter(out_path, arch, endianess=rp.endianess)
override = {'qwen35.block_count': (GGUFValueType.UINT32, 65),
            'general.name': (GGUFValueType.STRING, 'Ternary Bonsai 2 27B PTQ1_0 with adapted Qwen MTP')}
for f in rp.fields.values():
    if f.name == 'general.architecture' or f.name.startswith('GGUF.'):
        continue
    vt = f.types[0]
    sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
    if f.name in override:
        vt, val = override[f.name]
        w.add_key_value(f.name, val, vt)
    else:
        w.add_key_value(f.name, f.contents(), vt, sub_type=sub)
w.add_key_value('qwen35.nextn_predict_layers', 1, GGUFValueType.UINT32)

tensors = list(rp.tensors) + head
total = 0
for t in tensors:
    total += t.n_bytes
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=rp.endianess)
w.close()
print('geschrieben:', out_path, '%.3f GiB' % (os.path.getsize(out_path) / 2**30), '|', len(tensors), 'Tensoren,', len(head), 'Kopf-Tensoren,', '%.3f GiB Nutzdaten' % (total / 2**30))
