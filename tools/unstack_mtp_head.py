#!/usr/bin/env python3
"""Un-stack shisa-ai's distilled Ornith MTP head into the per-expert layout
that grimoire.cpp's MTP loader reads by name (mtp.layers.0.mlp.experts.<e>.*).

shisa-ai ships 19 tensors with experts stacked as [E,2I,H] / [E,H,I]; the
runtime MTP loader (grimoire.cpp:2300) resolves experts.0.gate_proj.weight
and friends individually, and unlike the main-model loader it has no fused
fallback.  Gate/up ordering follows qwen35_loader.cpp:371-375: gate is the
first moe_inter rows, up the next.
"""
import json, struct, sys, os

SRC = '/mnt/storage/Models/Ornith-MTP-shisa/model-mtp.safetensors'
DST = '/mnt/storage/Models/Ornith-MTP-shisa/model-mtp-native.safetensors'
E, I, H = 256, 512, 2048
BF16 = 2  # bytes per element

def read_header(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
    return f, hdr, 8 + n

f, hdr, data0 = read_header(SRC)
meta = hdr.pop('__metadata__', None)

gu = hdr['mtp.layers.0.mlp.experts.gate_up_proj']
dn = hdr['mtp.layers.0.mlp.experts.down_proj']
assert gu['shape'] == [E, 2 * I, H], gu['shape']
assert dn['shape'] == [E, H, I], dn['shape']
assert gu['dtype'] == 'BF16' and dn['dtype'] == 'BF16'

# Read the two stacked blobs once.
f.seek(data0 + gu['data_offsets'][0])
gu_bytes = f.read(gu['data_offsets'][1] - gu['data_offsets'][0])
f.seek(data0 + dn['data_offsets'][0])
dn_bytes = f.read(dn['data_offsets'][1] - dn['data_offsets'][0])

# Pass-through tensors: everything except the two stacked expert blobs.
passthru = {k: v for k, v in hdr.items()
            if k not in ('mtp.layers.0.mlp.experts.gate_up_proj',
                         'mtp.layers.0.mlp.experts.down_proj')}

out_hdr, blobs, off = {}, [], 0

def add(name, shape, payload):
    global off
    out_hdr[name] = {'dtype': 'BF16', 'shape': shape,
                     'data_offsets': [off, off + len(payload)]}
    blobs.append(payload)
    off += len(payload)

for name, v in sorted(passthru.items()):
    f.seek(data0 + v['data_offsets'][0])
    add(name, v['shape'], f.read(v['data_offsets'][1] - v['data_offsets'][0]))

gu_e = 2 * I * H * BF16          # bytes per expert in gate_up
half = I * H * BF16
dn_e = H * I * BF16
for e in range(E):
    b = e * gu_e
    p = f'mtp.layers.0.mlp.experts.{e}.'
    add(p + 'gate_proj.weight', [I, H], gu_bytes[b:b + half])
    add(p + 'up_proj.weight',   [I, H], gu_bytes[b + half:b + gu_e])
    add(p + 'down_proj.weight', [H, I], dn_bytes[e * dn_e:(e + 1) * dn_e])

if meta is not None:
    out_hdr['__metadata__'] = meta
blob = json.dumps(out_hdr, separators=(',', ':')).encode()
pad = (-len(blob)) % 8
blob += b' ' * pad

with open(DST, 'wb') as o:
    o.write(struct.pack('<Q', len(blob)))
    o.write(blob)
    for b in blobs:
        o.write(b)
f.close()

n_t = len([k for k in out_hdr if k != '__metadata__'])
print(f'wrote {DST}')
print(f'  tensors {n_t} (expect 785), bytes {os.path.getsize(DST):,}')
