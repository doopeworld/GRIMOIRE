#!/usr/bin/env python3
# GRIMOIRE — Copyright (C) 2026 Ian Ernst
# SPDX-License-Identifier: GPL-3.0-or-later
# CPU-only fixture data with an independent rational BF16 oracle.
import json, random, struct, sys
from fractions import Fraction
from pathlib import Path
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
def bf(value):
    value = Fraction(value)
    if not value: return 0
    sign = 0x8000 if value < 0 else 0; value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    pow2 = lambda n: Fraction(2**n) if n >= 0 else Fraction(1, 2**(-n))
    if value < pow2(exponent): exponent -= 1
    scaled = value / pow2(exponent-7)
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2*r > scaled.denominator or (2*r == scaled.denominator and q & 1): q += 1
    if q == 256: q = 128; exponent += 1
    return sign | ((exponent+127) << 7) | (q-128)
def bvalue(bits):
    return Fraction(struct.unpack('<f', struct.pack('<I', bits << 16))[0])
def half(bits):
    return Fraction(struct.unpack('<e', struct.pack('<H', bits))[0])
def expected(code, bits):
    if not half(bits) and code < 0: return 0x8000  # IEEE signed zero, as in BF16 multiplication.
    return bf(code*bvalue(bf(half(bits))))
rng = random.Random(20261010)
scales = sorted(set([0, 1, 2, 3, 0x3c00, 0x3bff, 0x7bff, 0x3401, 0x1a1b] + [rng.randrange(0x7c00) for _ in range(192)]))
with (root/'oracle.txt').open('w') as f:
    for bits in scales:
        for code in range(-128,128): f.write(f'{code} {bits} {expected(code,bits)}\n')
def tensor_file(path, tensors):
    header = {}; data = bytearray()
    for name,(dtype,shape,payload) in sorted(tensors.items()):
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(data),len(data)+len(payload)])
        data.extend(payload)
    text = json.dumps(header, separators=(',',':')).encode(); text += b' ' * (-len(text)%8)
    path.write_bytes(struct.pack('<Q',len(text))+text+data)
H,V=8,4
prefix='model.language_model.'
weights={prefix+'embed_tokens.weight':('BF16',[V,H],struct.pack('<H',0x3f80)*(V*H)),prefix+'norm.weight':('BF16',[H],b'\0'*(2*H))}
for name in ['input_layernorm','post_attention_layernorm']:
    weights[prefix+'layers.0.'+name+'.weight']=('BF16',[H],b'\0'*(2*H))
for name in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
    weights[prefix+'layers.0.'+name+'.weight']=('BF16',[H,H],b'\0'*(2*H*H))
weights['lm_head.weight']=('BF16',[V,H],b'\0'*(2*V*H))
codes=bytes((i*7)%256 for i in range(V*H))
scale_bits=[0x1a1b,0x3401,0x3bff,0]
for case in ['early','late','missing','bits','packed','path','shape','dtype','negative','nan','unscaled','dense']:
    folder=root/case; folder.mkdir(exist_ok=True)
    side='a-side.safetensors' if case=='early' else 'z-side.safetensors'
    config=dict(model_type='qwen3_5', hidden_size=H, vocab_size=V, num_hidden_layers=1,
                num_attention_heads=1,num_key_value_heads=1,head_dim=H,intermediate_size=H,
                layer_types=['full_attention'],attn_output_gate=False,dtype='bfloat16')
    if case not in ['unscaled','dense']:
        config['embed_tokens_quant']=dict(bits=4 if case=='bits' else 8,packed=case=='packed',side_file='../'+side if case=='path' else side,key_weight='w',key_scale='s',scheme='per_row_absmax_symmetric')
    (folder/'config.json').write_text(json.dumps(config))
    tensor_file(folder/'model.safetensors',weights)
    if case not in ['missing','dense']:
        vals=list(scale_bits)
        if case=='negative': vals[1]=0xbc00
        if case=='nan': vals[1]=0x7e00
        dtype='F32' if case=='dtype' else 'F16'
        payload=struct.pack('<4f',1,1,1,0) if dtype=='F32' else struct.pack('<4H',*vals)
        wn=prefix+'embed_tokens.weight' if case=='unscaled' else 'w'
        sn=prefix+'embed_tokens.weight_scale' if case=='unscaled' else 's'
        tensor_file(folder/side,{wn:('I8',[V,H],codes),sn:(dtype,[V] if case=='shape' else [V,1],payload)})
print('generated', len(scales)*256, 'independent oracle values and loader cases')
