#!/usr/bin/env python3
"""Verify llama.cpp's qwen35moe conversion transforms against REAL tensor values.

Compares tensors of the published BF16 GGUF with the original HF safetensors of the same model,
fetching only the needed byte ranges over HTTP. This checks docs/gguf/FORMAT_SUPPORT.md section 4
against the FILES, independently of any GRIMOIRE adapter code (CLAUDE.md rule 21).

Dev tool, not runtime. Needs numpy:  /tmp/gguf-golden-venv/bin/python tests/gguf/verify_transforms.py
"""
import json, struct, sys, urllib.request
import numpy as np

GGUF_URL = ("https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/"
            "12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-BF16.gguf")
HF_REV = "10fbf86fed7ecee4a061f8b499a618f46001cac1"
HF_BASE = f"https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/{HF_REV}/"
GGUF_TYPES = {0: ("F32", 4), 30: ("BF16", 2)}

def get(url, lo=None, hi=None):
    req = urllib.request.Request(url)
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if lo is not None and len(data) != hi - lo:
        raise RuntimeError(f"short range read {len(data)} != {hi - lo}")
    return data

def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16_rne(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)

# ---------------- GGUF side (header parsed once) ----------------
def parse_gguf_header(buf):
    o = 4
    def u(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, buf, o)[0]; o += struct.calcsize("<" + fmt); return v
    def s():
        nonlocal o
        n = u("Q"); v = buf[o:o + n].decode(); o += n; return v
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        nonlocal o
        if t == 8: s()
        elif t == 9:
            et = u("I"); n = u("Q")
            for _ in range(n): skip(et)
        else: o += sizes[t]
    ver, nt, nkv = u("I"), u("Q"), u("Q")
    align = 32
    for _ in range(nkv):
        k = s(); t = u("I")
        if k == "general.alignment": align = u("I")
        else: skip(t)
    tens = {}
    for _ in range(nt):
        name = s(); nd = u("I"); ne = [u("Q") for _ in range(nd)]; ty = u("I"); off = u("Q")
        tens[name] = (ne, ty, off)
    data_start = (o + align - 1) // align * align
    return tens, data_start

print("fetching GGUF header ...", flush=True)
G_TENS, G_DATA = parse_gguf_header(get(GGUF_URL, 0, 16 << 20))

def gguf_tensor(name, row_lo=None, row_hi=None):
    """Return (array shaped [rows, ne0] or flat, ne). Optionally only rows [row_lo,row_hi) of the
    flattened 2-D view (rows = product of ne[1:])."""
    ne, ty, off = G_TENS[name]
    tname, es = GGUF_TYPES[ty]
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    lo_r, hi_r = (0, rows) if row_lo is None else (row_lo, row_hi)
    lo = G_DATA + off + lo_r * ne[0] * es
    hi = G_DATA + off + hi_r * ne[0] * es
    raw = get(GGUF_URL, lo, hi)
    a = bf16_to_f32(raw) if tname == "BF16" else np.frombuffer(raw, dtype=np.float32)
    return a.reshape(hi_r - lo_r, ne[0]) if len(ne) > 1 else a, ne

# ---------------- HF safetensors side ----------------
INDEX = json.loads(get(HF_BASE + "model.safetensors.index.json"))["weight_map"]
_hdr_cache = {}
def st_header(shard):
    if shard not in _hdr_cache:
        n = struct.unpack("<Q", get(HF_BASE + shard, 0, 8))[0]
        _hdr_cache[shard] = (json.loads(get(HF_BASE + shard, 8, 8 + n)), 8 + n)
    return _hdr_cache[shard]

def hf_tensor(name, row_lo=None, row_hi=None):
    shard = INDEX[name]
    hdr, base = st_header(shard)
    meta = hdr[name]; shape = meta["shape"]; dt = meta["dtype"]
    es = {"BF16": 2, "F32": 4, "F16": 2}[dt]
    inner = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    lo0, hi0 = meta["data_offsets"]
    if row_lo is None:
        lo, hi = base + lo0, base + hi0; rows = shape[0]
    else:
        lo = base + lo0 + row_lo * inner * es; hi = base + lo0 + row_hi * inner * es; rows = row_hi - row_lo
    raw = get(HF_BASE + shard, lo, hi)
    a = {"BF16": bf16_to_f32, "F32": lambda r: np.frombuffer(r, np.float32),
         "F16": lambda r: np.frombuffer(r, np.float16).astype(np.float32)}[dt](raw)
    return a.reshape([rows] + list(shape[1:])), shape, dt

# ---------------- transforms (llama.cpp conversion/qwen.py @711780ffc75b) ----------------
NK, NV, DK, DV = 16, 32, 128, 128
RV = NV // NK
def reorder_v(t, dim, head_dim):
    """grouped (HF) -> tiled (GGUF): _reorder_v_heads."""
    t = np.moveaxis(t, dim, 0)
    sh = t.shape
    t = t.reshape((NK, RV, head_dim) + sh[1:])
    t = np.swapaxes(t, 0, 1).reshape(sh)
    return np.moveaxis(t, 0, dim)

results = []
def check(label, got, want, note=""):
    got = np.asarray(got, np.float32); want = np.asarray(want, np.float32)
    if got.shape != want.shape:
        results.append((label, "SHAPE", f"{got.shape} vs {want.shape}", note)); return
    d = np.abs(got - want); exact = bool(np.array_equal(got.view(np.uint32), want.view(np.uint32)))
    rel = float(d.max() / max(np.abs(want).max(), 1e-30))
    results.append((label, "EXACT" if exact else "DIFF", f"maxabs={d.max():.3e} rel={rel:.3e}", note))

P = "model.language_model."
L0, L3 = P + "layers.0.", P + "layers.3."
print("comparing ...", flush=True)

# T1 + T4: ssm_a = reorder(-exp(A_log))
a_log, _, dt = hf_tensor(L0 + "linear_attn.A_log")
g, _ = gguf_tensor("blk.0.ssm_a")
check("T1+T4 ssm_a == reorder(-exp(A_log))", g, reorder_v((-np.exp(a_log.astype(np.float64))).astype(np.float32)[:, None], 0, 1)[:, 0], f"A_log {dt}")
check("  (control) ssm_a == -exp(A_log) WITHOUT reorder", g, (-np.exp(a_log.astype(np.float64))).astype(np.float32))
# T5 + T4: dt_bias
dtb, _, dt = hf_tensor(L0 + "linear_attn.dt_bias")
g, _ = gguf_tensor("blk.0.ssm_dt.bias")
check("T5+T4 ssm_dt.bias == reorder(dt_bias)", g, reorder_v(dtb[:, None], 0, 1)[:, 0], f"dt_bias {dt}")
# T2: norms +1
for gname, hname in [("blk.0.attn_norm.weight", L0 + "input_layernorm.weight"),
                     ("blk.0.post_attention_norm.weight", L0 + "post_attention_layernorm.weight"),
                     ("blk.3.attn_q_norm.weight", L3 + "self_attn.q_norm.weight"),
                     ("blk.3.attn_k_norm.weight", L3 + "self_attn.k_norm.weight"),
                     ("output_norm.weight", P + "norm.weight")]:
    w, _, dt = hf_tensor(hname); g, _ = gguf_tensor(gname)
    check(f"T2 {gname} == f32(1+w)", g, w + np.float32(1), f"src {dt}")
    check(f"  (control) {gname} == bf16(1+w)", g, f32_to_bf16_rne(w + np.float32(1)))
w, _, dt = hf_tensor(L0 + "linear_attn.norm.weight"); g, _ = gguf_tensor("blk.0.ssm_norm.weight")
check("T2-exception ssm_norm == w (no +1)", g, w, f"src {dt}")
# T3 + T4: conv1d [C,1,4] -> [C,4], V channels reordered
cw, shp, dt = hf_tensor(L0 + "linear_attn.conv1d.weight"); cw = cw.reshape(shp[0], -1)
g, ne = gguf_tensor("blk.0.ssm_conv1d.weight")
qk = DK * NK * 2
check("T3+T4 ssm_conv1d == [qk | reorder(v)]", g, np.concatenate([cw[:qk], reorder_v(cw[qk:], 0, DV)]), f"HF {shp} {dt}, GGUF ne {ne}")
# T4 row reorders on projections
for gname, hname, hd in [("blk.0.ssm_alpha.weight", L0 + "linear_attn.in_proj_a.weight", 1),
                         ("blk.0.ssm_beta.weight", L0 + "linear_attn.in_proj_b.weight", 1),
                         ("blk.0.attn_gate.weight", L0 + "linear_attn.in_proj_z.weight", DV)]:
    w, _, dt = hf_tensor(hname); g, _ = gguf_tensor(gname)
    check(f"T4 {gname} == reorder_rows({hname.split('.')[-2]})", g, reorder_v(w, 0, hd), dt)
check("  (control) ssm_alpha == in_proj_b reordered (swap test)", gguf_tensor("blk.0.ssm_alpha.weight")[0],
      reorder_v(hf_tensor(L0 + "linear_attn.in_proj_b.weight")[0], 0, 1))
w, _, dt = hf_tensor(L0 + "linear_attn.in_proj_qkv.weight"); g, _ = gguf_tensor("blk.0.attn_qkv.weight")
qd = DK * NK * 2
check("T4 attn_qkv == [q k | reorder(v)]", g, np.concatenate([w[:qd], reorder_v(w[qd:], 0, DV)]), dt)
w, _, dt = hf_tensor(L0 + "linear_attn.out_proj.weight"); g, _ = gguf_tensor("blk.0.ssm_out.weight")
check("T4 ssm_out == reorder_cols(out_proj)", g, reorder_v(w, 1, DV), dt)
# full attention: unchanged
for gname, hname in [("blk.3.attn_q.weight", L3 + "self_attn.q_proj.weight"),
                     ("blk.3.attn_k.weight", L3 + "self_attn.k_proj.weight"),
                     ("blk.3.attn_v.weight", L3 + "self_attn.v_proj.weight"),
                     ("blk.3.attn_output.weight", L3 + "self_attn.o_proj.weight")]:
    w, _, dt = hf_tensor(hname); g, _ = gguf_tensor(gname)
    check(f"{gname} == {hname.split('.')[-2]} (unchanged)", g, w, dt)
# MoE: router, shared, experts (T6)
for gname, hname in [("blk.0.ffn_gate_inp.weight", L0 + "mlp.gate.weight"),
                     ("blk.0.ffn_gate_inp_shexp.weight", L0 + "mlp.shared_expert_gate.weight"),
                     ("blk.0.ffn_gate_shexp.weight", L0 + "mlp.shared_expert.gate_proj.weight"),
                     ("blk.0.ffn_down_shexp.weight", L0 + "mlp.shared_expert.down_proj.weight")]:
    w, _, dt = hf_tensor(hname); g, _ = gguf_tensor(gname)
    check(f"{gname} == {hname.split('mlp.')[1]}", g.reshape(w.shape), w, dt)
I, H = 512, 2048
for e in (0, 137, 255):
    gu, shp, dt = hf_tensor(L0 + "mlp.experts.gate_up_proj", e, e + 1)
    gate, _ = gguf_tensor("blk.0.ffn_gate_exps.weight", e * I, (e + 1) * I)
    up, _ = gguf_tensor("blk.0.ffn_up_exps.weight", e * I, (e + 1) * I)
    check(f"T6 expert {e}: gate_exps == gate_up[:I]", gate, gu[0, :I], f"HF {shp} {dt}")
    check(f"T6 expert {e}: up_exps == gate_up[I:]", up, gu[0, I:])
    dn, shp, dt = hf_tensor(L0 + "mlp.experts.down_proj", e, e + 1)
    g, _ = gguf_tensor("blk.0.ffn_down_exps.weight", e * H, (e + 1) * H)
    check(f"T6 expert {e}: down_exps == down_proj", g, dn[0], f"HF {shp}")
# embeddings / head: a few rows
for gname, hname in [("token_embd.weight", P + "embed_tokens.weight"), ("output.weight", "lm_head.weight")]:
    for r in (0, 151643, 248319):
        w, _, dt = hf_tensor(hname, r, r + 1); g, _ = gguf_tensor(gname, r, r + 1)
        check(f"{gname} row {r}", g, w, dt)

w = max(len(r[0]) for r in results)
fails = 0
for label, status, stats, note in results:
    ctrl = "(control)" in label
    bad = (status != "EXACT") != ctrl
    fails += bad
    print(f"{'!!' if bad else 'ok'} {label:<{w}}  {status:5s} {stats}  {note}")
print(f"\n{len(results)} comparisons, {fails} unexpected")
sys.exit(1 if fails else 0)
