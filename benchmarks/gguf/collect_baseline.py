#!/usr/bin/env python3
"""Aggregate Phase 1 baseline runs into benchmarks/gguf/baseline.csv.

Reads only files written by run_llamacpp.sh / run_served.sh under RUNROOT; every CSV row names
the file it came from. No number is typed in by hand (CLAUDE.md rule 8).

usage: collect_baseline.py RUNROOT OUT.csv
"""
import csv, glob, json, os, re, sys

ROOT, OUT = sys.argv[1], sys.argv[2]
Q4KM = ("Ornith-1.5-35B-Q4_K_M.gguf",
        "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f")
COLS = ["date", "engine", "engine_version", "image", "model", "model_sha256", "weights_format",
        "cross_format", "gpu", "pcie", "harness", "workload", "concurrency", "runs", "metric",
        "median_or_mean", "stddev", "unit", "coherent", "source"]
rows = []

def kv(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if ":" in line:
                k, v = line.split(":", 1); d.setdefault(k.strip(), v.strip())
    return d

# --- engine-internal: llama-bench JSON ------------------------------------------------------
for backend in ("sycl", "vulkan"):
    env = kv(f"{ROOT}/{backend}/env.txt")
    coh = open(f"{ROOT}/{backend}/coherence.txt", errors="replace").read() if os.path.exists(
        f"{ROOT}/{backend}/coherence.txt") else ""
    coherent = "yes(read)" if "Rayleigh" in coh or "scatter" in coh else "UNREAD"
    for ub in (512, 2048):
        raw = f"{ROOT}/{backend}/bench_ub{ub}.raw"
        if not os.path.exists(raw):
            continue
        lines = open(raw, errors="replace").read().splitlines()
        s = [i for i, l in enumerate(lines) if l.strip() == "["]
        e = [i for i, l in enumerate(lines) if l.strip() == "]"]
        if not s or not e:
            continue
        for r in json.loads("\n".join(lines[s[0]:e[-1] + 1])):
            test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
            rows.append(dict(
                date=r.get("test_time", env.get("date", "")), engine=f"llama.cpp-{backend}",
                engine_version=f"build {r.get('build_number')} {r.get('build_commit')}",
                image=env.get("image", ""), model=Q4KM[0], model_sha256=Q4KM[1],
                weights_format="GGUF Q4_K_M (Q4_K 379 / Q6_K 64 / F32 310)", cross_format="no",
                gpu=r.get("gpu_info", ""), pcie=env.get("link 0000:00:01.0", ""),
                harness=f"llama-bench -ngl 99 -fa 1 -b 2048 -ub {ub} -r {r.get('n_reps', 5)}",
                workload=test, concurrency=1, runs=r.get("n_reps", 5),
                metric="prefill tok/s" if r["n_prompt"] else "decode tok/s",
                median_or_mean=f"{r['avg_ts']:.2f}", stddev=f"{r['stddev_ts']:.2f}", unit="tok/s (mean)",
                coherent=coherent, source=os.path.relpath(raw, ROOT)))

# --- served: llama-benchy markdown -----------------------------------------------------------
META = {
    "llamacpp-sycl": ("llama.cpp-sycl (llama-server)", Q4KM, "GGUF Q4_K_M", "no"),
    "llamacpp-vulkan": ("llama.cpp-vulkan (llama-server)", Q4KM, "GGUF Q4_K_M", "no"),
    "grimoire-v1.9.0-Ornith-1.5-35B-A3B-mxfp4": (
        "GRIMOIRE v1.9.0", ("Ornith-1.5-35B-A3B (HF 10fbf86f, BF16 safetensors)", ""),
        "BF16 safetensors -> --proj mxfp4", "yes"),
    "grimoire-v1.9.0-Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE-mxfp4": (
        "GRIMOIRE v1.9.0", ("Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE", ""),
        "pre-converted MXFP4 safetensors", "yes"),
    "vllm-gptq-int4": ("vLLM 0.31.0+xpu", ("Ornith-1.5-35B-A3B-GPTQ-Int4", ""),
                       "GPTQ-Int4 safetensors, fp16 act, fp8 KV", "yes"),
}
for d in sorted(glob.glob(f"{ROOT}/served/*/")):
    name = os.path.basename(d.rstrip("/"))
    md, log = f"{d}benchy.md", f"{d}benchy.log"
    if not os.path.exists(md) or name not in META:
        continue
    eng, (model, sha), fmt, cross = META[name]
    env = kv(f"{d}engine.txt")
    coherent = "PASSED" if "Coherence test PASSED" in open(log, errors="replace").read() else "FAILED/absent"
    parsed = []
    for line in open(md):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and re.match(r"^(pp|tg)\d+$", cells[1]):
            parsed.append(cells)
    # llama-benchy emits (ppN, tgM) pairs for every pp x tg combination: label each row by its pair
    for i, cells in enumerate(parsed):
        m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", cells[2])
        if not m:
            continue
        mate = parsed[i + 1][1] if cells[1].startswith("pp") and i + 1 < len(parsed) else (
            parsed[i - 1][1] if i > 0 else "")
        test = f"{cells[1]} (pair {mate})"
        rows.append(dict(
            date=env.get("date", ""), engine=eng, engine_version=eng, image=env.get("image", ""),
            model=model, model_sha256=sha, weights_format=fmt, cross_format=cross,
            gpu="Arc Pro B70 gpu0 0000:03:00.0", pcie=env.get("link 00:01.0", ""),
            harness="llama-benchy 0.4.0 --no-cache --exact-tg (OpenAI API)", workload=test,
            concurrency=1, runs=5,
            metric="prefill tok/s (served)" if cells[1].startswith("pp") else "decode tok/s (served)",
            median_or_mean=m.group(1), stddev=m.group(2), unit="tok/s (llama-benchy mean)",
            coherent=coherent, source=os.path.relpath(md, ROOT)))
        if cells[1].startswith("pp") and len(cells) > 6:
            t = re.match(r"([\d.]+)\s*±\s*([\d.]+)", cells[6])
            if t:
                rows.append(dict(rows[-1], metric="e2e TTFT (served)", median_or_mean=t.group(1),
                                 stddev=t.group(2), unit="ms"))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {OUT}")
