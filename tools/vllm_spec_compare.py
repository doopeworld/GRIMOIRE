"""Controlled single-stream speculative decode benchmark for Tower's B70.

One engine is loaded per invocation so DSpark and DFlash can be compared with
identical target/cache/scheduler settings.  METHOD=baseline omits speculation.
"""
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams


TARGET = os.environ.get(
    "MODEL", "/models/Qwen3.8-27B-GPTQ-Int4-MTP-BF16"
)
METHOD = os.environ.get("METHOD", "dspark").lower()
DRAFT = os.environ.get("DRAFT", "/models/Qwen3.8-27B-DSpark")
NTOK = int(os.environ.get("NTOK", "256"))
SPEC_K = int(os.environ.get("SPEC_K", "7"))
REPEATS = int(os.environ.get("REPEATS", "2"))
DRAFT_SAMPLE_METHOD = os.environ.get("DRAFT_SAMPLE_METHOD", "")
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "0"))
PREFIX_FILE = os.environ.get("PREFIX_FILE", "/grimoire/src/grimoire.cpp")
PREFIX_CACHE = os.environ.get("PREFIX_CACHE", "0") == "1"

PROMPTS = {
    "prose": "Write a clear explanation of how a four-stroke engine works.",
    "code": (
        "Write a Python merge_sort function. Explain its time complexity, "
        "then include three assert-based tests."
    ),
    "agentic": (
        "You are editing a C++ inference engine. Give a concise patch plan to "
        "fuse quantization into a decoder FFN while preserving correctness."
    ),
}


def main():
    engine = dict(
        model=TARGET,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        block_size=64,
        trust_remote_code=True,
        enable_prefix_caching=PREFIX_CACHE,
        kv_cache_dtype="fp8",
        async_scheduling=True,
    )
    if METHOD != "baseline":
        spec = {
            "method": METHOD,
            "num_speculative_tokens": SPEC_K,
        }
        if METHOD != "mtp":
            spec["model"] = DRAFT
        if DRAFT_SAMPLE_METHOD:
            spec["draft_sample_method"] = DRAFT_SAMPLE_METHOD
        engine["speculative_config"] = spec

    print(
        f"TARGET={TARGET} METHOD={METHOD} DRAFT={DRAFT} "
        f"SPEC_K={SPEC_K} NTOK={NTOK} REPEATS={REPEATS} "
        f"DRAFT_SAMPLE_METHOD={DRAFT_SAMPLE_METHOD or 'default'} "
        f"PREFIX_TOKENS={PREFIX_TOKENS} PREFIX_CACHE={PREFIX_CACHE}",
        flush=True,
    )
    print(
        "ENGINE=block_size:64 dtype:bfloat16 max_num_batched_tokens:4096 "
        "trust_remote_code:true async_scheduling:true kv_cache_dtype:fp8",
        flush=True,
    )
    llm = LLM(**engine)
    prompts = PROMPTS
    if PREFIX_TOKENS:
        tokenizer = llm.get_tokenizer()
        source = Path(PREFIX_FILE).read_text(errors="replace")
        prefix_ids = tokenizer.encode(source, add_special_tokens=False)
        if len(prefix_ids) < PREFIX_TOKENS:
            raise ValueError(
                f"{PREFIX_FILE} has only {len(prefix_ids)} tokens; "
                f"requested {PREFIX_TOKENS}"
            )
        prefix = tokenizer.decode(prefix_ids[:PREFIX_TOKENS])
        prompts = {
            name: f"Background source file:\n{prefix}\n\nTask:\n{task}"
            for name, task in PROMPTS.items()
        }
    sampling = SamplingParams(
        max_tokens=NTOK, temperature=0, ignore_eos=True, seed=0
    )

    # Compile and warm the exact decode path without counting it.
    warm_prompts = prompts.values() if PREFIX_CACHE else [prompts["prose"]]
    for prompt in warm_prompts:
        llm.generate([prompt], sampling)

    rates = []
    for workload, prompt in prompts.items():
        best = 0.0
        last = None
        for _ in range(REPEATS):
            start = time.perf_counter()
            out = llm.generate([prompt], sampling)
            elapsed = time.perf_counter() - start
            count = len(out[0].outputs[0].token_ids)
            best = max(best, count / elapsed)
            last = out[0].outputs[0].text
        rates.append(best)
        sample = (last or "").replace("\n", " ")[:160]
        print(f"RESULT workload={workload} tok_s={best:.2f} sample={sample!r}")
    print(
        f"SUMMARY method={METHOD} min={min(rates):.2f} "
        f"mean={sum(rates) / len(rates):.2f} max={max(rates):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
