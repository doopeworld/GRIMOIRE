# Qwen3.8-Flash-Next on four Arc Pro B70s

This path uses the native SYCL server with one tensor-parallel rank per
GPU. Each rank keeps 128 of the 512 GPTQ routed experts resident for
each layer. The checkpoint must be the Flash-Next W4A16 safetensors
layout with BF16 shared experts. Use BF16 projection loading and turn
off speculative decoding. The TP4 path validates the checkpoint layout
at startup.

Build with the oneAPI environment loaded and `make native SYCL_TARGET=spir64`.
This builds the C++/SYCL server and CLI. Level Zero compiles the device code
on first use.
Launch all four ranks in the same environment. The following example
assumes exactly four visible Level Zero GPUs and a checkpoint containing
its PLE tensors. If the PLE tensors are in a separate indexed
safetensors directory, export `GRIMOIRE_QFN_PLE_SOURCE` with its path.

```bash
set -euo pipefail
model=/path/to/qwen38-w4a16
server=bin/native-spir64/grimoire-server
run_dir=$(mktemp -d)
pids=()
cleanup() {
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
    rm -f -- "$run_dir/tp"
    rmdir -- "$run_dir"
}
trap cleanup EXIT
trap "exit 130" INT
trap "exit 143" TERM
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_TP_SOCKET="$run_dir/tp"
export GRIMOIRE_QFN_TP4=1 GRIMOIRE_P2P=1 GRIMOIRE_P2P_CHUNK_MB=1
export GRIMOIRE_QFN_TP_SYNC=fanout GRIMOIRE_MTP=0 GRIMOIRE_SEQ_SLOTS=1
export GRIMOIRE_DECODE_GRAPH=0 GRIMOIRE_PREFIX_CACHE=1
export GRIMOIRE_PREFILL_CHUNK=2048 GRIMOIRE_QFN_PREFILL_RSAG=1
for rank in 0 1 2 3; do
    GRIMOIRE_TP_RANK=$rank "$server" -m "$model" --proj bf16 --ctx 32768 \
        --host 127.0.0.1 --port 8080 > "rank-$rank.log" 2>&1 &
    pids+=("$!")
done
wait -n "${pids[@]}"
```

Only rank 0 listens for HTTP. Check `/health` once the model has loaded.
Each rank should log `TP4 P2P: direct peer-push, FP32 payloads and rank-order sum`.
Required P2P fails startup if peer imports are unavailable. All four ranks
must use the same P2P and synchronization settings. The optional
reduce-scatter/all-gather path applies to full 2,048-row reductions at
the Flash-Next hidden width. Other payload sizes use peer push.

The dense single-slot TP4 server can reuse an exact prompt or a prompt
extended with more tokens. It saves a prompt-head snapshot before decode
and keeps that prompt’s KV rows resident. All ranks agree on the match
before skipping any rows. The default prompt-head threshold is 1,024
new tokens. Set `GRIMOIRE_PREFIX_CACHE=0` to measure a cold request.

To test the collective without loading a checkpoint, reserve the four
GPUs and start the same four ranks with the native `grimoire` CLI.
Clear `GRIMOIRE_QFN_TP_SYNC` and `GRIMOIRE_QFN_PREFILL_RSAG` for this
test, export `GRIMOIRE_QFN_TP4_PEER_REGRESSION=1`, and use
`-m unused -p probe -n 1` as the CLI arguments. Each rank checks a
full prefill reduction and four smaller reductions against the
host FP32 rank-order reference.
