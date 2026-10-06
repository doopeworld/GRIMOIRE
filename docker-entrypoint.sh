#!/bin/bash
# docker-entrypoint.sh -- dispatches to bin/grimoire-server or bin/grimoire,
# and ALWAYS wraps the real work in `timeout --signal=TERM --kill-after=60`
# so a `docker stop` gets a graceful TERM with time to drain GPU work
# before anything resorts to SIGKILL. A B70 that gets SIGKILLed mid-
# submission can fall off the PCI bus and need a power cycle -- this is
# not a hypothetical, it is this project's most repeated failure mode.
#
# That protects against a SLOW stop. It does NOT replace --init: without
# it, THIS script is PID 1, gets the kernel's default (do-nothing) signal
# disposition for most signals, and never reaps the binary's children --
# so `docker stop` still has no clean way to ask the workload to finish.
# Pass --init on `docker run` every time. The warning below fires if you
# forget.
set -u

if [ "$$" -eq 1 ]; then
    echo "WARNING: no --init on this container (docker-entrypoint.sh is PID 1)." >&2
    echo "         docker stop can only SIGKILL from here -- GPU work in flight" >&2
    echo "         may wedge the card. Re-run with 'docker run --init ...'." >&2
fi

TIMEOUT_SECS="${GRIMOIRE_TIMEOUT_SECS:-0}"   # 0 = no limit (a server should not have one)
run() {
    if [ "$TIMEOUT_SECS" -gt 0 ]; then
        exec timeout --signal=TERM --kill-after=60 "$TIMEOUT_SECS" "$@"
    else
        exec "$@"
    fi
}

case "${1:-server}" in
    server)
        shift
        run /grimoire/bin/grimoire-server "$@"
        ;;
    generate|cli)
        shift
        run /grimoire/bin/grimoire "$@"
        ;;
    multi)
        # Several GPUs in one container: one grimoire-server rank per card.
        #   multi [PP|TP] <grimoire-server args...>
        # PP (default) gives each rank a block of layers (GRIMOIRE_PP_SPLIT =
        # layers on rank 0 with two ranks, or GRIMOIRE_PP_LAYERS=a,b,...); TP
        # gives every rank a slice of every weight.  GRIMOIRE_MULTI_GPUS ranks
        # (default 2): pass exactly that many render nodes with --device and
        # ZE_AFFINITY_MASK=0,1[,2,...].  Only rank 0 binds the HTTP port; the
        # others follow it.  The same launcher as tools/serve_pp2_worker.sh,
        # inside the image so an Unraid template can run it.
        shift
        mode=PP
        case "${1:-}" in PP|TP) mode=$1; shift ;; esac
        n="${GRIMOIRE_MULTI_GPUS:-2}"
        if [ "$mode" = TP ]; then
            export GRIMOIRE_TP_WORLD_SIZE="$n"
            export GRIMOIRE_TP_SOCKET="${GRIMOIRE_TP_SOCKET:-/tmp/grimoire-tp-serve.sock}"
            sock="$GRIMOIRE_TP_SOCKET"
        else
            export GRIMOIRE_PP_WORLD_SIZE="$n"
            export GRIMOIRE_PP_SOCKET="${GRIMOIRE_PP_SOCKET:-/tmp/grimoire-pp-serve.sock}"
            sock="$GRIMOIRE_PP_SOCKET"
        fi
        rm -f "${sock}"* 2>/dev/null || true
        # One card per process: rank r sees ONLY the r-th device of
        # GRIMOIRE_MULTI_DEVICES (default: the container's ZE_AFFINITY_MASK,
        # e.g. 0,1).  The B70 inference cookbook found that two-card workers
        # which can each see both cards are not equivalent to one mask per
        # worker (host memory ~11 GB vs < 1 GB per process, and a dual-B70
        # host that crashed without the per-worker masks).  GRIMOIRE_DEVICES
        # then maps every rank to index 0, the one card its process sees.
        IFS=',' read -r -a devs <<< "${GRIMOIRE_MULTI_DEVICES:-${ZE_AFFINITY_MASK:-}}"
        if [ "${#devs[@]}" -lt "$n" ]; then
            echo "multi: $n ranks need $n devices in GRIMOIRE_MULTI_DEVICES or ZE_AFFINITY_MASK" \
                 "(got '${GRIMOIRE_MULTI_DEVICES:-${ZE_AFFINITY_MASK:-}}')" >&2
            exit 2
        fi
        zeros=$(printf '0,%.0s' $(seq 1 "$n")); zeros=${zeros%,}
        pids=()
        # Later ranks listen and earlier ones connect, so start from the back.
        # Process substitution keeps $! the server's PID (not the log
        # prefixer's) -- see tools/serve_pp2_worker.sh.
        for ((r = n - 1; r >= 0; --r)); do
            env "GRIMOIRE_${mode}_RANK=$r" "ZE_AFFINITY_MASK=${devs[r]}" "GRIMOIRE_DEVICES=$zeros" \
                /grimoire/bin/grimoire-server "$@" > >(sed -u "s/^/[rank$r] /") 2>&1 &
            pids[r]=$!
        done
        # docker stop reaches this script (tini forwards to it), not the
        # servers: hand the TERM on and keep waiting so GPU work can drain.
        trap 'kill -TERM "${pids[@]}" 2>/dev/null' TERM INT
        reap() { local st=0; wait "$1"; st=$?
                 while kill -0 "$1" 2>/dev/null; do wait "$1"; st=$?; done; return "$st"; }
        reap "${pids[0]}"; st=$?
        kill -TERM "${pids[@]}" 2>/dev/null || true
        for ((r = 1; r < n; ++r)); do reap "${pids[r]}" || true; done
        echo "front end (rank 0) exited $st"
        exit "$st"
        ;;
    /grimoire/bin/*|bin/*)
        run "$@"
        ;;
    *)
        echo "usage: docker run ... <image> {server|multi|generate} [args...]" >&2
        echo "  server   -> bin/grimoire-server (OpenAI-compatible HTTP), e.g.:" >&2
        echo "              server --model /models/K2-Horizon-MoVA-36B-A4B --proj mxfp4 --ctx 8192 --port 8000" >&2
        echo "  multi    -> one server rank per GPU (pipeline parallel; 'multi TP' = tensor parallel), e.g.:" >&2
        echo "              multi --model /models/Ornith-1.5-35B-A3B-FP8 --proj fp8 --ctx 32000 --port 8000" >&2
        echo "  generate -> bin/grimoire (one-shot CLI), e.g.:" >&2
        echo "              generate -m /models/K2-Horizon-MoVA-36B-A4B --proj mxfp4 --ctx 8192 -p \"...\" -n 256" >&2
        exit 2
        ;;
esac
