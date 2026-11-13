#!/usr/bin/env python3
"""Summarize unitrace -d (device timing) output from a Grimoire server log (2026-11-13).
Usage: ut_summary1113.py SERVER.log [STEPS]
Prints the kernels sorted by total device time; with STEPS (number of decode/verify steps in the run) also the
per-step time and calls per step, so a verify layer's small kernels can be counted against its matrix kernels."""
import csv, io, re, sys

log = open(sys.argv[1], errors='replace').read().splitlines()
steps = float(sys.argv[2]) if len(sys.argv) > 2 else None
rows, hdr = [], None
for line in log:
    line = re.sub(r'^\[rank\d\] ', '', line)
    if re.match(r'\s*Kernel,\s*Calls,', line):
        hdr = [h.strip() for h in line.split(',')]
        continue
    if hdr and line.strip().startswith('"'):
        r = next(csv.reader(io.StringIO(line.strip()), skipinitialspace=True))
        if len(r) >= 4:
            try:
                rows.append((r[0], int(r[1]), int(r[2]), float(r[3])))
            except ValueError:
                pass
    elif hdr and not line.strip():
        hdr = None if rows else hdr
rows.sort(key=lambda x: -x[2])
tot = sum(r[2] for r in rows)
print(f'{len(rows)} kernels, total device time {tot / 1e6:.1f} ms')
for name, calls, ns, pct in rows[:60]:
    short = re.sub(r'\s+', ' ', name)[:110]
    extra = f'  {calls / steps:7.1f}/step {ns / steps / 1e3:9.1f} us/step' if steps else ''
    print(f'{ns / 1e6:10.1f} ms {pct:6.2f}% {calls:9d} calls {ns / max(calls, 1) / 1e3:8.1f} us/call{extra}  {short}')
