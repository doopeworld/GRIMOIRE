#!/usr/bin/env python3
# codecmp.py A.code.json B.code.json -- compare coding_probe answers (JSON lines) trial by trial
import json, sys
def load(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return [r for r in out if "text" in r]
a, b = load(sys.argv[1]), load(sys.argv[2])
for i, (x, y) in enumerate(zip(a, b)):
    tx, ty = x["text"], y["text"]
    n = 0
    while n < min(len(tx), len(ty)) and tx[n] == ty[n]:
        n += 1
    print("trial %d: prompt %s/%s ttft %.3f/%.3f  %s" % (i, x.get("prompt_tokens"), y.get("prompt_tokens"),
          x.get("ttft_s", 0), y.get("ttft_s", 0),
          "IDENTICAL (%d chars)" % len(tx) if tx == ty else "differ at char %d of %d/%d" % (n, len(tx), len(ty))))
