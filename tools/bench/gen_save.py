#!/usr/bin/env python3
# gen_save.py PORT OUT.json -- three fixed greedy requests through grimoire-server; saves the full
# text (reasoning + content) and timing so two engine configurations can be compared exactly.
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
url = "http://127.0.0.1:%s/v1/chat/completions" % port
PROMPTS = [
    ("capital", "What is the capital of France? Answer in one word.", 40),
    ("story", "Write a short story about a lighthouse keeper.", 160),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", 160),
]
res = {}
for tag, text, n in PROMPTS:
    body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": n,
            "temperature": 0.0}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    m = r["choices"][0]["message"]
    full = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    toks = r.get("usage", {}).get("completion_tokens", 0)
    res[tag] = {"text": full, "tokens": toks, "seconds": dt}
    print("%-8s %3d tokens %6.2fs  %s" % (tag, toks, dt, repr(full.replace("\x00", " | ")[-90:])))
json.dump(res, open(out, "w"), indent=1)
