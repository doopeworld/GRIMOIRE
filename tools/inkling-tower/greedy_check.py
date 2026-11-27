#!/usr/bin/env python3
"""Greedy (temperature 0) answers from the running Inkling server for fixed prompts -> JSON (ids not exposed: text)."""
import json, sys, urllib.request
out = sys.argv[1]
filler = " ".join(["In distributed systems the tension between consistency, availability, and partition tolerance shapes almost every design decision."] * 60)
P = [
    [{"role": "user", "content": "Write a Python function that checks whether a number is prime. Keep it short."}],
    [{"role": "user", "content": "Read the following context, then answer.\n\n" + filler + "\n\nIn one sentence: what tension shapes design decisions in distributed systems?"}],
    [{"role": "user", "content": "Explain the difference between a mutex and a semaphore in three sentences."}],
]
res = []
for msgs in P:
    body = json.dumps({"model": "inkling-2-mini", "messages": msgs, "max_tokens": 160, "temperature": 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % (int(sys.argv[2]) if len(sys.argv) > 2 else 6911) + "", body, {"Content-Type": "application/json"}), timeout=600)
    d = json.load(r)
    m = d["choices"][0]["message"]
    res.append({"usage": d["usage"], "reasoning": m.get("reasoning_content", ""), "content": m.get("content", "")})
json.dump(res, open(out, "w"), indent=1)
for x in res:
    print(x["usage"], repr((x["reasoning"] + "|" + x["content"])[:160]))
