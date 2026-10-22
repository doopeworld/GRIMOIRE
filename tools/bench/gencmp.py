#!/usr/bin/env python3
# gencmp.py A.gen.json B.gen.json -- where two greedy generations (gen_save.py) first differ
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for tag in a:
    x, y = a[tag]["text"], b.get(tag, {}).get("text", "")
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    same = "IDENTICAL" if x == y else "differ at char %d of %d/%d" % (n, len(x), len(y))
    print("%-8s %s" % (tag, same))
    if x != y:
        print("   A: %r" % x[max(0, n - 40):n + 50])
        print("   B: %r" % y[max(0, n - 40):n + 50])
