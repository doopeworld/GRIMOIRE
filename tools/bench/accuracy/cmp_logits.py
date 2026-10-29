#!/usr/bin/env python3
# cmp_logits.py REF.bin TEST.bin VOCAB [LABEL] -- per position: cosine of the logits, KL(ref || test)
# of the softmax distributions, top-1 agreement; mean / median / worst over the positions.
import sys
import numpy as np

ref = np.fromfile(sys.argv[1], dtype=np.float32)
tst = np.fromfile(sys.argv[2], dtype=np.float32)
V = int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else sys.argv[2]
ref = ref.reshape(-1, V)
tst = tst.reshape(-1, V)
n = min(len(ref), len(tst))
ref, tst = ref[:n].astype(np.float64), tst[:n].astype(np.float64)

def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))

lr, lt = log_softmax(ref), log_softmax(tst)
kl = (np.exp(lr) * (lr - lt)).sum(axis=1)
cos = (ref * tst).sum(axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(tst, axis=1))
top1 = (ref.argmax(axis=1) == tst.argmax(axis=1)).mean()
print("%-34s positions %4d | cosine mean %.4f median %.4f min %.4f | KL mean %.4f median %.4f max %.3f | top-1 %.1f%%"
      % (label, n, cos.mean(), np.median(cos), cos.min(), kl.mean(), np.median(kl), kl.max(), 100.0 * top1))
