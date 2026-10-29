#!/usr/bin/env python3
# ref_logits.py MODEL_DIR TEXT_FILE N IDS_OUT LOGITS_OUT
# Reference logits for GRIMOIRE's accuracy check (2026-10-29): the original checkpoint in Hugging
# Face transformers on the CPU (bf16 weights), teacher-forced over the first N tokens of TEXT_FILE.
# A measuring tool in its own container -- nothing of it is part of GRIMOIRE.
# Writes the token ids (one per line: the input of `grimoire --score-ids`) and the logits
# (fp32 [N][vocab], raw little-endian -- the layout of `grimoire --score-out`).
import os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir, text_file, n, ids_out, logits_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok(open(text_file, encoding="utf-8", errors="replace").read(), add_special_tokens=False)["input_ids"][:n]
with open(ids_out, "w") as f:
    f.write("\n".join(str(i) for i in ids) + "\n")
t0 = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
model.eval()
print("loaded %s in %.0f s" % (type(model).__name__, time.time() - t0), flush=True)
t0 = time.time()
with torch.no_grad():
    out = model(input_ids=torch.tensor([ids]), use_cache=False)
lg = out.logits[0].float().contiguous().numpy()
lg.tofile(logits_out)
print("forward of %d tokens in %.0f s, logits %s, transformers %s, torch %s" % (
    len(ids), time.time() - t0, tuple(lg.shape), __import__("transformers").__version__, torch.__version__), flush=True)
