# grimoire-ref:cpu -- the reference side of GRIMOIRE's accuracy check (2026-10-29): Hugging Face
# transformers on the CPU, to produce the original model's logits.  A measuring tool only; nothing
# here is part of GRIMOIRE or its image.
FROM python:3.12
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy safetensors tokenizers sentencepiece "transformers>=5.8" \
    || pip install --no-cache-dir "git+https://github.com/huggingface/transformers"
RUN python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
