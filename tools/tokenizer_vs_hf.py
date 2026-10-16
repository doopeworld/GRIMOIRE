#!/usr/bin/env python3
"""Compare every GRIMOIRE token ID with Hugging Face's tokenizer on a file."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="/tmp/wiki.test.raw")
    parser.add_argument("--tokenizer-dir", default=os.getenv("GRIMOIRE_TOKENIZER_DIR", "/tmp/toktest"))
    parser.add_argument("--hf-python", default="/tmp/gguf-golden-venv/bin/python")
    parser.add_argument("--dump", default="bin/tok_dump")
    parser.add_argument("--expected-count", type=int,
                        help="required token count (defaults to 297193 for wiki.test.raw)")
    args = parser.parse_args()
    model = Path(args.tokenizer_dir) / "tokenizer.json"
    source = Path(args.input)
    if not model.is_file() or not source.is_file():
        parser.error(f"missing input or tokenizer: {source}, {model}")
    hf_code = (
        "from tokenizers import Tokenizer; import json,sys; "
        "t=Tokenizer.from_file(sys.argv[1]); "
        "e=t.encode(open(sys.argv[2], 'rb').read().decode('utf-8'), "
        "add_special_tokens=False); "
        "print(json.dumps({'ids':e.ids,'offsets':e.offsets}))"
    )
    hf = subprocess.run(
        [args.hf_python, "-c", hf_code, str(model), str(source)],
        check=True, capture_output=True, text=True,
    )
    reference = json.loads(hf.stdout)
    ours = subprocess.run(
        [args.dump, args.tokenizer_dir, str(source)],
        check=True, capture_output=True, text=True,
    )
    actual = [int(line) for line in ours.stdout.splitlines()]
    expected = reference["ids"]
    print(f"HF: {len(expected):,} tokens; GRIMOIRE: {len(actual):,} tokens")
    required = args.expected_count
    if required is None and source.name == "wiki.test.raw":
        required = 297193
    index = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                 min(len(expected), len(actual)))
    if index != len(expected) or index != len(actual):
        content = source.read_bytes().decode("utf-8")
        offset = reference["offsets"][index][0] if index < len(expected) else len(content)
        print(f"first divergence at token {index}: HF {expected[index] if index < len(expected) else 'END'}, "
              f"GRIMOIRE {actual[index] if index < len(actual) else 'END'}")
        print(f"text near character {offset}: {content[max(0, offset-40):offset+80]!r}")
        return 1
    if required is not None and len(expected) != required:
        print(f"FAIL: expected {required:,} tokens, got {len(expected):,}")
        return 1
    print(f"PASS: all {len(expected):,} token IDs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
