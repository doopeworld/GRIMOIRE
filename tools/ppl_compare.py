#!/usr/bin/env python3
"""Compare GRIMOIRE --ppl-dump files made from the same tokenized text."""
import argparse
import math
import statistics


def read_dump(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 14:
                raise ValueError(f"{path}:{line_number}: expected 14 TSV fields")
            chunk, pos, target = map(int, fields[:3])
            logprob = float(fields[3])
            argmax = int(fields[4])
            top = {}
            for item in fields[6:]:
                token, lp = item.split(":", 1)
                top[int(token)] = math.exp(float(lp))
            if len(top) != 8 or argmax not in top:
                raise ValueError(f"{path}:{line_number}: invalid top-8")
            rows.append(((chunk, pos, target), logprob, argmax, top))
    return rows


def approximate_kl(ref, test, vocab_size):
    """Uniformly allocate each unlisted tail over its unlisted vocab ids."""
    ref_tail = max(0.0, 1.0 - sum(ref.values()))
    test_tail = max(0.0, 1.0 - sum(test.values()))
    ref_each = ref_tail / (vocab_size - len(ref))
    test_each = test_tail / (vocab_size - len(test))
    union = ref.keys() | test.keys()
    kl = 0.0
    for token in union:
        p = ref.get(token, ref_each)
        q = test.get(token, test_each)
        if p > 0:
            if q <= 0:
                return math.inf
            kl += p * math.log(p / q)
    p_rest = 1.0 - sum(ref.get(token, ref_each) for token in union)
    q_rest = 1.0 - sum(test.get(token, test_each) for token in union)
    if p_rest > 0:
        if q_rest <= 0:
            return math.inf
        kl += p_rest * math.log(p_rest / q_rest)
    return kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref")
    parser.add_argument("test")
    parser.add_argument("--vocab-size", type=int, default=248320,
                        help="model vocabulary size (default: Qwen-family 248320)")
    args = parser.parse_args()
    ref, test = read_dump(args.ref), read_dump(args.test)
    if not ref or len(ref) != len(test):
        parser.error("dumps must contain the same nonzero number of scored tokens")
    if args.vocab_size <= 8:
        parser.error("--vocab-size must exceed 8")
    for index, (a, b) in enumerate(zip(ref, test), 1):
        if a[0] != b[0]:
            parser.error(f"token {index} differs: {a[0]} vs {b[0]}")
        if max(*a[3], *b[3]) >= args.vocab_size:
            parser.error("--vocab-size is smaller than an id in the dumps")
    differences = sorted(abs(a[1] - b[1]) for a, b in zip(ref, test))
    p99_index = 0.99 * (len(differences) - 1)
    low = int(p99_index)
    p99 = differences[low] + (differences[min(low + 1, len(differences) - 1)] - differences[low]) * (p99_index - low)
    agreement = sum(a[2] == b[2] for a, b in zip(ref, test)) / len(ref)
    kl = statistics.fmean(approximate_kl(a[3], b[3], args.vocab_size)
                          for a, b in zip(ref, test))
    print(f"tokens: {len(ref)}")
    print(f"top-1 agreement: {100 * agreement:.2f}%")
    print(f"mean |Δ logprob_target|: {statistics.fmean(differences):.8g}")
    print(f"p99 |Δ logprob_target|: {p99:.8g}")
    print(f"approximate KL(ref||test), nats: {kl:.8g} "
          f"(union of top-8; unlisted mass uniform over {args.vocab_size}-word vocabulary, remainder lumped)")


if __name__ == "__main__":
    main()
