"""Local test + benchmark for MLA competition.

Usage (inside container):
    python /workspace/mla/mla_local_bench.py --gpu 4 --ref
"""
import sys
sys.path.insert(0, "/workspace/mla")

import time
import argparse
import torch

# Parse early so we can set device before imports trigger CUDA init
_parser = argparse.ArgumentParser()
_parser.add_argument("--gpu", type=int, default=4, help="GPU device index")
_parser.add_argument("--bench", action="store_true", help="Run benchmarks")
_parser.add_argument("--ref", action="store_true", help="Also benchmark reference")
_parser.add_argument("--test-only", action="store_true", help="Only run tests")
_args = _parser.parse_args()

torch.cuda.set_device(_args.gpu)

from reference import generate_input, ref_kernel, check_implementation
from submission import custom_kernel


TEST_CASES = [
    # (bs, qseqlen, kvseqlen, tp, seed)
    (4, 1, 1024, 8, 4220),
    (4, 4, 1024, 8, 4231),
    (32, 1, 1024, 4, 5412),
    (32, 1, 8192, 8, 5415),
    (128, 1, 8192, 8, 7816),
    (128, 4, 8192, 4, 7827),
]

BENCH_CASES = [
    (4, 1, 1024, 4, 4237),
    (4, 4, 8192, 4, 4251),
    (32, 1, 8192, 8, 5415),
    (32, 4, 1024, 8, 5420),
    (32, 1, 1024, 4, 5432),
    (32, 4, 8192, 4, 5443),
    (128, 1, 8192, 8, 7816),
    (128, 4, 8192, 8, 7824),
]


def clone_data(data):
    if isinstance(data, tuple):
        return tuple(clone_data(x) for x in data)
    elif isinstance(data, list):
        return [clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    return data


def run_tests():
    print("=== Correctness Tests ===")
    all_pass = True
    for bs, qsl, kvsl, tp, seed in TEST_CASES:
        data = generate_input(bs, qsl, kvsl, tp, seed)
        torch.cuda.synchronize()
        out = custom_kernel(clone_data(data))
        torch.cuda.synchronize()
        ok, msg = check_implementation(data, out)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  bs={bs:>3d} qsl={qsl} kvsl={kvsl:>5d} tp={tp} => {status} {msg[:80] if msg else ''}")
    return all_pass


def run_benchmarks(ref_too=False):
    print("\n=== Benchmarks ===")
    warmup = 10
    iters = 50

    for bs, qsl, kvsl, tp, seed in BENCH_CASES:
        data = generate_input(bs, qsl, kvsl, tp, seed)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(warmup):
            _ = custom_kernel(clone_data(data))
        torch.cuda.synchronize()

        # Benchmark submission
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = custom_kernel(data)
        end.record()
        torch.cuda.synchronize()
        sub_us = start.elapsed_time(end) * 1000 / iters

        ref_us = 0
        if ref_too:
            # Warmup ref
            for _ in range(warmup):
                _ = ref_kernel(clone_data(data))
            torch.cuda.synchronize()

            start.record()
            for _ in range(iters):
                _ = ref_kernel(data)
            end.record()
            torch.cuda.synchronize()
            ref_us = start.elapsed_time(end) * 1000 / iters

        line = f"  bs={bs:>3d} qsl={qsl} kvsl={kvsl:>5d} tp={tp} => sub={sub_us:.1f}us"
        if ref_too:
            speedup = ref_us / sub_us if sub_us > 0 else 0
            line += f"  ref={ref_us:.1f}us  speedup={speedup:.2f}x"
        print(line)


if __name__ == "__main__":
    args = _args
    ok = run_tests()
    if not ok:
        print("\nSome tests FAILED!")
        sys.exit(1)

    if not args.test_only:
        run_benchmarks(ref_too=args.ref)
