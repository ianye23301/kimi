"""Benchmark fused GEMV+allreduce (Iris) vs separate GEMV+RCCL allreduce.

Shapes from Kimi K2.5 TP4/TP8 decode:
- Dense projections: [batch, hidden] x [hidden, hidden/TP] -> allreduce
- hidden=7168, TP=4 or TP=8

Usage:
    python bench_iris_fused.py -r 4   # TP4
    python bench_iris_fused.py -r 8   # TP8 (if 8 GPUs available)
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris
from iris.ops.config import FusedConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--num_ranks", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--init_url", type=str, default="tcp://127.0.0.1:29550")
    return parser.parse_args()


def _worker(local_rank, world_size, args):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    shmem = iris.iris(heap_size=1 << 31)
    rank = shmem.get_rank()
    hidden = args.hidden
    shard = hidden // world_size  # per-rank output dim

    dtype = torch.bfloat16

    # Test multiple batch sizes (decode=1, small batch, medium batch)
    for batch in [1, 4, 32, 128]:
        # ── Shapes ──
        # TP pattern: each rank has A=[batch, hidden], B=[hidden, shard]
        # Computes partial C=[batch, shard], then allreduce across ranks
        # But for allreduce pattern: A=[batch, shard_K], B=[shard_K, hidden]
        # Each rank computes partial [batch, hidden], allreduce to sum
        M = batch
        K = shard   # each rank's portion of the reduction dim
        N = hidden  # output dim (same on all ranks)

        # ── Baseline: separate GEMV + RCCL allreduce ──
        A_torch = torch.randn(M, K, dtype=dtype, device=f"cuda:{local_rank}")
        B_torch = torch.randn(K, N, dtype=dtype, device=f"cuda:{local_rank}")
        C_torch = torch.zeros(M, N, dtype=dtype, device=f"cuda:{local_rank}")

        # Warmup
        for _ in range(args.warmup):
            C_torch = A_torch @ B_torch
            dist.all_reduce(C_torch, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            C_torch = A_torch @ B_torch
            dist.all_reduce(C_torch, op=dist.ReduceOp.SUM)
        end.record()
        torch.cuda.synchronize()
        baseline_ms = start.elapsed_time(end) / args.iters
        dist.barrier()

        # ── Iris fused matmul+allreduce ──
        # Need M >= block_size_m, so adjust block sizes for small M
        blk_m = min(64, max(1, M))
        # For M=1, block_size must be >= M
        if M < 64:
            blk_m = M  # triton needs block_size >= 1

        # The iris fused op requires block sizes that work
        # For very small M (1,4), we need small block sizes
        A_iris = shmem.randn((M, K), dtype=dtype)
        B_iris = shmem.randn((K, N), dtype=dtype)
        C_iris = shmem.zeros((M, N), dtype=dtype)

        config = FusedConfig(
            block_size_m=max(M, 1),  # can't be larger than M for small problems
            block_size_n=64,
            block_size_k=64,
            all_reduce_variant="two_shot",
        )

        workspace = None
        try:
            # Warmup
            for _ in range(min(args.warmup, 10)):
                workspace = shmem.ops.matmul_all_reduce(
                    C_iris, A_iris, B_iris, config=config, workspace=workspace
                )
            torch.cuda.synchronize()
            shmem.barrier()

            # Benchmark
            start2 = torch.cuda.Event(enable_timing=True)
            end2 = torch.cuda.Event(enable_timing=True)
            start2.record()
            for _ in range(args.iters):
                workspace = shmem.ops.matmul_all_reduce(
                    C_iris, A_iris, B_iris, config=config, workspace=workspace
                )
            end2.record()
            torch.cuda.synchronize()
            fused_ms = start2.elapsed_time(end2) / args.iters
            shmem.barrier()

            if rank == 0:
                speedup = baseline_ms / fused_ms if fused_ms > 0 else 0
                print(
                    f"batch={M:>4d}  "
                    f"baseline(GEMV+RCCL)={baseline_ms*1000:.1f}us  "
                    f"fused(Iris)={fused_ms*1000:.1f}us  "
                    f"speedup={speedup:.2f}x"
                )
        except Exception as e:
            if rank == 0:
                print(f"batch={M:>4d}  fused FAILED: {e}")
            shmem.barrier()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    if args.num_ranks == 0:
        print("Need at least 1 rank")
        return
    print(f"Benchmarking fused GEMV+allreduce, hidden={args.hidden}, ranks={args.num_ranks}")
    print(f"Shape per rank: A=[batch, {args.hidden // args.num_ranks}] x B=[{args.hidden // args.num_ranks}, {args.hidden}]")
    print()
    mp.spawn(fn=_worker, args=(args.num_ranks, args), nprocs=args.num_ranks, join=True)


if __name__ == "__main__":
    main()
