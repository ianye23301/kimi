"""
Head-to-head benchmark: aiter fp8 ASM vs our MXFP4 Triton MLA decode.
"""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import torch
torch.cuda.set_device(4)

from aiter import dtypes as aiter_dtypes
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
NHEADS = 16
NUM_KV_HEADS = 1
QK_DIM = 576
V_DIM = 512
SM_SCALE = 1.0 / (QK_DIM ** 0.5)
PAGE_SIZE = 1
NUM_KV_SPLITS = 16
device = "cuda"


def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)


def build_aiter_meta(batch_size, qo, kv_ind, q_dtype, kv_dtype):
    kv_last = (kv_ind[1:] - kv_ind[:-1]).to(torch.int32)
    info = get_mla_metadata_info_v1(
        batch_size, 1, NHEADS, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=NUM_KV_SPLITS, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device=device) for s, t in info]
    get_mla_metadata_v1(
        qo, kv_ind, kv_last,
        NHEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        *work, page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
        max_split_per_batch=NUM_KV_SPLITS, intra_batch_mode=True,
        dtype_q=q_dtype, dtype_kv=kv_dtype,
    )
    return dict(zip(["work_meta_data", "work_indptr", "work_info_set",
                     "reduce_indptr", "reduce_final_map", "reduce_partial_map"], work))


def run_aiter_fp8(q_fp8, q_scale, kv_fp8, kv_scale, kv_4d, kv_indices,
                   kv_last, qo, kv_ind, meta, o):
    mla_decode_fwd(
        q_fp8.view(-1, NHEADS, QK_DIM), kv_4d, o,
        qo, kv_ind, kv_indices, kv_last,
        1, page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE, logit_cap=0.0, num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=True, **meta,
    )


# Import our MXFP4 kernel
import sys
sys.path.insert(0, '/workspace/mla')
from mla_decode_mxfp4_v5 import mla_decode_mxfp4, _quantize_mxfp4_torch, _pad_tensors


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== Head-to-head: aiter fp8 ASM vs MXFP4 Triton ===\n")
    print(f"nheads={NHEADS}, qk_dim={QK_DIM}, v_dim={V_DIM}")
    print(f"aiter: fp8 KV, ASM stage1 + ASM reduce")
    print(f"ours:  MXFP4 KV, Triton stage1 + Triton reduce\n")

    for BATCH, KV_SL in [(4, 4096), (32, 4096), (61, 4096), (128, 4096)]:
        tq = BATCH
        tkv = BATCH * KV_SL

        # Generate data
        q_bf16 = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        kv_bf16 = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        qo = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
        kv_ind = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SL

        # ── aiter fp8 setup ──
        q_fp8, q_scale = quantize_fp8(q_bf16)
        kv_fp8, kv_scale = quantize_fp8(kv_bf16)
        kv_4d = kv_fp8.view(tkv, PAGE_SIZE, NUM_KV_HEADS, QK_DIM)
        kv_last = (kv_ind[1:] - kv_ind[:-1]).to(torch.int32)
        total_kv = int(kv_ind[-1].item())
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)
        meta = build_aiter_meta(BATCH, qo, kv_ind, q_fp8.dtype, kv_fp8.dtype)
        o_aiter = torch.empty(tq, NHEADS, V_DIM, dtype=torch.bfloat16, device=device)

        # Warmup aiter
        for _ in range(3):
            run_aiter_fp8(q_fp8, q_scale, kv_fp8, kv_scale, kv_4d, kv_indices,
                          kv_last, qo, kv_ind, meta, o_aiter)
        torch.cuda.synchronize()

        # Bench aiter
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        NI = 50
        start.record()
        for _ in range(NI):
            run_aiter_fp8(q_fp8, q_scale, kv_fp8, kv_scale, kv_4d, kv_indices,
                          kv_last, qo, kv_ind, meta, o_aiter)
        end.record()
        torch.cuda.synchronize()
        us_aiter = start.elapsed_time(end) * 1000.0 / NI

        # ── MXFP4 setup ──
        kv_fp4, kv_sc = _quantize_mxfp4_torch(kv_bf16)
        q_pad, kv_data_pad, kv_scale_pad = _pad_tensors(q_bf16, kv_fp4, kv_sc, NHEADS, QK_DIM)

        # Warmup mxfp4
        for _ in range(3):
            mla_decode_mxfp4(q_bf16, kv_fp4, kv_sc, qo, kv_ind,
                             NHEADS, QK_DIM, V_DIM, SM_SCALE, 16, 64,
                             q_pad, kv_data_pad, kv_scale_pad)
        torch.cuda.synchronize()

        # Bench mxfp4
        start.record()
        for _ in range(NI):
            mla_decode_mxfp4(q_bf16, kv_fp4, kv_sc, qo, kv_ind,
                             NHEADS, QK_DIM, V_DIM, SM_SCALE, 16, 64,
                             q_pad, kv_data_pad, kv_scale_pad)
        end.record()
        torch.cuda.synchronize()
        us_mxfp4 = start.elapsed_time(end) * 1000.0 / NI

        # Bandwidth calculations
        fp8_bytes = tkv * QK_DIM + tq * NHEADS * QK_DIM + tq * NHEADS * V_DIM * 2
        fp4_bytes = tkv * (QK_DIM // 2 + QK_DIM // 32) + tq * NHEADS * QK_DIM * 2 + tq * NHEADS * V_DIM * 2
        bw_aiter = fp8_bytes / (us_aiter * 1e-6) / 1e12
        bw_mxfp4 = fp4_bytes / (us_mxfp4 * 1e-6) / 1e12

        ratio = us_aiter / us_mxfp4
        print(f"bs={BATCH:4d} kv={KV_SL}:")
        print(f"  aiter fp8:  {us_aiter:7.1f}μs  BW={bw_aiter:.2f} TB/s  ({fp8_bytes/1e6:.1f}MB)")
        print(f"  mxfp4 tri:  {us_mxfp4:7.1f}μs  BW={bw_mxfp4:.2f} TB/s  ({fp4_bytes/1e6:.1f}MB)")
        print(f"  ratio:      {ratio:.2f}x {'(mxfp4 faster)' if ratio > 1 else '(aiter faster)'}")
        print()
