"""Benchmark aiter's fp8 MLA decode for comparison."""
import os
os.environ.pop("HIP_VISIBLE_DEVICES", None)
import torch
torch.cuda.set_device(0)
from aiter import dtypes as aiter_dtypes
from aiter.mla import mla_decode_fwd, _fwd_kernel_stage2_asm
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8

NHEADS = 16
NUM_KV_HEADS = 1
QK_DIM = 576
V_DIM = 512
SM_SCALE = 1.0 / (QK_DIM ** 0.5)
PAGE_SIZE = 1
NUM_KV_SPLITS = 16

def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)

def run_aiter_mla(q_fp8, q_scale, kv_fp8, kv_scale, qo_indptr, kv_indptr, batch_size, max_q_len):
    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, NHEADS, q_fp8.dtype, kv_fp8.dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=NUM_KV_SPLITS, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        NHEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len, uni_seqlen_qo=max_q_len,
        fast_mode=False, max_split_per_batch=NUM_KV_SPLITS,
        intra_batch_mode=True, dtype_q=q_fp8.dtype, dtype_kv=kv_fp8.dtype,
    )

    meta = {
        "work_meta_data": work_metadata, "work_indptr": work_indptr,
        "work_info_set": work_info_set, "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map, "reduce_partial_map": reduce_partial_map,
    }

    o = torch.empty((q_fp8.shape[0], NHEADS, V_DIM), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q_fp8.view(-1, NHEADS, QK_DIM), kv_4d, o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        max_q_len, page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE, logit_cap=0.0, num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=True, **meta,
    )
    return o


if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(42)

    print("=== aiter fp8 MLA decode benchmark ===\n")

    for BATCH, KV_SL in [(4, 4096), (32, 4096), (61, 4096), (128, 4096)]:
        tq = BATCH
        tkv = BATCH * KV_SL
        q_bf16 = torch.randn(tq, NHEADS, QK_DIM, dtype=torch.bfloat16, device=device)
        q_fp8, q_scale = quantize_fp8(q_bf16)
        kv_bf16 = torch.randn(tkv, QK_DIM, dtype=torch.bfloat16, device=device) * 0.1
        kv_fp8, kv_scale = quantize_fp8(kv_bf16)
        qo = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
        kv_ind = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * KV_SL

        # Warmup (with metadata pre-built)
        _ = run_aiter_mla(q_fp8, q_scale, kv_fp8, kv_scale, qo, kv_ind, BATCH, 1)
        torch.cuda.synchronize()

        # Time total call
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        NI = 50
        start.record()
        for _ in range(NI):
            run_aiter_mla(q_fp8, q_scale, kv_fp8, kv_scale, qo, kv_ind, BATCH, 1)
        end.record()
        torch.cuda.synchronize()
        us_total = start.elapsed_time(end) * 1000.0 / NI

        # Time with pre-built metadata (skip python overhead)
        # The mla_decode_fwd function includes stage1_asm + stage2_reduce
        # Build metadata once
        total_kv_len = int(kv_ind[-1].item())
        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])
        kv_last_page_len = (kv_ind[1:] - kv_ind[:-1]).to(torch.int32)
        info = get_mla_metadata_info_v1(
            BATCH, 1, NHEADS, q_fp8.dtype, kv_fp8.dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=NUM_KV_SPLITS, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        get_mla_metadata_v1(
            qo, kv_ind, kv_last_page_len,
            NHEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            *work, page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
            max_split_per_batch=NUM_KV_SPLITS, intra_batch_mode=True,
            dtype_q=q_fp8.dtype, dtype_kv=kv_fp8.dtype,
        )
        meta = {
            "work_meta_data": work[0], "work_indptr": work[1],
            "work_info_set": work[2], "reduce_indptr": work[3],
            "reduce_final_map": work[4], "reduce_partial_map": work[5],
        }
        o = torch.empty((tq, NHEADS, V_DIM), dtype=torch.bfloat16, device="cuda")

        # Time just mla_decode_fwd (metadata pre-built)
        for _ in range(5):
            mla_decode_fwd(
                q_fp8.view(-1, NHEADS, QK_DIM), kv_4d, o,
                qo, kv_ind, kv_indices, kv_last_page_len,
                1, page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS,
                sm_scale=SM_SCALE, logit_cap=0.0, num_kv_splits=NUM_KV_SPLITS,
                q_scale=q_scale, kv_scale=kv_scale,
                intra_batch_mode=True, **meta,
            )
        torch.cuda.synchronize()
        start.record()
        for _ in range(NI):
            mla_decode_fwd(
                q_fp8.view(-1, NHEADS, QK_DIM), kv_4d, o,
                qo, kv_ind, kv_indices, kv_last_page_len,
                1, page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS,
                sm_scale=SM_SCALE, logit_cap=0.0, num_kv_splits=NUM_KV_SPLITS,
                q_scale=q_scale, kv_scale=kv_scale,
                intra_batch_mode=True, **meta,
            )
        end.record()
        torch.cuda.synchronize()
        us_kernel = start.elapsed_time(end) * 1000.0 / NI

        kv_bytes = tkv * QK_DIM  # fp8
        q_bytes = tq * NHEADS * QK_DIM
        o_bytes = tq * NHEADS * V_DIM * 2
        total = q_bytes + kv_bytes + o_bytes
        bw = total / (us_kernel * 1e-6) / 1e12

        print(f"  bs={BATCH:4d} kv={KV_SL}: total={us_total:7.1f}μs  kernel={us_kernel:7.1f}μs  "
              f"BW={bw:.2f} TB/s  ({total/1e6:.1f}MB)")
