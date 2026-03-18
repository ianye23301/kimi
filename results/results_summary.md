# Kimi K2.5 MXFP4 — Baseline Benchmark Results

**Date**: 2026-03-17
**Model**: Kimi-K2.5-MXFP4 (1T params, MXFP4 quantized, 256 experts, top-8, 64 attn heads, 61 layers)
**Hardware**: 8x AMD MI355X (gfx950), 288GB HBM per GPU
**Benchmark**: ISL=8000, OSL=1024, random dataset, request_rate=inf

## Performance Targets

| Conc | Interactivity (tok/s/user) | Throughput/GPU (tok/s) | E2E Latency (s) |
|-----:|---------------------------:|-----------------------:|-----------------:|
| 4 | ≥150 | ≥1350 | ≤6.0 |
| 32 | ≥65 | ≥4500 | ≤14.0 |
| 128 | ≥35 | ≥5300 | ≤24.5 |

Interactivity = 1000 / median_TPOT. Throughput/GPU = output_tok/s ÷ num_GPUs.

## Results: conc=4

| Metric | ATOM TP4 | ATOM TP8 | vLLM TP4 | vLLM TP8 | Target |
|---|---:|---:|---:|---:|---:|
| Interactivity (tok/s/user) | **74.8** | — | 68.1 | 60.2 | ≥150 |
| Throughput/GPU (tok/s) | **71.1** | — | 64.9 | 28.8 | ≥1350 |
| Median E2E (s) | **12.7** | — | 13.9 | 15.7 | ≤6.0 |
| Median TPOT (ms) | **13.4** | — | 14.7 | 16.6 | — |
| Output tok/s (total) | 284.3 | — | 259.6 | 230.8 | — |

## Results: conc=32

| Metric | ATOM TP4 | ATOM TP8 | vLLM TP4 | vLLM TP8 | Target |
|---|---:|---:|---:|---:|---:|
| Interactivity (tok/s/user) | **31.0** | — | 26.5 | 29.6 | ≥65 |
| Throughput/GPU (tok/s) | **238.3** | — | 203.4 | 113.4 | ≥4500 |
| Median E2E (s) | **30.4** | — | 35.5 | 31.8 | ≤14.0 |
| Median TPOT (ms) | **32.3** | — | 37.7 | 33.8 | — |
| Output tok/s (total) | 953.2 | — | 813.7 | 907.3 | — |

## Results: conc=128

| Metric | ATOM TP4 | ATOM TP8 | vLLM TP4 | vLLM TP8 | Target |
|---|---:|---:|---:|---:|---:|
| Interactivity (tok/s/user) | **12.3** | — | 10.3 | 12.7 | ≥35 |
| Throughput/GPU (tok/s) | **388.0** | — | 322.2 | 198.7 | ≥5300 |
| Median E2E (s) | **75.1** | — | 90.1 | 72.8 | ≤24.5 |
| Median TPOT (ms) | **81.3** | — | 97.2 | 78.8 | — |
| Output tok/s (total) | 1552.0 | — | 1288.8 | 1589.4 | — |

## Key Observations

- **ATOM TP4 is the best config** for throughput/GPU across all concurrencies
- **vLLM TP8** has higher total throughput at conc=128 (1589 vs 1552 tok/s) but worse per-GPU efficiency (198.7 vs 388.0 tok/s/GPU)
- **TP8 hurts per-GPU efficiency** — communication overhead from 8-way AllReduce dominates (28.7% of GPU time at TP8 vs 12.1% at TP4)
- All configs are **~10-20x below targets** — targets likely assume extensive optimization (speculative decoding, kernel fusion, scheduling improvements)
- **vLLM TP8 required a patch** to AITER MLA (head-repeat 8→16) — see below

## vLLM TP8 AITER MLA Patch

AITER's MLA kernel requires 16 or 128 attention heads per rank. With TP8, Kimi K2.5 has 64/8=8 heads/rank.

Patch applied to `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`:
- Replaced hard assertion with head padding: `padded_num_heads = max(num_heads, 16)`
- Before decode kernel: `q = q.repeat_interleave(repeat_factor, dim=1)` (8→16 heads)
- After decode kernel: `o = o[:, ::repeat_factor, :].contiguous()` (stride back to 8)

Same approach used by ATOM (`atom/model_ops/attention_mla.py`).

---

## Kernel Breakdown: ATOM TP4

Total GPU kernel time: **46.0 ms**

### Category Summary

- **Dense GEMM**: 13.87 ms (30.2%)
- **MoE**: 10.95 ms (23.8%)
- **MLA Attention**: 8.41 ms (18.3%)
- **Communication**: 5.58 ms (12.1%)
- **Norm/Activation**: 4.96 ms (10.8%)
- **KV/Misc**: 2.19 ms (4.8%)

### Top Kernels

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 13.3 | 488 | 6.13 | 12.6 | AITER bf16 GEMV 32x64 splitk | bf16->fp32->bf16 |
| 2 | 10.8 | 480 | 4.97 | 10.4 | hipBLASLt GEMM | tile=32x16x128, splitK=3 |
| 3 | 8.8 | 244 | 4.02 | 16.5 | MLA Decode (a8w8) | qheads=16, seqlen=1, gqa_ratio=16 |
| 4 | 7.2 | 488 | 3.29 | 6.7 | AllReduce (reduce_scatter) | ranks=4 |
| 5 | 7.0 | 240 | 3.24 | 13.5 | MoE CK Flatmm gate+up (stage1) | tile=<16,128,256>, MXFP4->bf16 |
| 6 | 5.0 | 240 | 2.30 | 9.6 | MoE Sorting (topk route) | int/float |
| 7 | 4.7 | 488 | 2.16 | 4.4 | Fused Load+RMSNorm+AllReduce | hidden=512 |
| 8 | 4.6 | 244 | 2.10 | 8.6 | AITER bf16 GEMV 64x64 splitk | bf16->fp32->bf16 |
| 9 | 4.3 | 484 | 1.96 | 4.0 | SiLU Activation (act_and_mul) | bf16 |
| 10 | 4.2 | 240 | 1.95 | 8.1 | MoE Weighted Sum (wv_splitk) | bf16 |
| 11 | 3.9 | 244 | 1.81 | 7.4 | MLA Reduce | heads=16, splits=1 |
| 12 | 3.8 | 240 | 1.74 | 7.2 | MoE CK Flatmm down (stage2) | tile=<16,128,256>, MXFP4->bf16 |
| 13 | 3.8 | 240 | 1.72 | 7.2 | MoE Grouped TopK | bf16 |
| 14 | 3.0 | 244 | 1.40 | 5.7 | MLA Batched GEMM (a8w8) | M=16, N=32, K=128 |
| 15 | 2.3 | 244 | 1.06 | 4.3 | MLA Batched GEMM (a8w8) | M=16, N=128, K=128 |

---

## Kernel Breakdown: ATOM TP8

Total GPU kernel time: **54.6 ms** (+18.7% vs TP4)

### Category Summary

- **Communication**: 15.68 ms (28.7%) — up from 12.1%
- **MoE**: 12.13 ms (22.2%)
- **Dense GEMM**: 8.94 ms (16.4%) — down from 30.2%
- **MLA Attention**: 8.38 ms (15.4%)
- **Norm/Activation**: 5.02 ms (9.2%)
- **KV/Misc**: 2.21 ms (4.1%)
- **Other**: 2.19 ms (4.0%)

### Top Kernels

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 24.1 | 488 | 13.13 | 26.9 | AllReduce (reduce_scatter) | ranks=8 |
| 2 | 7.3 | 244 | 3.99 | 16.3 | MLA Decode (a8w8) | qheads=16, seqlen=1, gqa_ratio=16 |
| 3 | 7.1 | 480 | 3.88 | 8.1 | MoE Weighted Sum (wv_splitk) | bf16 |
| 4 | 6.3 | 244 | 3.45 | 14.1 | AITER bf16 GEMV 32x64 splitk | bf16->fp32->bf16 |
| 5 | 5.7 | 484 | 3.13 | 6.5 | hipBLASLt GEMM | tile=32x16x128, splitK=3 |
| 6 | 4.6 | 240 | 2.52 | 10.5 | MoE CK Flatmm gate+up (stage1) | tile=<16,128,256>, MXFP4->bf16 |
| 7 | 4.3 | 240 | 2.32 | 9.7 | MoE Sorting (topk route) | int/float |
| 8 | 4.1 | 488 | 2.22 | 4.6 | Fused Load+RMSNorm+AllReduce | hidden=512 |
| 9 | 3.6 | 244 | 1.97 | 8.1 | hipBLASLt GEMM | tile=16x16x256, splitK=3 |
| 10 | 3.6 | 244 | 1.94 | 8.0 | MLA Reduce | heads=16, splits=1 |
| 11 | 3.5 | 480 | 1.93 | 4.0 | SiLU Activation (act_and_mul) | bf16 |
| 12 | 3.2 | 240 | 1.77 | 7.4 | MoE Grouped TopK | bf16 |
| 13 | 3.0 | 240 | 1.65 | 6.9 | MoE CK Flatmm down (stage2) | tile=<16,128,256>, MXFP4->bf16 |
| 14 | 2.4 | 244 | 1.33 | 5.5 | MLA Batched GEMM (a8w8) | M=16, N=32, K=128 |
| 15 | 2.1 | 244 | 1.13 | 4.6 | float8_copy | elementwise |
| 16 | 1.9 | 244 | 1.06 | 4.3 | direct_copy | elementwise |
| 17 | 1.9 | 244 | 1.04 | 4.3 | Fused Add+RMSNorm+Quant | hidden=64 |
| 18 | 1.9 | 244 | 1.03 | 4.2 | Fused Add+RMSNorm+Quant | hidden=256 |
| 19 | 1.8 | 244 | 1.00 | 4.1 | MLA Batched GEMM (a8w8) | M=16, N=128, K=128 |
| 20 | 1.8 | 240 | 0.98 | 4.1 | Triton Fused Add+AllReduce+RMSNorm | |

### TP4 vs TP8 Key Differences

- **AllReduce**: 3.29ms (6.7us avg, 4 ranks) → 13.13ms (26.9us avg, 8 ranks) — **4x increase**, now #1 bottleneck
- **Dense GEMM**: 11.10ms → 6.58ms — smaller weight shards per rank
- **MoE kernels**: roughly same total time
- **MLA Decode**: ~same (both use padded 16 heads)
- **New at TP8**: float8_copy + direct_copy kernels (~4% total) from head-repeat overhead
- **Net**: TP8 is 18.7% slower per-step despite 2x more GPUs

---

## Kernel Breakdown: vLLM TP4

Total GPU kernel time: **2163.1 ms** (includes prefill + decode for 4x 8k-token prompts, 128 output tokens each)

### Category Summary

- **Other** (unfused ops, copies, triton fused): 1233.18 ms (57.0%)
- **MoE**: 432.60 ms (20.0%)
- **Communication**: 325.91 ms (15.1%)
- **KV/Misc**: 90.75 ms (4.2%)
- **Dense GEMM**: 35.80 ms (1.7%)
- **Norm/Activation**: 31.26 ms (1.4%)

### Top Kernels

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 10.2 | 31232 | 220.41 | 7.1 | wvSplitK (MoE weighted sum) | bf16, 64, 2, 16, 8, 2, 4 |
| 2 | 9.7 | 253 | 209.07 | 826.4 | NCCL Collective | |
| 3 | 8.2 | 7740 | 177.44 | 22.9 | MoE CK Flatmm gate+up (stage1) | tile=<16,128,256>, MXFP4->bf16 |
| 4 | 7.4 | 7869 | 160.06 | 20.3 | MLA Decode (aiter) | mla_dec_stage1_bf16_a16w16_subQ16_mqa16 |
| 5 | 5.7 | 7740 | 122.53 | 15.8 | MoE CK Flatmm down (stage2) | tile=<16,128,256>, MXFP4->bf16 |
| 6 | 5.5 | 15860 | 118.08 | 7.4 | CK RMSNorm | |
| 7 | 5.4 | 15867 | 116.84 | 7.4 | AllReduce (cross_device_reduce) | |
| 8 | 4.9 | 15360 | 106.54 | 6.9 | wvSplitK (MoE weighted sum) | bf16, 64, 1, 16, 8, 4, 4 |
| 9 | 3.9 | 60 | 85.19 | 1419.8 | CK MoE MXGEMM (prefill) | mxgemm_2lds |
| 10 | 3.6 | 7740 | 78.15 | 10.1 | MoE Sorting (topk route) | int/float |
| 11 | 3.1 | 15609 | 66.59 | 4.3 | Fill (zeros) | bf16 |
| 12 | 2.9 | 7869 | 63.31 | 8.0 | MLA Decode stage2 (asm) | _fwd_kernel_stage2_asm |
| 13 | 2.5 | 7800 | 54.49 | 7.0 | MoE Grouped TopK | bf16 |
| 14 | 2.1 | 7869 | 44.63 | 5.7 | CatArrayBatchedCopy | |
| 15 | 1.7 | 7930 | 37.54 | 4.7 | CK RMSNorm (variant 2) | |

---

## Kernel Breakdown: vLLM TP8

Total GPU kernel time: **2388.6 ms** (+10.4% vs TP4)

### Category Summary

- **Other** (unfused ops, copies, triton fused): 1200.53 ms (50.3%)
- **Communication**: 654.24 ms (27.4%) — up from 15.1%
- **MoE**: 368.81 ms (15.4%)
- **KV/Misc**: 90.27 ms (3.8%)
- **Norm/Activation**: 31.52 ms (1.3%)
- **Dense GEMM**: 29.09 ms (1.2%)

### Top Kernels

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 22.8 | 15867 | 545.02 | 34.3 | AllReduce (cross_device_reduce) | |
| 2 | 8.1 | 31360 | 194.14 | 6.2 | wvSplitK (MoE weighted sum) | bf16, 64, 2, 16, 8, 2, 4 |
| 3 | 6.8 | 7869 | 161.80 | 20.6 | MLA Decode (aiter) | mla_dec_stage1_bf16_a16w16_subQ16_mqa16 |
| 4 | 5.2 | 7680 | 123.87 | 16.1 | MoE CK Flatmm gate+up (stage1) | tile=<16,128,256>, MXFP4->bf16 |
| 5 | 4.8 | 15860 | 115.77 | 7.3 | CK RMSNorm | |
| 6 | 4.7 | 7740 | 112.97 | 14.6 | MoE CK Flatmm down (stage2) | tile=<16,128,256>, MXFP4->bf16 |
| 7 | 4.6 | 253 | 109.22 | 431.7 | NCCL Collective | |
| 8 | 4.3 | 15360 | 103.62 | 6.7 | wvSplitK (MoE weighted sum) | bf16, 64, 1, 16, 8, 4, 4 |
| 9 | 3.3 | 7740 | 77.99 | 10.1 | MoE Sorting (topk route) | int/float |
| 10 | 3.0 | 15988 | 72.41 | 4.5 | direct_copy | elementwise |
| 11 | 2.8 | 15609 | 66.14 | 4.2 | Fill (zeros) | bf16 |
| 12 | 2.7 | 7869 | 63.87 | 8.1 | MLA Decode stage2 (asm) | _fwd_kernel_stage2_asm |
| 13 | 2.2 | 7800 | 53.05 | 6.8 | MoE Grouped TopK | bf16 |
| 14 | 1.9 | 7869 | 45.33 | 5.8 | CatArrayBatchedCopy | |
| 15 | 1.8 | 60 | 43.49 | 724.8 | CK MoE MXGEMM (prefill) | mxgemm_2lds |

### vLLM TP4 vs TP8 Key Differences

- **AllReduce**: 116.84ms → 545.02ms — **4.7x increase**, now #1 bottleneck at 22.8%
- **NCCL Collective**: 209.07ms → 109.22ms — fewer large prefill collectives
- **MoE Flatmm stage1**: 177.44ms → 123.87ms — fewer experts per rank
- **MLA Decode**: roughly same (~160ms)
- **New at TP8**: direct_copy 72.41ms (3.0%) from head-repeat overhead

---

## ATOM vs vLLM Kernel Comparison

Key differences (decode-only, normalized per step):

- **ATOM uses CUDA graphs** — eliminates kernel launch overhead, fewer Fill/copy kernels
- **ATOM fuses more ops** — Fused Load+RMSNorm+AllReduce, Fused Add+RMSNorm+Quant vs separate CK RMSNorm + triton fused ops in vLLM
- **vLLM has higher "Other" overhead** (57% vs ~5% in ATOM) — unfused triton ops, copies, CatArrayBatchedCopy
- **MoE kernels are identical** — both use CK Flatmm MXFP4 2-stage
- **MLA Decode kernels are identical** — both use aiter mla_dec_stage1
- **Communication is similar** — both use RCCL AllReduce, TP8 is 4-5x more expensive than TP4
