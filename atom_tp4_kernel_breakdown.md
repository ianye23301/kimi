# ATOM TP4 Kernel Breakdown

Total GPU kernel time: **46.0 ms**

## Category Summary

- **Dense GEMM**: 13.87 ms (30.2%)
- **MoE**: 10.95 ms (23.8%)
- **MLA Attention**: 8.41 ms (18.3%)
- **Communication**: 5.58 ms (12.1%)
- **Norm/Activation**: 4.96 ms (10.8%)
- **KV/Misc**: 2.19 ms (4.8%)

## Per-Kernel Detail

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 13.3 | 488 | 6.13 | 12.6 | AITER bf16 GEMV 32x64 splitk | bf16->fp32->bf16 |
| 2 | 10.8 | 480 | 4.97 | 10.4 | hipBLASLt GEMM | tile=32x16x128, splitK=3, ISA=950, WS=64, WG=16_4_4 |
| 3 | 8.8 | 244 | 4.02 | 16.5 | MLA Decode (a8w8) | qheads=16, seqlen=1, gqa_ratio=16 |
| 4 | 7.2 | 488 | 3.29 | 6.7 | AllReduce (reduce_scatter) | dtype=bfloat16_t, ranks=4 |
| 5 | 7.0 | 240 | 3.24 | 13.5 | MoE CK Flatmm gate+up (stage1) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 6 | 5.0 | 240 | 2.30 | 9.6 | MoE Sorting (topk route) | int/float, sorted |
| 7 | 4.7 | 488 | 2.16 | 4.4 | Fused Load+RMSNorm+AllReduce | dtype=bfloat16_t, hidden=512 |
| 8 | 4.6 | 244 | 2.10 | 8.6 | AITER bf16 GEMV 64x64 splitk | bf16->fp32->bf16 |
| 9 | 4.3 | 484 | 1.96 | 4.0 | SiLU Activation (act_and_mul) | bf16 |
| 10 | 4.2 | 240 | 1.95 | 8.1 | MoE Weighted Sum (wv_splitk) | __hip_bfloat16, 64, 1, 1, 8, 4, 1 |
| 11 | 3.9 | 244 | 1.81 | 7.4 | MLA Reduce | block=512, heads=16, splits=1 |
| 12 | 3.8 | 240 | 1.74 | 7.2 | MoE CK Flatmm down (stage2) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 13 | 3.8 | 240 | 1.72 | 7.2 | MoE Grouped TopK | c10::BFloat16, float __vector(4), 1, true, true, false |
| 14 | 3.0 | 244 | 1.40 | 5.7 | MLA Batched GEMM (a8w8) | M=16, N=32, K=128, grid=4 |
| 15 | 2.3 | 244 | 1.06 | 4.3 | MLA Batched GEMM (a8w8) | M=16, N=128, K=128, grid=4 |
| 16 | 2.2 | 244 | 1.02 | 4.2 | Fused Add+RMSNorm+Quant | hidden=256 |
| 17 | 2.2 | 240 | 0.99 | 4.1 | Triton Fused Add+AllReduce+RMSNorm |  |
| 18 | 2.1 | 244 | 0.98 | 4.0 | Fused Add+RMSNorm+Quant | hidden=64 |
| 19 | 2.1 | 244 | 0.97 | 4.0 | Fused RoPE+KV Cache Store | bf16->fp8 |
| 20 | 2.1 | 240 | 0.95 | 4.0 | Fill (zeros) | bf16 |
| 21 | 1.0 | 5 | 0.46 | 92.7 | hipBLASLt GEMM | tile=256x16x128, splitK=3, ISA=950, WS=64, WG=64_4_1 |
| 22 | 0.5 | 5 | 0.22 | 43.8 | Sampling (mix_sample) | vocab=1024 |
| 23 | 0.3 | 8 | 0.15 | 19.0 | hipBLASLt GEMM | tile=256x16x64, splitK=3, ISA=950, WS=64, WG=64_4_1 |
| 24 | 0.3 | 4 | 0.12 | 30.7 | MLA Metadata | page_size=128 |
| 25 | 0.1 | 5 | 0.06 | 12.5 | NCCL Collective |  |
| 26 | 0.1 | 4 | 0.04 | 9.2 | AllReduce (cross_device_reduce) |  |
| 27 | 0.1 | 5 | 0.04 | 7.1 | AllGather | dtype=bfloat16_t |
| 28 | 0.1 | 4 | 0.03 | 6.8 | hipBLASLt GEMM (PostGSU16) | tile=? |
