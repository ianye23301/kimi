# ATOM TP8 Kernel Breakdown

Total GPU kernel time: **54.6 ms**

## Category Summary

- **Communication**: 15.68 ms (28.7%)
- **MoE**: 12.13 ms (22.2%)
- **Dense GEMM**: 8.94 ms (16.4%)
- **MLA Attention**: 8.38 ms (15.4%)
- **Norm/Activation**: 5.02 ms (9.2%)
- **KV/Misc**: 2.21 ms (4.1%)
- **Other**: 2.19 ms (4.0%)

## Per-Kernel Detail

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 24.1 | 488 | 13.13 | 26.9 | AllReduce (reduce_scatter) | dtype=bfloat16_t, ranks=8 |
| 2 | 7.3 | 244 | 3.99 | 16.3 | MLA Decode (a8w8) | qheads=16, seqlen=1, gqa_ratio=16 |
| 3 | 7.1 | 480 | 3.88 | 8.1 | MoE Weighted Sum (wv_splitk) | __hip_bfloat16, 64, 1, 1, 8, 4, 1 |
| 4 | 6.3 | 244 | 3.45 | 14.1 | AITER bf16 GEMV 32x64 splitk | bf16->fp32->bf16 |
| 5 | 5.7 | 484 | 3.13 | 6.5 | hipBLASLt GEMM | tile=32x16x128, splitK=3, ISA=950, WS=64, WG=16_4_4 |
| 6 | 4.6 | 240 | 2.52 | 10.5 | MoE CK Flatmm gate+up (stage1) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 7 | 4.3 | 240 | 2.32 | 9.7 | MoE Sorting (topk route) | int/float, sorted |
| 8 | 4.1 | 488 | 2.22 | 4.6 | Fused Load+RMSNorm+AllReduce | dtype=bfloat16_t, hidden=512 |
| 9 | 3.6 | 244 | 1.97 | 8.1 | hipBLASLt GEMM | tile=16x16x256, splitK=3, ISA=950, WS=64, WG=16_4_2 |
| 10 | 3.6 | 244 | 1.94 | 8.0 | MLA Reduce | block=512, heads=16, splits=1 |
| 11 | 3.5 | 480 | 1.93 | 4.0 | SiLU Activation (act_and_mul) | bf16 |
| 12 | 3.2 | 240 | 1.77 | 7.4 | MoE Grouped TopK | c10::BFloat16, float __vector(4), 1, true, true, false |
| 13 | 3.0 | 240 | 1.65 | 6.9 | MoE CK Flatmm down (stage2) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 14 | 2.4 | 244 | 1.33 | 5.5 | MLA Batched GEMM (a8w8) | M=16, N=32, K=128, grid=4 |
| 15 | 2.1 | 244 | 1.13 | 4.6 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::float8_copy_ker |  |
| 16 | 1.9 | 244 | 1.06 | 4.3 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_ker |  |
| 17 | 1.9 | 244 | 1.04 | 4.3 | Fused Add+RMSNorm+Quant | hidden=64 |
| 18 | 1.9 | 244 | 1.03 | 4.2 | Fused Add+RMSNorm+Quant | hidden=256 |
| 19 | 1.8 | 244 | 1.00 | 4.1 | MLA Batched GEMM (a8w8) | M=16, N=128, K=128, grid=4 |
| 20 | 1.8 | 240 | 0.98 | 4.1 | Triton Fused Add+AllReduce+RMSNorm |  |
| 21 | 1.8 | 240 | 0.97 | 4.0 | Fill (zeros) | bf16 |
| 22 | 1.8 | 244 | 0.97 | 4.0 | Fused RoPE+KV Cache Store | bf16->fp8 |
| 23 | 0.5 | 5 | 0.27 | 53.6 | hipBLASLt GEMM | tile=256x16x128, splitK=3, ISA=950, WS=64, WG=64_4_1 |
| 24 | 0.4 | 5 | 0.22 | 44.6 | Sampling (mix_sample) | vocab=1024 |
| 25 | 0.4 | 4 | 0.21 | 52.6 | AllReduce (cross_device_reduce) |  |
| 26 | 0.2 | 4 | 0.12 | 30.8 | MLA Metadata | page_size=128 |
| 27 | 0.1 | 5 | 0.06 | 12.8 | NCCL Collective |  |
| 28 | 0.1 | 4 | 0.06 | 14.2 | hipBLASLt GEMM | tile=256x16x64, splitK=3, ISA=950, WS=64, WG=64_4_1 |
| 29 | 0.1 | 5 | 0.05 | 10.4 | AllGather | dtype=bfloat16_t |
| 30 | 0.1 | 4 | 0.04 | 11.0 | hipBLASLt GEMM | tile=32x16x256, splitK=3, ISA=950, WS=64, WG=16_4_4 |
