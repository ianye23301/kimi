# vLLM TP8 Kernel Breakdown

Total GPU kernel time: **2388.6 ms**

## Category Summary

- **Other**: 1200.53 ms (50.3%)
- **Communication**: 654.24 ms (27.4%)
- **MoE**: 368.81 ms (15.4%)
- **KV/Misc**: 90.27 ms (3.8%)
- **Norm/Activation**: 31.52 ms (1.3%)
- **Dense GEMM**: 29.09 ms (1.2%)

## Per-Kernel Detail

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 22.8 | 15867 | 545.02 | 34.3 | AllReduce (cross_device_reduce) |  |
| 2 | 8.1 | 31360 | 194.14 | 6.2 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 3 | 6.8 | 7869 | 161.80 | 20.6 | aiter::mla_dec_stage1_bf16_a16w16_subQ16_mqa16 |  |
| 4 | 5.2 | 7680 | 123.87 | 16.1 | MoE CK Flatmm gate+up (stage1) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 5 | 4.8 | 15860 | 115.77 | 7.3 | void ck_tile::kentry<1, ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<ck_tile::Rmsnorm2dFwdPipelineProblem< |  |
| 6 | 4.7 | 7740 | 112.97 | 14.6 | MoE CK Flatmm down (stage2) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 7 | 4.6 | 253 | 109.22 | 431.7 | NCCL Collective |  |
| 8 | 4.3 | 15360 | 103.62 | 6.7 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 1, 16, 8, 4, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 9 | 3.3 | 7740 | 77.99 | 10.1 | MoE Sorting (topk route) | int/float, sorted |
| 10 | 3.0 | 15988 | 72.41 | 4.5 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_ker |  |
| 11 | 2.8 | 15609 | 66.14 | 4.2 | Fill (zeros) | bf16 |
| 12 | 2.7 | 7869 | 63.87 | 8.1 | _fwd_kernel_stage2_asm |  |
| 13 | 2.2 | 7800 | 53.05 | 6.8 | MoE Grouped TopK | c10::BFloat16, float __vector(4), 1, true, true, false |
| 14 | 1.9 | 7869 | 45.33 | 5.8 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned  |  |
| 15 | 1.8 | 60 | 43.49 | 724.8 | void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout:: |  |
| 16 | 1.5 | 7930 | 37.02 | 4.7 | void ck_tile::kentry<1, ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<ck_tile::Rmsnorm2dFwdPipelineProblem< |  |
| 17 | 1.5 | 7930 | 36.04 | 4.5 | void ck_tile::kentry<1, ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<ck_tile::Rmsnorm2dFwdPipelineProblem< |  |
| 18 | 1.5 | 7930 | 35.63 | 4.5 | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, __hip_bfloat16, (vllm::Fp8KVCacheDataType)0>(__hip_bfloat16 const |  |
| 19 | 1.4 | 7800 | 34.15 | 4.4 | triton_poi_fused_add_clone_copy_expand_index_mul_neg_slice_split_stack_unsqueeze_view_2 |  |
| 20 | 1.4 | 7800 | 33.23 | 4.3 | triton_poi_fused_add_all_reduce_0 |  |
| 21 | 1.4 | 7869 | 33.14 | 4.2 | _batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_32_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SI |  |
| 22 | 1.4 | 7800 | 33.03 | 4.2 | triton_poi_fused_add_clone_expand_index_mul_neg_slice_split_split_with_sizes_stack_unsqueeze_view_1 |  |
| 23 | 1.4 | 7800 | 32.93 | 4.2 | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambd |  |
| 24 | 1.3 | 7740 | 31.52 | 4.1 | SiLU Activation (act_and_mul) | bf16 |
| 25 | 1.3 | 7869 | 31.47 | 4.0 | _batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_8_BLOCK_SIZE_N_32_BLOCK_SIZE_K_512_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SI |  |
| 26 | 1.3 | 7800 | 30.49 | 3.9 | triton_poi_fused_mul_silu_slice_0 |  |
| 27 | 1.0 | 127 | 24.13 | 190.0 | Fill (zeros) | bf16 |
| 28 | 0.6 | 61 | 13.42 | 220.1 | hipBLASLt GEMM | tile=160x256x64, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 29 | 0.5 | 60 | 10.87 | 181.2 | void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout:: |  |
| 30 | 0.4 | 61 | 9.12 | 149.6 | aiter::fmha_fwd_hd192_hd128_bf16_causal_group |  |
| 31 | 0.4 | 62 | 8.71 | 140.5 | Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_NTD_SK3_UserArgs_MT256x256x64_MI16x16x1_shortname0_gfx950 |  |
| 32 | 0.3 | 130 | 6.96 | 53.6 | void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float, at::native::(anonymous namespace |  |
| 33 | 0.3 | 130 | 6.44 | 49.5 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4,  |  |
| 34 | 0.3 | 126 | 6.11 | 48.5 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 4, 16, 8, 1, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 35 | 0.2 | 60 | 4.07 | 67.9 | hipBLASLt GEMM | tile=128x128x128, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 36 | 0.2 | 60 | 3.68 | 61.3 | hipBLASLt GEMM | tile=128x96x128, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 37 | 0.1 | 60 | 3.36 | 56.0 | hipBLASLt GEMM | tile=128x256x32, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 38 | 0.1 | 120 | 3.04 | 25.3 | _moe_mxfp4_sort_kernel |  |
| 39 | 0.1 | 62 | 2.83 | 45.6 | hipBLASLt GEMM | tile=256x192x64, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 40 | 0.1 | 120 | 2.28 | 19.0 | void aiter::dynamic_per_group_scaled_quant_kernel<std::bfloat16_t, ck_tile::fp4x2_t, 32>(ck_tile::fp4x2_t*, float*, std: |  |
| 41 | 0.1 | 60 | 1.74 | 28.9 | void ck_tile::kentry<2, ck_tile::MoeSortingMultiPhaseKernel_P23<ck_tile::MoeSortingProblemMp<int, float, unsigned char,  |  |
| 42 | 0.1 | 61 | 1.72 | 28.2 | hipBLASLt GEMM | tile=256x256x32, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 43 | 0.1 | 129 | 1.50 | 11.7 | void rocprim::ROCPRIM_400000_NS::detail::partition_kernel<(rocprim::ROCPRIM_400000_NS::detail::select_method)0, true, ro |  |
| 44 | 0.1 | 247 | 1.40 | 5.7 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 1>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
