# vLLM TP4 Kernel Breakdown

Total GPU kernel time: **2163.1 ms**

## Category Summary

- **Other**: 1233.18 ms (57.0%)
- **MoE**: 432.60 ms (20.0%)
- **Communication**: 325.91 ms (15.1%)
- **KV/Misc**: 90.75 ms (4.2%)
- **Dense GEMM**: 35.80 ms (1.7%)
- **Norm/Activation**: 31.26 ms (1.4%)

## Per-Kernel Detail

| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |
|--:|----:|------:|-----------:|---------:|--------|--------|
| 1 | 10.2 | 31232 | 220.41 | 7.1 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 2 | 9.7 | 253 | 209.07 | 826.4 | NCCL Collective |  |
| 3 | 8.2 | 7740 | 177.44 | 22.9 | MoE CK Flatmm gate+up (stage1) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 4 | 7.4 | 7869 | 160.06 | 20.3 | aiter::mla_dec_stage1_bf16_a16w16_subQ16_mqa16 |  |
| 5 | 5.7 | 7740 | 122.53 | 15.8 | MoE CK Flatmm down (stage2) | tile=<16, 128, 256>, MXFP4->bf16, scale=e8m0/group32 |
| 6 | 5.5 | 15860 | 118.08 | 7.4 | void ck_tile::kentry<1, ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<ck_tile::Rmsnorm2dFwdPipelineProblem< |  |
| 7 | 5.4 | 15867 | 116.84 | 7.4 | AllReduce (cross_device_reduce) |  |
| 8 | 4.9 | 15360 | 106.54 | 6.9 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 1, 16, 8, 4, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 9 | 3.9 | 60 | 85.19 | 1419.8 | void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout:: |  |
| 10 | 3.6 | 7740 | 78.15 | 10.1 | MoE Sorting (topk route) | int/float, sorted |
| 11 | 3.1 | 15609 | 66.59 | 4.3 | Fill (zeros) | bf16 |
| 12 | 2.9 | 7869 | 63.31 | 8.0 | _fwd_kernel_stage2_asm |  |
| 13 | 2.5 | 7800 | 54.49 | 7.0 | MoE Grouped TopK | c10::BFloat16, float __vector(4), 1, true, true, false |
| 14 | 2.1 | 7869 | 44.63 | 5.7 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned  |  |
| 15 | 1.7 | 7930 | 37.54 | 4.7 | void ck_tile::kentry<1, ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<ck_tile::Rmsnorm2dFwdPipelineProblem< |  |
| 16 | 1.6 | 7930 | 35.62 | 4.5 | void ck_tile::kentry<1, ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<ck_tile::Rmsnorm2dFwdPipelineProblem< |  |
| 17 | 1.6 | 7930 | 34.53 | 4.4 | void vllm::concat_and_cache_mla_kernel<__hip_bfloat16, __hip_bfloat16, (vllm::Fp8KVCacheDataType)0>(__hip_bfloat16 const |  |
| 18 | 1.6 | 7800 | 34.07 | 4.4 | triton_poi_fused_add_all_reduce_0 |  |
| 19 | 1.6 | 7800 | 33.81 | 4.3 | triton_poi_fused_add_clone_copy_expand_index_mul_neg_slice_split_stack_unsqueeze_view_2 |  |
| 20 | 1.5 | 7800 | 33.37 | 4.3 | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambd |  |
| 21 | 1.5 | 7800 | 33.20 | 4.3 | triton_poi_fused_add_clone_expand_index_mul_neg_slice_split_split_with_sizes_stack_unsqueeze_view_1 |  |
| 22 | 1.5 | 7869 | 33.06 | 4.2 | _batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_32_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SI |  |
| 23 | 1.5 | 7869 | 32.17 | 4.1 | _batched_gemm_a16wfp4_kernel_BLOCK_SIZE_M_8_BLOCK_SIZE_N_32_BLOCK_SIZE_K_512_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SI |  |
| 24 | 1.4 | 7740 | 31.26 | 4.0 | SiLU Activation (act_and_mul) | bf16 |
| 25 | 1.4 | 7800 | 30.79 | 3.9 | triton_poi_fused_mul_silu_slice_0 |  |
| 26 | 1.1 | 127 | 24.16 | 190.2 | Fill (zeros) | bf16 |
| 27 | 0.9 | 60 | 19.20 | 320.1 | void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout:: |  |
| 28 | 0.8 | 61 | 18.02 | 295.5 | aiter::fmha_fwd_hd192_hd128_bf16_causal_group |  |
| 29 | 0.6 | 61 | 13.86 | 227.2 | hipBLASLt GEMM | tile=160x256x64, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 30 | 0.6 | 63 | 13.67 | 217.0 | Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_NTD_SK3_UserArgs_MT256x256x64_MI16x16x1_shortname0_gfx950 |  |
| 31 | 0.5 | 126 | 11.35 | 90.1 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 4, 16, 8, 1, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 32 | 0.3 | 130 | 6.95 | 53.5 | void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float, at::native::(anonymous namespace |  |
| 33 | 0.3 | 130 | 6.37 | 49.0 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4,  |  |
| 34 | 0.3 | 60 | 5.96 | 99.3 | hipBLASLt GEMM | tile=128x256x64, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 35 | 0.2 | 60 | 4.93 | 82.1 | hipBLASLt GEMM | tile=128x256x32, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 36 | 0.2 | 61 | 4.32 | 70.8 | hipBLASLt GEMM | tile=256x192x64, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 37 | 0.2 | 60 | 3.71 | 61.9 | hipBLASLt GEMM | tile=128x96x128, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 38 | 0.1 | 120 | 3.24 | 27.0 | _moe_mxfp4_sort_kernel |  |
| 39 | 0.1 | 128 | 3.04 | 23.7 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 3, 16, 8, 2, 4>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 40 | 0.1 | 61 | 3.02 | 49.5 | hipBLASLt GEMM | tile=256x256x32, splitK=3, ISA=950, WS=64, WG=32_8_1 |
| 41 | 0.1 | 120 | 2.77 | 23.1 | void aiter::dynamic_per_group_scaled_quant_kernel<std::bfloat16_t, ck_tile::fp4x2_t, 32>(ck_tile::fp4x2_t*, float*, std: |  |
| 42 | 0.1 | 250 | 2.52 | 10.1 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_ker |  |
| 43 | 0.1 | 60 | 1.73 | 28.9 | void ck_tile::kentry<2, ck_tile::MoeSortingMultiPhaseKernel_P23<ck_tile::MoeSortingProblemMp<int, float, unsigned char,  |  |
| 44 | 0.1 | 247 | 1.64 | 6.7 | void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 1>(int, int, int, int, int, int, __hip_bfloat16 const*, __hip_bfl |  |
| 45 | 0.1 | 129 | 1.56 | 12.1 | void rocprim::ROCPRIM_400000_NS::detail::partition_kernel<(rocprim::ROCPRIM_400000_NS::detail::select_method)0, true, ro |  |
