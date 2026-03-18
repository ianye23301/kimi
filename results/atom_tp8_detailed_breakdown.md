# ATOM TP8 Detailed Kernel Breakdown — Kimi K2.5-MXFP4

**Model**: Kimi K2.5-MXFP4 (KimiK25ForConditionalGeneration → DeepseekV3ForCausalLM)
**Hardware**: 8x AMD MI355X (gfx950), 256 CUs/GPU, ~8 TB/s HBM BW
**Config**: TP=8, kv_cache_dtype=fp8, CUDAGraph level 3, batch=1 decode
**Total decode step**: **54.6 ms** (measured via rocprof, baseline)

---

## Source File Index

### Model Layer Definitions
- `kimi/ATOM/atom/models/deepseek_v2.py:1544` — DeepseekV2DecoderLayer (per-layer forward)
- `kimi/ATOM/atom/models/deepseek_v2.py:1219` — DeepseekV2MLAAttention
- `kimi/ATOM/atom/models/deepseek_v2.py:732` — DeepseekV2MoE
- `kimi/ATOM/atom/models/deepseek_v2.py:691` — DeepseekV2MLP (dense + shared expert)
- `kimi/ATOM/atom/models/deepseek_v2.py:1751` — DeepseekV2Model (full model)

### MLA Attention Kernels
- `kimi/ATOM/atom/model_ops/attention_mla.py:115` — MLAAttention (dispatch logic, BMMs, decode path selection)
- `aiter_local/aiter/mla.py:146` — mla_decode_fwd (ASM MLA decode entry)
- `aiter_local/csrc/cpp_itfs/mla/asm_mla_decode_fwd.py:97` — ASM kernel Python wrapper
- `aiter_local/aiter/ops/triton/attention/mla_decode_rope.py:144` — decode_attention_fwd_grouped_rope (Triton MLA decode entry)
- `aiter_local/aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py` — Triton kernel implementation (stage1 + stage2)
- `aiter_local/aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py:222` — batched_gemm_a16wfp4 (k-up, v-up BMMs)

### GEMM / Linear Layers
- `kimi/ATOM/atom/model_ops/linear.py:470` — ReplicatedLinear
- `kimi/ATOM/atom/model_ops/linear.py:496` — ColumnParallelLinear
- `kimi/ATOM/atom/model_ops/linear.py:722` — RowParallelLinear
- `kimi/ATOM/atom/model_ops/linear.py:526` — MergedColumnParallelLinear
- `kimi/ATOM/atom/model_ops/linear.py:767` — MergedReplicatedLinear
- `aiter_local/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py:406` — gemm_afp4wfp4_preshuffle (FP4 GEMM for fused_qkv_a)
- `aiter_local/aiter/tuned_gemm.py:292` — skinny_gemm (bf16 GEMV dispatch for o_proj, gate)
- `aiter_local/csrc/kernels/custom_kernels.cu:432` — wv_splitk_small_fp16_bf16_kernel (HIP GEMV implementation)

### Normalization + Quantization
- `kimi/ATOM/atom/model_ops/layernorm.py:172` — RMSNorm (PyTorch module, handles fused_allreduce flag)
- `aiter_local/aiter/ops/triton/quant/fused_mxfp4_quant.py:22` — fused_rms_mxfp4_quant
- `aiter_local/aiter/ops/triton/quant/fused_mxfp4_quant.py:151` — mxfp4_quant_shuffled (quant-only wrapper)
- `aiter_local/aiter/ops/triton/quant/fused_mxfp4_quant.py:403` — fused_reduce_rms_mxfp4_quant
- `kimi/ATOM/atom/models/deepseek_v2.py:204` — _fuse_rmsnorm_fp4_quant (residual+norm+quant)
- `kimi/ATOM/atom/models/deepseek_v2.py:401` — _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4 (fused GEMM+split+norm+quant)
- `kimi/ATOM/atom/models/deepseek_v2.py:630` — _fuse_qkv_a_proj_reduce_rmsnorm_quant (dispatcher)

### KV Cache + RoPE
- `aiter_local/aiter/ops/cache.py:121` — fused_qk_rope_concat_and_cache_mla
- `aiter_local/aiter/ops/cache.py:90` — concat_and_cache_mla
- `aiter_local/aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:36` — fused_fp4_bmm_rope_cat_and_cache_mla (blocked on AMD: fp8e4nv)
- `aiter_local/aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py:109` — Triton kernel impl

### MoE
- `kimi/ATOM/atom/model_ops/moe.py:1852` — FusedMoE (PyTorch wrapper)
- `aiter_local/aiter/fused_moe.py:1022` — fused_moe_2stages (orchestrator)
- `aiter_local/aiter/fused_moe.py:1654` — cktile_moe_stage1 (CK gate+up GEMM)
- `aiter_local/aiter/fused_moe.py:1720` — cktile_moe_stage2 (CK down GEMM)
- `aiter_local/aiter/ops/topk.py:29` — grouped_topk (expert routing)
- `aiter_local/aiter/ops/topk.py:16` — biased_grouped_topk_hip (HIP kernel)
- `aiter_local/aiter/ops/topk.py:226` — top_k_per_row_decode

### Activation
- `kimi/ATOM/atom/model_ops/activation.py:57` — SiluAndMul (PyTorch module)
- `aiter_local/aiter/ops/activation.py:11` — silu_and_mul (kernel)

### Communication (AllReduce)
- `aiter_local/aiter/dist/communication_op.py:30` — tensor_model_parallel_all_reduce
- `aiter_local/aiter/dist/device_communicators/custom_all_reduce.py:282` — CustomAllReduce.all_reduce
- `aiter_local/aiter/dist/device_communicators/communicator_cuda.py:150` — CommunicatorCuda.all_reduce
- `aiter_local/aiter/dist/device_communicators/communicator_cuda.py:193` — fused_allreduce_rmsnorm (HIP fused AR+RMSNorm)
- `aiter_local/aiter/dist/device_communicators/communicator_cuda.py:232` — fused_allreduce_rmsnorm_quant
- `aiter_local/csrc/include/custom_all_reduce.cuh` — HIP kernel: local_device_load_rmsnorm

---

## Model Config

- hidden_size: 7168
- num_layers: 61 (layer 0 = dense MLP, layers 1-60 = MoE)
- num_attention_heads: 64 (8 per GPU at TP8)
- q_lora_rank: 1536
- kv_lora_rank: 512
- qk_nope_head_dim: 128
- qk_rope_head_dim: 64
- v_head_dim: 128
- n_routed_experts: 384, top_k: 8
- n_shared_experts: 1
- moe_intermediate_size: 2048 (per expert)
- dense intermediate_size: 18432 (layer 0 only)
- first_k_dense_replace: 1
- routed_scaling_factor: 2.827
- quantization: MXFP4 block [128,128], e8m0 scales

---

## TP8 Weight Shapes (Per GPU)

### MLA Attention (all 61 layers)

| Weight | Type | Global Shape | Per-GPU Shape | Dtype | Notes |
|--------|------|-------------|---------------|-------|-------|
| fused_qkv_a_proj | MergedReplicatedLinear | [7168, 2112] | [7168, 2112] | MXFP4 | REPLICATED, output=[q_lora:1536, kv_lora:512, k_pe:64] |
| q_a_layernorm | RMSNorm | [1536] | [1536] | bf16 | |
| kv_a_layernorm | RMSNorm | [512] | [512] | bf16 | |
| q_b_proj | ColumnParallelLinear | [1536, 12288] | [1536, 1536] | MXFP4 | 12288 = 64 heads * 192 qk_head_dim. Per-GPU: 8 heads * 192 = 1536 |
| kv_b_proj | ColumnParallelLinear | [512, 16384] | [512, 2048] | MXFP4 | 16384 = 64 * (128+128). Per-GPU: 8 * 256 = 2048. Split: k_nope [512,1024], v [512,1024] |
| o_proj | RowParallelLinear | [8192, 7168] | [1024, 7168] | bf16 | 8192 = 64*128. Per-GPU input: 8*128=1024. Output AllReduced |

### Dense MLP (layer 0 only)

| Weight | Type | Global Shape | Per-GPU Shape | Dtype |
|--------|------|-------------|---------------|-------|
| gate_up_proj | MergedColumnParallelLinear | [7168, 36864] | [7168, 4608] | MXFP4 | 36864=18432*2 |
| down_proj | RowParallelLinear | [18432, 7168] | [2304, 7168] | MXFP4 | AllReduce output |

### MoE (layers 1-60, 60 layers)

| Weight | Type | Global Shape | Per-GPU Shape | Dtype |
|--------|------|-------------|---------------|-------|
| gate | ReplicatedLinear | [7168, 384] | [7168, 384] | bf16 | REPLICATED, unquantized |
| e_score_correction_bias | Parameter | [384] | [384] | fp32 | REPLICATED (cast to bf16 each step in baseline) |
| experts.w13 (gate+up fused) | FusedMoE | [384, 4096, 7168] | [384, 512, 7168] | MXFP4 | 4096=2048*2. Per-GPU: 512=256*2 |
| experts.w2 (down) | FusedMoE | [384, 7168, 2048] | [384, 7168, 256] | MXFP4 | Per-GPU: 256=2048/8 |
| shared_experts.gate_up_proj | MergedColumnParallelLinear | [7168, 4096] | [7168, 512] | MXFP4 | 4096=2048*2 |
| shared_experts.down_proj | RowParallelLinear | [2048, 7168] | [256, 7168] | MXFP4 | AllReduce output |

### Layer Norms (per layer)

| Weight | Shape | Dtype |
|--------|-------|-------|
| input_layernorm | [7168] | bf16 |
| post_attention_layernorm | [7168] | bf16 |

---

## Baseline Single Decode Step Timeline (per layer)

For batch=1, seqlen=1, context_len=S (varies).
All tensor dims shown as [batch, ...] with batch=1.

In baseline ATOM, `fuse_ar_input_norm` and `fuse_input_norm_quant` are **mutually exclusive** — when MXFP4 quant fusion is enabled (which it is for this model), AR fusion into input_layernorm is disabled. This means MoE output goes through a **standalone AllReduce**, then input_layernorm does fused Residual+RMSNorm+MXFP4Quant as one Triton kernel.

### Phase 1: Input LayerNorm + MXFP4 Quantization

**Dispatch**: `deepseek_v2.py:1648` (DeepseekV2DecoderLayer.forward)

**Layers 1-60 (receives MoE output from previous layer — standalone AllReduce already happened in Phase 4e):**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 1 | Residual Add + RMSNorm + MXFP4 Quant | hidden=[1,7168] bf16, residual=[1,7168] bf16 | data=[1,3584] uint8, scale=[1,56] uint8, residual=[1,7168] bf16 | Triton Fused Add+RMSNorm+Quant | `deepseek_v2.py:204` (_fuse_rmsnorm_fp4_quant) → `fused_mxfp4_quant.py:22` | 4.3 |

**Layer 0 (first layer, no previous AllReduce):**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 1 | Residual Add + RMSNorm + MXFP4 Quant | hidden=[1,7168] bf16, residual=[1,7168] bf16 | data=[1,3584] uint8, scale=[1,56] uint8, residual=[1,7168] bf16 | Triton Fused Add+RMSNorm+Quant | `deepseek_v2.py:204` → `fused_mxfp4_quant.py:22` | 4.3 |

### Phase 2: MLA Attention

**Dispatch**: `deepseek_v2.py:1450` (DeepseekV2MLAAttention.forward) → `attention_mla.py:115` (MLAAttention)

#### 2a. Fused QKV-A Projection

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2a | fused_qkv_a_proj GEMM | [1,7168] MXFP4 (packed [1,3584]+scale) | [1,2112] bf16 | Triton gemm_afp4wfp4_preshuffle | `deepseek_v2.py:1460` → `gemm_afp4wfp4.py:406` | ~6.5 |
| | *Weight*: [7168,2112] MXFP4 (REPLICATED) | | Split: q_c=[1,1536], kv_c=[1,512], k_pe=[1,64] | | |

Note: When fuse_qknorm_quant is enabled, this fuses with the RMSNorm+quant below into a single Triton kernel.
Fused dispatch: `deepseek_v2.py:630` → `deepseek_v2.py:401` (_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4)

#### 2b. Q/KV LayerNorm + Quant

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2b-1 | q_a RMSNorm + MXFP4 quant | q_c=[1,1536] bf16 | q_c_quant=[1,768] uint8, scale | Fused Add+RMSNorm+Quant (hidden=64) | `deepseek_v2.py:283` (_fuse_rmsnorm_quant) | 4.3 |
| 2b-2 | kv_a RMSNorm | kv_c=[1,512] bf16 | kv_c_normed=[1,512] bf16 | Fused Add+RMSNorm+Quant (hidden=256) | same as above (dual-input path) | 4.2 |

#### 2c. Q-B Projection (generates Q heads)

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2c | q_b_proj GEMM | [1,1536] MXFP4 | [1,1536] bf16 | hipBLASLt GEMM (tile=32x16x128, splitK=3) | `linear.py:496` (ColumnParallelLinear) → `tuned_gemm.py:433` (TunedGemm) | 6.5 |
| | *Weight*: [1536,1536] MXFP4 per-GPU | | Reshape to [1, 8, 192] = 8 heads * (128 nope + 64 rope) | | |

#### 2d. K-up-proj BMM (absorbed into latent space)

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2d | Batched GEMM: q_nope @ W_K | q_nope=[8,1,128] bf16, W_K=[8,128,512] MXFP4 | k_nope_out=[8,1,512] bf16 | MLA Batched GEMM a8w8 (M=16,N=32,K=128) | `attention_mla.py:115` (_q_proj_and_k_up_proj) → `batched_gemm_a16wfp4.py:222` | 5.5 |

Note: "M=16" because the ASM kernel pads 8 heads to 16. The GEMM is actually [1,128]x[128,512] batched 8 times but dispatched as a single batched kernel with padded batch dim.

#### 2e. RoPE + KV Cache Store

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2e | Fused RoPE + concat + cache write | q_rope=[8,1,64], k_nope=[1,512], k_pe=[1,64], positions=[1] | q_out=[1,16,576] fp8 (padded to 16 heads) | Triton fused_qk_rope_concat_and_cache_mla | `cache.py:121` | 4.0 |
| | | | Side effect: kv_cache[slot] = cat(kv_c_normed, k_pe_rotated) = [576] fp8 | | |

Note: q_out is fp8 for ASM decode, bf16 for Triton decode. 576 = kv_lora_rank(512) + qk_rope_head_dim(64).

#### 2f. float8_copy (head repeat for ASM kernel)

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2f | q.repeat(1, 2, 1) | [1,8,576] fp8 | [1,16,576] fp8 | float8_copy_kernel | `attention_mla.py:653` (torch.Tensor.repeat) | 4.6 |

This is needed because the ASM MLA kernel requires >= 16 heads, but TP8 gives only 8. In baseline, uses `repeat_interleave` which produces [h0,h0,h1,h1,...] layout.

#### 2g. MLA Decode (Attention)

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2g-1 | MLA Decode Stage 1 | q=[1,16,576] fp8, kv_cache=[S,576] fp8 | attn_out=[1,16,512] bf16 | AITER ASM mla_decode_fwd (a8w8, qh=16, gqa=16) | `attention_mla.py:628` → `mla.py:146` → `asm_mla_decode_fwd.py:97` | 16.3 |
| 2g-2 | MLA Reduce | attn_out=[1,16,512] bf16 | reduced=[1,16,512] bf16 | AITER ASM mla_reduce (block=512, heads=16, splits=1) | `mla.py:21` (_fwd_kernel_stage2_asm) | 8.0 |

Note: S = context length (variable). Time shown is for a moderate S (~4K). Stage 1 computes Q@K softmax, stage 2 applies to V (V = kv_cache[:,:512]).

#### 2h. V-up-proj BMM + O-proj

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 2h-1 | Slice to real heads | [1,16,512] bf16 | [1,8,512] bf16 (first 8 of repeated 16) | No kernel (view/slice) | `attention_mla.py` (_v_up_proj_and_o_proj) | 0 |
| 2h-2 | V-up-proj Batched GEMM | attn_out=[8,1,512] bf16, W_V=[8,512,128] MXFP4 | v_out=[8,1,128] bf16 | MLA Batched GEMM a8w8 (M=16,N=128,K=128) | `attention_mla.py` → `batched_gemm_a16wfp4.py:222` | 4.1 |
| 2h-3 | Reshape + flatten | [8,1,128] → [1,8,128] → [1,1024] | | No kernel (view) | | 0 |
| 2h-4 | o_proj GEMM | [1,1024] bf16, W_o=[1024,7168] bf16 | [1,7168] bf16 | AITER bf16 GEMV 32x64 splitk | `linear.py:722` (RowParallelLinear) → `tuned_gemm.py:292` (skinny_gemm) → `custom_kernels.cu:432` | 14.1 |
| 2h-5 | AllReduce (o_proj) | [1,7168] bf16 | [1,7168] bf16 | AllReduce reduce_scatter | `linear.py:722` (reduce_results) → `communication_op.py:30` → `custom_all_reduce.py:282` | 26.9 |

Note: o_proj is bf16 (not MXFP4) because it's a RowParallelLinear with base_quant_config=None for FP4 models.

### Phase 3: Post-Attention LayerNorm

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 3 | Fused AllReduce + Residual Add + RMSNorm | attn_out=[1,7168] bf16, residual=[1,7168] bf16 | normed=[1,7168] bf16, residual=[1,7168] bf16 | Fused Load+RMSNorm+AllReduce (HIP) | `deepseek_v2.py:1648` → `layernorm.py:172` → `communicator_cuda.py:193` → `custom_all_reduce.cuh` | 4.0 |

Note: post_attention_layernorm has fused_allreduce=True in baseline (attention output AR is fused here). This is NOT the same mutual exclusion — the mutual exclusion only applies to `input_layernorm` (MXFP4 quant path).

### Phase 4: MoE (layers 1-60) / Dense MLP (layer 0)

#### Layer 0: Dense MLP

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 4-dense-1 | gate_up_proj GEMM | [1,7168] bf16, W=[7168,4608] MXFP4 | [1,4608] bf16 | hipBLASLt GEMM (tile=256x16x128) | `deepseek_v2.py:691` (DeepseekV2MLP) → `linear.py:526` (MergedColumnParallelLinear) → `tuned_gemm.py:433` | 53.6 |
| 4-dense-2 | SiLU + Mul | [1,4608] bf16 (split: gate=[1,2304], up=[1,2304]) | [1,2304] bf16 | SiLU Activation | `deepseek_v2.py:691` → `activation.py:57` → `aiter/ops/activation.py:11` | 4.0 |
| 4-dense-3 | down_proj GEMM | [1,2304] bf16, W=[2304,7168] MXFP4 | [1,7168] bf16 | hipBLASLt GEMM | `deepseek_v2.py:691` → `linear.py:722` (RowParallelLinear) → `tuned_gemm.py:433` | ~8.0 |
| 4-dense-4 | AllReduce | [1,7168] bf16 | [1,7168] bf16 | AllReduce reduce_scatter | `linear.py:722` (reduce_results) → `communication_op.py:30` → `custom_all_reduce.py:282` | 26.9 |

#### Layers 1-60: MoE (60 layers)

**4a. Shared Expert (parallel with routed, on alt_stream if mori enabled):**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 4a-1 | shared gate_up_proj | [1,7168] bf16, W=[7168,512] MXFP4 | [1,512] bf16 | hipBLASLt GEMM (tile=32x16x128) | `deepseek_v2.py:691` (DeepseekV2MLP) → `linear.py:526` (MergedColumnParallelLinear) → `tuned_gemm.py:433` | 6.5 |
| 4a-2 | shared SiLU+Mul | [1,512] bf16 (gate=[1,256], up=[1,256]) | [1,256] bf16 | SiLU Activation | `deepseek_v2.py:691` → `activation.py:57` → `aiter/ops/activation.py:11` | 4.0 |
| 4a-3 | shared down_proj | [1,256] bf16, W=[256,7168] MXFP4 | [1,7168] bf16 | AITER bf16 GEMV 32x64 splitk | `deepseek_v2.py:691` → `linear.py:722` (RowParallelLinear) → `tuned_gemm.py:292` (skinny_gemm) → `custom_kernels.cu:432` | 14.1 |

Note: Shared expert per-GPU intermediate = 2048/8 = 256.

**4b. Gate + Routing:**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 4b-1 | Gate projection | [1,7168] bf16, W=[7168,384] bf16 | [1,384] bf16 | AITER bf16 GEMV (replicated, no reduce) | `deepseek_v2.py:732` (DeepseekV2MoE) → `linear.py:470` (ReplicatedLinear) → `tuned_gemm.py:292` (skinny_gemm) → `custom_kernels.cu:432` | ~14.1 |
| 4b-2 | Grouped TopK (+ bias + sigmoid + softmax) | logits=[1,384] bf16, bias=[384] fp32 | weights=[1,8] bf16, ids=[1,8] int32 | MoE Grouped TopK (HIP) | `deepseek_v2.py:732` → `topk.py:29` (grouped_topk) → `topk.py:16` (biased_grouped_topk_hip) | 7.4 |
| 4b-3 | MoE Sorting | weights=[1,8], ids=[1,8] | sorted permutation indices | MoE Sorting (topk route) | `deepseek_v2.py:732` → `moe.py:1852` (FusedMoE) → `fused_moe.py:1022` (fused_moe_2stages) | 9.7 |

Note: In baseline, `e_score_correction_bias` is fp32 — causes a bfloat16_copy cast kernel each call (240 calls/step).

**4c. Routed Expert GEMMs:**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 4c-1 | Fill zeros (stage1 output buffer) | | [8,512] bf16 | Fill (zeros) | `fused_moe.py:1654` (cktile_moe_stage1) — torch.zeros alloc | 4.0 |
| 4c-2 | MoE Stage 1: gate+up GEMM | sorted_input=[8,7168] MXFP4, W13=[selected_experts, 512, 7168] MXFP4 | [8,512] bf16 | CK cktile_moe_stage1 (tile=16x128x256, MXFP4->bf16) | `moe.py:1852` → `fused_moe.py:1022` → `fused_moe.py:1654` | 10.5 |
| 4c-3 | SiLU + Mul | [8,512] bf16 (gate=[8,256], up=[8,256]) | [8,256] bf16 | SiLU Activation | `fused_moe.py:1022` → `activation.py:57` → `aiter/ops/activation.py:11` | 4.0 |
| 4c-4 | MoE Stage 2: down GEMM | [8,256] bf16, W2=[selected_experts, 7168, 256] MXFP4 | [8,7168] bf16 | CK cktile_moe_stage2 (tile=16x128x256) | `fused_moe.py:1022` → `fused_moe.py:1720` | 6.9 |

Note: input is duplicated/gathered per expert. For batch=1 with top-8, we have 8 expert-token pairs, each a [1,7168] → [1,512] → [1,256] → [1,7168] GEMM chain. The "8" batch is from 8 experts processing the same token. Per-GPU intermediate = 2048/8 = 256, but gate+up is fused to 512.

**4d. Weighted Sum + Combine:**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 4d-1 | Weighted sum (unsort + scale) | expert_out=[8,7168] bf16, weights=[1,8] bf16 | [1,7168] bf16 | MoE Weighted Sum (wv_splitk) | `fused_moe.py:1022` (fused_moe_2stages) → `custom_kernels.cu:432` (wv_splitk) | 8.1 |
| 4d-2 | Add shared expert output | routed=[1,7168] + shared=[1,7168] | [1,7168] bf16 | elementwise add | `deepseek_v2.py:732` (DeepseekV2MoE.forward) — torch.add | ~1.0 |

**4e. MoE AllReduce (standalone in baseline):**

| Step | Operation | Input Shape/Dtype | Output Shape/Dtype | Kernel | Source | Time (us) |
|------|-----------|-------------------|--------------------|---------|----|---|
| 4e | AllReduce | [1,7168] bf16 | [1,7168] bf16 | AllReduce reduce_scatter | `linear.py:722` (reduce_results) → `communication_op.py:30` → `custom_all_reduce.py:282` | 26.9 |

Note: In baseline, this is a **standalone** AllReduce — NOT fused into the next layer's input_layernorm. This is because `fuse_ar_input_norm` is forced to False when `fuse_input_norm_quant` is True (mutual exclusion in `deepseek_v2.py:1577-1583`).

---

## Per-Layer Time Budget (Baseline, MoE layer, batch=1)

### MLA Attention Phase: ~78 us/layer

| Sub-step | Kernel | us |
|-----------|--------|----|
| Fused QKV-A GEMM | gemm_afp4wfp4 | ~6.5 |
| Q/KV RMSNorm + Quant | Fused Add+RMSNorm+Quant (x2) | 8.5 |
| Q-B proj GEMM | hipBLASLt | 6.5 |
| K-up BMM | Batched GEMM a8w8 | 5.5 |
| RoPE + KV store | Triton fused | 4.0 |
| float8_copy (head repeat) | elementwise | 4.6 |
| MLA Decode stage1 | ASM a8w8 | 16.3 |
| MLA Reduce stage2 | ASM | 8.0 |
| V-up BMM | Batched GEMM a8w8 | 4.1 |
| o_proj GEMV | AITER bf16 GEMV | 14.1 |
| **Subtotal attention compute** | | **78.1** |

### MoE Phase: ~57 us/layer (routed only, excluding shared)

| Sub-step | Kernel | us |
|-----------|--------|----|
| Gate proj | GEMV | ~6.5 |
| Grouped TopK | HIP | 7.4 |
| Sorting | HIP | 9.7 |
| Fill zeros | | 4.0 |
| Stage1 gate+up GEMM | CK Flatmm | 10.5 |
| SiLU + Mul | Triton | 4.0 |
| Stage2 down GEMM | CK Flatmm | 6.9 |
| Weighted sum | wv_splitk | 8.1 |
| **Subtotal MoE routed** | | **57.1** |

### Shared Expert: ~25 us/layer

| Sub-step | Kernel | us |
|-----------|--------|----|
| gate_up_proj | hipBLASLt | 6.5 |
| SiLU+Mul | Triton | 4.0 |
| down_proj | AITER GEMV | 14.1 |
| **Subtotal shared** | | **24.6** |

### Communication + Norms (Baseline): ~71 us/layer

| Sub-step | Kernel | us |
|-----------|--------|----|
| AllReduce (o_proj output) | reduce_scatter | 26.9 |
| AllReduce (MoE output) — **standalone** | reduce_scatter | 26.9 |
| Input LN + MXFP4 quant (fused, no AR) | Triton Fused Add+RMSNorm+Quant | 4.3 |
| Post-attn LN (fused with attn AR) | HIP Fused Load+RMSNorm+AllReduce | 4.0 |
| bfloat16_copy (correction_bias cast) | elementwise | ~1.0 |
| **Subtotal comm+norms** | | **63.1** |

Note: The o_proj AllReduce is fused into post_attention_layernorm (Phase 3), so its time is already counted in Phase 3's 4.0us. The standalone MoE AllReduce (26.9us) is the one that hits the timeline as extra overhead in baseline.

### Total per MoE layer (baseline): ~223 us

Predicted total for 61 layers: 60 * 223 + ~300 (layer 0 dense) + ~500 (final norm + LM head) = **13,380 + 800 = ~14.2 ms**

vs measured 54.6 ms — the discrepancy is because:
1. Profile kernel times include GPU idle/stall between kernels
2. AllReduce has synchronization overhead not captured in kernel duration alone
3. CUDAGraph dispatch overhead
4. Some kernels appear more often than expected (480 wv_splitk vs 240 expected)

---

## Kernel Call Count Analysis

Expected per decode step (61 layers):

| Kernel | Expected Calls | Actual Calls | Notes |
|--------|----------------|--------------|-------|
| AllReduce | 122 (2 per layer) | 488 | 4x — includes multi-stage reduce_scatter |
| MLA Decode | 61 | 244 | 4x — stage1 + reduce + metadata = 4 calls per decode |
| wv_splitk | 60 (MoE layers) | 480 | 8x — one per top-k expert? Or shared expert uses it too |
| AITER bf16 GEMV | 61 (o_proj) | 244 | 4x — likely 2 GEMV calls per o_proj + shared expert |
| hipBLASLt GEMM | 122 (q_b + shared) | 484 | ~4x per layer — multiple matmul passes |
| MoE CK Stage1 | 60 | 240 | 4x per layer |
| MoE Sorting | 60 | 240 | 4x per layer |
| SiLU | 120 (routed+shared) | 480 | 4x per layer |
| Grouped TopK | 60 | 240 | 4x per layer |
| MoE CK Stage2 | 60 | 240 | 4x per layer |
| Fused RMSNorm+AR | 122 | 488 | 4x per layer |
| Fused RMSNorm+Quant | 61 | 244 | 4x per layer |
| MLA Batched GEMM | 122 (k-up + v-up) | 488 | 4x per layer |
| float8_copy | 61 | 244 | 4x per layer |
| direct_copy | 61 | 244 | 4x per layer |
| RoPE+KV store | 61 | 244 | 4x per layer |
| Fill zeros | 60 | 240 | 4x per layer |
| bfloat16_copy (bias cast) | 60 | 240 | baseline only — correction_bias fp32→bf16 |

The 4x multiplier is from CUDAGraph replay — each captured graph region replays all kernels. This is expected and does not add overhead (the replay is essentially a single launch).

---

## Kernel Time by Category (from profile, baseline)

| Category | Time (ms) | % | Calls | Key Kernels |
|----------|----------|---|-------|-------------|
| Communication | 15.68 | 28.7% | 488+240 | AllReduce reduce_scatter (13.13), Fused AR+RMSNorm (2.22), Triton Fused Add+AR+RMSNorm (0.98) |
| MoE Compute | 12.13 | 22.2% | 240*4 | CK Stage1 (2.52), CK Stage2 (1.65), Sorting (2.32), TopK (1.77), wv_splitk (3.88) |
| Dense GEMM | 8.94 | 16.4% | 244+484 | AITER GEMV (3.45), hipBLASLt (3.13+1.97) |
| MLA Attention | 8.38 | 15.4% | 244*3 | MLA Decode (3.99), MLA Reduce (1.94), Batched GEMM x2 (1.33+1.00) |
| Norm/Activation | 5.02 | 9.2% | 244+480 | Fused RMSNorm+Quant x2 (1.04+1.03), SiLU (1.93), RoPE+KV (0.97) |
| Copy/Misc | 2.21 | 4.1% | 244*2 | float8_copy (1.13), direct_copy (1.06) |
| Other | 2.19 | 4.0% | ~20 | Sampling, prefill GEMMs, NCCL, MLA metadata |

---

## Assignable Work Packages

### Package A: AllReduce / Communication (15.68 ms, 28.7%)
- 488 calls to reduce_scatter at 26.9 us avg
- 488 calls to Fused Load+RMSNorm+AllReduce at 4.6 us avg
- 240 calls to Triton Fused Add+AllReduce+RMSNorm at 4.0 us avg
- Shapes: [1, 7168] bf16 across 8 GPUs
- These are RCCL-backed, kernel time dominated by inter-GPU latency
- Source: `custom_all_reduce.py:282`, `communicator_cuda.py:193`, `custom_all_reduce.cuh`

### Package B: MoE GEMMs (4.17 ms, 7.6%) — OFF LIMITS (separate competition)
- CK Stage1 gate+up: 240 calls, 10.5 us avg, tile=16x128x256, MXFP4
- CK Stage2 down: 240 calls, 6.9 us avg, same tile
- Shape: [1,7168] x [7168,512] (stage1), [1,256] x [256,7168] (stage2), per-GPU
- Source: `fused_moe.py:1654`, `fused_moe.py:1720`

### Package C: MoE Overhead (8.0 ms, 14.6%)
- Sorting: 240 calls, 9.7 us avg
- Grouped TopK: 240 calls, 7.4 us avg
- wv_splitk weighted sum: 480 calls, 8.1 us avg — **unexpectedly 480 not 240**
- Fill zeros: 240 calls, 4.0 us avg
- Shapes: logits [1,384], ids [1,8], expert_out [8,7168]
- Source: `fused_moe.py:1022`, `topk.py:29`, `custom_kernels.cu:432`

### Package D: MLA Decode Attention (5.93 ms, 10.9%)
- MLA Decode stage1: 244 calls, 16.3 us avg, q=[1,16,576] fp8, kv=[S,576] fp8
- MLA Reduce: 244 calls, 8.0 us avg
- Note: S (context length) determines memory bandwidth. At S=8000, reading 8000*576*8 GPUs worth of KV.
- Source: `mla.py:146`, `asm_mla_decode_fwd.py:97`

### Package E: MLA BMMs (2.33 ms, 4.3%)
- K-up BMM: 244 calls, 5.5 us avg, [8,1,128]x[8,128,512] MXFP4, batched
- V-up BMM: 244 calls, 4.1 us avg, [8,1,512]x[8,512,128] MXFP4, batched
- Currently using batched_gemm_a8w8 dispatch — may benefit from custom kernel
- Source: `attention_mla.py:115`, `batched_gemm_a16wfp4.py:222`

### Package F: Dense GEMVs (3.45 ms, 6.3%)
- o_proj: 244 calls, 14.1 us avg, [1,1024]x[1024,7168] bf16
- shared expert down_proj: included in same kernel type
- gate proj: [1,7168]x[7168,384] bf16, replicated
- Uses AITER wv_splitk_small_fp16_bf16_kernel (skinny GEMV for M=1)
- Source: `tuned_gemm.py:292`, `custom_kernels.cu:432`

### Package G: hipBLASLt GEMMs (5.10 ms, 9.3%)
- q_b_proj: 244 calls, 6.5 us avg, [1,1536]x[1536,1536] MXFP4
- shared expert gate_up: 240 calls, 6.5 us avg, [1,7168]x[7168,512] MXFP4
- Additional: 244 calls, 8.1 us avg (tile=16x16x256)
- These are dispatch-limited at batch=1 — compute is minimal
- Source: `tuned_gemm.py:433`

### Package H: Copy Kernels (2.19 ms, 4.0%)
- float8_copy: 244 calls, 4.6 us avg — from q.repeat(1,2,1) for head padding fp8
- direct_copy: 244 calls, 4.3 us avg — source unclear (possibly dtype conversion or tensor contiguity)
- Eliminating head repeat would remove float8_copy entirely
- Source: `attention_mla.py:653` (repeat), unknown (direct_copy)

### Package I: Norm/Quant Fusion (3.01 ms, 5.5%)
- Fused RMSNorm+Quant (hidden=64): 244 calls, 4.3 us avg — q_a_layernorm + quant
- Fused RMSNorm+Quant (hidden=256): 244 calls, 4.2 us avg — kv_a_layernorm
- RoPE+KV store: 244 calls, 4.0 us avg
- SiLU: 480 calls, 4.0 us avg — routed (240) + shared (240)
- Source: `fused_mxfp4_quant.py:22`, `cache.py:121`, `activation.py:11`

---

## Mystery: 480 wv_splitk Calls

Expected: 240 (1 per MoE layer × 60 layers × 4 CUDAGraph replay = 240).
Actual: 480 — double.

Possible explanations:
- The wv_splitk kernel is also used for shared expert down_proj (skinny GEMV dispatch)
- Two wv_splitk variants in profile (rows 3 and... need to check if 480 is two groups of 240)
- Could be stage1+stage2 each doing a splitk reduction

This needs GPU profiling with kernel-to-layer correlation to resolve.

---

## Performance Targets vs Current (Baseline)

| Metric | conc=4 | Target | Gap |
|--------|--------|--------|-----|
| Interactivity (tok/s/user) | 72 | 150 | 2.1x |
| TPOT (ms) | 14.0 | 6.67 | 2.1x |
| Throughput/GPU (tok/s) | 34 | 1350 | 40x |

The throughput/GPU gap is because at batch=1 decode, each GPU produces tokens slowly. The 1350 target assumes batched decode with many concurrent users sharing GPU compute.

---

## Our (Ian's) Optimizations (Completed)

Work done so far on this model. All items tested, but **measured E2E impact is negligible (<1%)** under CUDAGraph — profiler-estimated savings don't translate to wall-clock gains due to CUDAGraph absorbing launch overhead.

### 1. AR Fusion into Input LayerNorm (code complete, tested)

**What**: Removed mutual exclusion between `fuse_ar_input_norm` and `fuse_input_norm_quant` in `deepseek_v2.py:1577-1583`. MoE output AllReduce is now deferred into next layer's input_layernorm.

**Baseline flow** (3 kernels):
- Standalone AllReduce (reduce_scatter) → 26.9 us
- Triton Fused Residual+RMSNorm+MXFP4Quant → 4.3 us

**Optimized flow** (2 kernels):
- HIP Fused AR+Residual+RMSNorm (`communicator_cuda.py:193`) → 4.6 us
- Triton MXFP4 quant-only (`fused_mxfp4_quant.py:151`, SKIP_RMSNORM flag) → 4.3 us

**Estimated savings**: ~6.5ms/step (244 standalone AllReduce calls eliminated)
**Measured savings**: <0.15ms (<1%) — CUDAGraph captures make standalone AR nearly free

**Files modified**:
- `kimi/ATOM/atom/models/deepseek_v2.py` — removed mutual exclusion, added 2-kernel forward branch
- `aiter_local/aiter/ops/triton/quant/fused_mxfp4_quant.py` — added SKIP_RMSNORM flag to Triton kernel, `mxfp4_quant_shuffled()` wrapper

### 2. correction_bias bf16 Pre-conversion (code complete, tested)

**What**: Added `process_weights_after_loading` to `DeepseekV2MoE` to convert `e_score_correction_bias` from fp32 to bf16 at load time.

**Baseline**: fp32 bias → bf16 cast every forward call (240 bfloat16_copy kernel launches/step)
**Optimized**: bf16 bias stored at init → no cast needed

**Estimated savings**: ~1ms/step (240 × ~4us)
**Measured savings**: too small to measure in E2E

**Files modified**:
- `kimi/ATOM/atom/models/deepseek_v2.py` — added `process_weights_after_loading` to DeepseekV2MoE

### 3. MLA Head-Repeat Layout Fix (code complete, tested)

**What**: Changed `repeat_interleave` to `repeat` for head padding 8→16 heads. `repeat_interleave` produces [h0,h0,h1,h1,...] which can't be sliced back to unique heads with [:8]. `repeat` produces [h0,...,h7,h0,...,h7] so [:8] gives unique heads.

**Also**: heads-first output layout — allocate (padded_N,B,L), pass to kernel, slice [:num_heads] contiguous without extra transpose.

**Net savings**: ~0.5ms/step (repeat is 3-4μs slower than repeat_interleave per call, but saves ~6μs/call from eliminating .contiguous())

**Files modified**:
- `kimi/ATOM/atom/model_ops/attention_mla.py` — repeat layout, heads_first flag

### 4. Triton MLA Decode for Small Batch (code complete, tested, REVERTED)

**What**: Hybrid dispatch — Triton MLA decode for bs≤32 (native 8 heads, bf16 Q, no head repeat), ASM for bs>32.

**Result**: HARMFUL with fp8 KV cache. The fp8→bf16 cast overhead in the Triton kernel's inner attention loop adds ~3ms/token. ASM handles fp8 natively.

**Measured**: +3ms regression at conc=4 (17.25ms TPOT vs 14.06ms baseline)

**Status**: Disabled (ATOM_MLA_TRITON_DECODE=0). Code exists but should not be enabled with fp8 KV.

### 5. Fused kv_bmm (code complete, BLOCKED)

**What**: Single Triton kernel for FP4 BMM + RoPE + KV cache store, replacing separate _q_proj_and_k_up_proj + fused_qk_rope_concat_and_cache_mla.

**Blocked**: aiter's `fused_fp4_bmm_rope_cat_and_cache_mla` uses fp8e4nv (NVIDIA format) which doesn't work on AMD.

### 6. MXFP4 KV Cache for MLA Decode (investigated, NOT WORTH IT)

**What**: Replace fp8 KV cache with MXFP4 for 47% less memory bandwidth.

**Result**: Triton MXFP4 kernel is 1.5x SLOWER than fp8 — compute-bound on V dequant. `dot_scaled` for P@V blocked by scale layout mismatch. Theoretical max savings from MXFP4 MLA decode: ~2ms/step (3.7%).

**Status**: Dead end. Only an ASM/HIP kernel could potentially achieve bandwidth savings.

### E2E Benchmark Results (ISL=8000, OSL=1024)

**conc=4 (40 prompts, gpu_mem_util=0.35, TP8):**

| Config | Median TPOT | Output tok/s | Interactivity |
|--------|------------|-------------|---------------|
| Baseline | 14.20ms | 271.75 | 70.4 |
| AR fusion + bias fix | 14.06ms | 272.93 | 71.1 |
| + Triton MLA decode | 17.25ms | 224.45 | 57.1 |

**conc=32 (320 prompts, baseline):**
- TPOT 27.86ms, throughput 1099 tok/s, interactivity 35.9
- Severely KV cache limited at 0.35 util (shared GPUs)

**Key takeaway**: Framework-level optimizations have hit diminishing returns under CUDAGraph. Remaining gains require kernel-level changes to the compute-heavy kernels themselves (MLA, MoE, GEMV).
