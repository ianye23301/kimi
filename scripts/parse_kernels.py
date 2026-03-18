import gzip, json, sys, re
from collections import defaultdict

trace_path = sys.argv[1]

with gzip.open(trace_path) as f:
    trace = json.load(f)

kernels = defaultdict(lambda: {"count": 0, "total_us": 0})
for ev in trace.get("traceEvents", []):
    if ev.get("cat") == "kernel" and ev.get("ph") == "X":
        name = ev.get("name", "unknown")
        dur = ev.get("dur", 0)
        kernels[name]["count"] += 1
        kernels[name]["total_us"] += dur

sorted_k = sorted(kernels.items(), key=lambda x: -x[1]["total_us"])
total = sum(v["total_us"] for v in kernels.values())


def classify(name):
    if "MoeFlatmmKernel" in name:
        kind_m = re.search(r"MoeFlatmmKind\)(\d)", name)
        kind = kind_m.group(1) if kind_m else "?"
        tile_m = re.search(r"sequence<(\d+, \d+, \d+)>", name)
        tile = tile_m.group(1) if tile_m else "?"
        stage = "gate+up (stage1)" if kind == "3" else "down (stage2)" if kind == "2" else f"kind={kind}"
        return f"MoE CK Flatmm {stage}", f"tile=<{tile}>, MXFP4->bf16, scale=e8m0/group32"
    if "MoeSortingKernel" in name:
        return "MoE Sorting (topk route)", "int/float, sorted"
    if "wv_splitk" in name:
        tpl = re.search(r"<([^>]+)>", name)
        return "MoE Weighted Sum (wv_splitk)", tpl.group(1) if tpl else ""
    if "grouped_topk" in name:
        m = re.search(r"<([^>]+)>", name)
        return "MoE Grouped TopK", m.group(1) if m else ""
    if name.startswith("Cijk_"):
        mt = re.search(r"MT(\d+x\d+x\d+)", name)
        tile = mt.group(1) if mt else "?"
        sk = re.search(r"SK(\d+)", name)
        splitk = sk.group(1) if sk else "?"
        isa = re.search(r"ISA(\d+)", name)
        isa_val = isa.group(1) if isa else "?"
        ws = re.search(r"_WS(\d+)_", name)
        ws_val = ws.group(1) if ws else "?"
        wg = re.search(r"WG(\d+_\d+_\d+)", name)
        wg_val = wg.group(1) if wg else "?"
        post = re.search(r"PostGSU(\d+)", name)
        if post:
            return f"hipBLASLt GEMM (PostGSU{post.group(1)})", f"tile={tile}"
        return "hipBLASLt GEMM", f"tile={tile}, splitK={splitk}, ISA={isa_val}, WS={ws_val}, WG={wg_val}"
    if "bf16gemm_fp32bf16_tn" in name:
        m = re.search(r"tn_(\d+x\d+)", name)
        tile = m.group(1) if m else "?"
        splitk = "splitk" if "splitk" in name else ""
        return f"AITER bf16 GEMV {tile} {splitk}".strip(), "bf16->fp32->bf16"
    if "mla_a8w8" in name:
        m = re.search(r"qh(\d+)_qseqlen(\d+)_gqaratio(\d+)", name)
        if m:
            return "MLA Decode (a8w8)", f"qheads={m.group(1)}, seqlen={m.group(2)}, gqa_ratio={m.group(3)}"
        return "MLA Decode (a8w8)", ""
    if "mla_reduce" in name:
        m = re.search(r"MlaReduceKernelV1Traits<(\d+), (\d+), (\d+)>", name)
        if m:
            return "MLA Reduce", f"block={m.group(1)}, heads={m.group(2)}, splits={m.group(3)}"
        return "MLA Reduce", ""
    if "mla_metadata" in name:
        m = re.search(r"MlaMetadataV12Traits<(\d+),", name)
        return "MLA Metadata", f"page_size={m.group(1)}" if m else ""
    if "batched_gemm_a8w8" in name:
        bm = re.search(r"BLOCK_SIZE_M_(\d+)", name)
        bn = re.search(r"BLOCK_SIZE_N_(\d+)", name)
        bk = re.search(r"BLOCK_SIZE_K_(\d+)", name)
        gm = re.search(r"GRID_MN_(\d+)", name)
        return "MLA Batched GEMM (a8w8)", f"M={bm.group(1)}, N={bn.group(1)}, K={bk.group(1)}, grid={gm.group(1)}" if bm else ""
    if "reduce_scatter" in name:
        m = re.search(r"<([^,]+), (\d+)>", name)
        dtype = m.group(1).split("::")[-1] if m else "?"
        return "AllReduce (reduce_scatter)", f"dtype={dtype}, ranks={m.group(2)}" if m else ""
    if "allgather" in name:
        m = re.search(r"<([^,]+), (\d+)>", name)
        return "AllGather", f"dtype={m.group(1).split('::')[-1]}" if m else ""
    if "cross_device_reduce" in name:
        return "AllReduce (cross_device_reduce)", ""
    if "local_device_load_rmsnorm" in name:
        m = re.search(r"<([^,]+), (\d+), (\d+)>", name)
        return "Fused Load+RMSNorm+AllReduce", f"dtype={m.group(1).split('::')[-1]}, hidden={m.group(2)}" if m else ""
    if "add_rmsnorm_quant" in name:
        m = re.search(r"<([^,]+), ([^,]+), (\d+), (\d+),", name)
        return "Fused Add+RMSNorm+Quant", f"hidden={m.group(3)}" if m else ""
    if "triton_poi_fused_add_fused_allreduce_rmsnorm" in name:
        return "Triton Fused Add+AllReduce+RMSNorm", ""
    if "act_and_mul" in name:
        return "SiLU Activation (act_and_mul)", "bf16"
    if "fuse_qk_rope_concat_and_cache" in name:
        return "Fused RoPE+KV Cache Store", "bf16->fp8"
    if "FillFunctor" in name:
        return "Fill (zeros)", "bf16"
    if "nccl" in name.lower():
        return "NCCL Collective", ""
    if "mix_sample" in name:
        m = re.search(r"<([^,]+), (\d+), (\d+), (\d+),", name)
        return "Sampling (mix_sample)", f"vocab={m.group(2)}" if m else ""
    if "copyBuffer" in name:
        return "HIP copyBuffer", ""
    if "masked_embedding" in name:
        return "Masked Embedding", ""
    if "kv_indices" in name:
        return "KV Indices Generate", ""
    return name[:120], ""


# Group by category for summary
categories = defaultdict(lambda: {"total_us": 0, "kernels": []})
cat_map = {
    "MoE": ["MoE CK Flatmm", "MoE Sorting", "MoE Weighted Sum", "MoE Grouped TopK"],
    "MLA Attention": ["MLA Decode", "MLA Reduce", "MLA Metadata", "MLA Batched GEMM"],
    "Dense GEMM": ["AITER bf16 GEMV", "hipBLASLt GEMM"],
    "Communication": ["AllReduce", "AllGather", "NCCL", "Fused Load+RMSNorm+AllReduce"],
    "Norm/Activation": ["Fused Add+RMSNorm", "Triton Fused", "SiLU Activation", "RMSNorm"],
    "KV/Misc": ["Fused RoPE", "Fill", "Sampling", "HIP copy", "Masked Embedding", "KV Indices"],
}

def get_category(common_name):
    for cat, prefixes in cat_map.items():
        for p in prefixes:
            if common_name.startswith(p):
                return cat
    return "Other"


entries = []
for name, v in sorted_k[:50]:
    common, params = classify(name)
    cat = get_category(common)
    pct = v["total_us"] / total * 100
    avg = v["total_us"] / v["count"]
    entries.append((common, params, v["count"], v["total_us"], avg, pct, cat))
    categories[cat]["total_us"] += v["total_us"]

# Print markdown
print(f"# ATOM TP4 Kernel Breakdown")
print()
print(f"Total GPU kernel time: **{total/1000:.1f} ms**")
print()

# Category summary
print("## Category Summary")
print()
sorted_cats = sorted(categories.items(), key=lambda x: -x[1]["total_us"])
for cat, cv in sorted_cats:
    cpct = cv["total_us"] / total * 100
    print(f"- **{cat}**: {cv['total_us']/1000:.2f} ms ({cpct:.1f}%)")
print()

# Per-kernel table
print("## Per-Kernel Detail")
print()
print("| # | % | Calls | Total (ms) | Avg (us) | Kernel | Params |")
print("|--:|----:|------:|-----------:|---------:|--------|--------|")
for i, (common, params, count, total_us, avg, pct, cat) in enumerate(entries):
    if pct < 0.05:
        continue
    print(f"| {i+1} | {pct:.1f} | {count} | {total_us/1000:.2f} | {avg:.1f} | {common} | {params} |")
