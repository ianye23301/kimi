"""Profile vLLM decode kernels by instrumenting the worker's model execution."""
import os
os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["NCCL_MIN_NCHANNELS"] = "112"

import sys
import torch
from vllm import LLM, SamplingParams

def main():
    tp = int(os.environ.get("TP", "8"))
    model = "/models/Kimi-K2.5-MXFP4"

    print(f"Loading model with TP={tp}...")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.85,
        collect_detailed_traces="all",
    )

    # Warmup
    print("Warming up...")
    sampling = SamplingParams(max_tokens=64, ignore_eos=True)
    prompt_short = " ".join(["word"] * 200)
    llm.generate([prompt_short] * 2, sampling)

    # Real run with profiling enabled via collect_detailed_traces
    print("Profiling decode...")
    sampling = SamplingParams(max_tokens=128, ignore_eos=True)
    prompt = " ".join(["word"] * 8000)
    llm.generate([prompt] * 4, sampling)

    print("Done. Check /tmp/ for traces.")


if __name__ == "__main__":
    main()
