# ATOM Baseline Results - Kimi K2.5 MXFP4

- **Model**: /models/Kimi-K2.5-MXFP4
- **Framework**: ATOM (patched for K2.5 support)
- **Config**: TP8, kv_cache_dtype=fp8, gpu_memory_utilization=0.85
- **Hardware**: 8x MI355X
- **ISL**: 8000, **OSL**: 1024, random_range_ratio=0.8

## Results

| Metric                    | conc=4     | conc=32     | conc=128     |
|---------------------------|------------|-------------|--------------|
| Median TPOT (ms)          | 14.47      | 28.05       | 62.74        |
| Interactivity (tok/s/user)| 69.1       | 35.7        | 15.9         |
| **Target interactivity**  | **>=150**  | **>=65**    | **>=35**     |
| Total throughput (tok/s)  | 2328       | 9503        | 17673        |
| Throughput/GPU            | 291        | 1188        | 2209         |
| **Target throughput/GPU** | **>=1350** | **>=4500**  | **>=5300**   |
| Median E2E (s)            | 13.58      | 26.64       | 58.36        |
| **Target E2E (s)**        | **<=6**    | **<=14**    | **<=24.5**   |
| Output throughput (tok/s) | 264        | 1084        | 2001         |
| Mean TTFT (ms)            | 374        | 692         | 1801         |
| Median TTFT (ms)          | 340        | 359         | 547          |

## Summary

All metrics significantly below targets. Roughly 2-5x off on throughput and interactivity.
Possible factors:
- Head-repeat mechanism (64 heads / TP8 = 8 heads/rank, padded to 16)
- Unoptimized baseline — no tuning done
- May need TP4 if model fits with less KV cache headroom
