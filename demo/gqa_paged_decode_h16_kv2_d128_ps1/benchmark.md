# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `kernel.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {} |
| **Correctness** | PASS |
| **Timing Method** | cuda_event |
| **Prewarm Calls** | 1 |
| **Cache Mode** | torch_l2_thrash |
| **Timing Scope** | Preallocated/static tensors; solution and selected baselines exclude per-call input cloning. |

## Timing

| Metric | Solution | Torch Compile | FlashInfer |
|--------|----------:|----------:|----------:|
| Mean Time (ms) | 0.0168 | 6.2403 | 0.2365 |
| Median Time (ms) | 0.0164 | 6.2044 | 0.2355 |
| P20 Time (ms) | 0.0164 | 6.1901 | 0.2314 |
| P80 Time (ms) | 0.0174 | 6.2265 | 0.2417 |
| Min Time (ms) | 0.0164 | 6.1706 | 0.2284 |
| Max Time (ms) | 0.0175 | 8.9303 | 0.2499 |
| Std dev (ms) | 0.0005 | 0.2740 | 0.0053 |
| Samples | 100 | 100 | 100 |
| Speedup (vs Torch Compile, mean) | 372.24x | - | - |
| Speedup (vs Torch Compile, median) | 378.69x | - | - |
| Speedup (vs FlashInfer, mean) | 14.11x | - | - |
| Speedup (vs FlashInfer, median) | 14.38x | - | - |
