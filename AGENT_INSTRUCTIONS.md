# CUDA Kernel Optimization Agent Instructions

You are a CUDA kernel optimization agent. Follow the kernel-opt-skill workflow exactly.

**SKILL_ROOT**: /home/kernel-opt-skill/.claude/skills/kernel-opt-skill

## OVERVIEW

You will optimize a PyTorch kernel by:
1. Running environment check (once)
2. Writing an initial CUDA kernel (v0)
3. Running 5 optimization iterations (v0→v1→v2→v3→v4→v5)
4. Each iteration: correctness → NCU profiling → bottleneck classification → hypothesis → next version
5. After all iterations: benchmark best version

## DETAILED STEPS

### A. Setup (once)

Run environment check:
```bash
python3 /home/kernel-opt-skill/.claude/skills/kernel-opt-skill/env/scripts/env_check.py -o <OUTPUT_DIR>/env_check.md --gpu 0
```
If it fails, report and exit.

### B. v0 - Initial CUDA Kernel

1. Read `<OUTPUT_DIR>/ref.py` to understand the PyTorch reference implementation.
2. Write `<OUTPUT_DIR>/v0/v0.cu` as a CUDA kernel implementing the same functionality.
3. The CUDA code must expose `extern "C" void solve(...)` accepting pointer and scalar arguments.
4. The parameter names must match those in ref.py (e.g., if ref.py uses `batch_size`, use `batch_size` in your CUDA signature).
5. Compile and check correctness before profiling.

### C. Iteration Loop (repeat for v0 through v5)

For each version n (0 to 5):

**C0. Correctness Check**
```bash
SKILL_ROOT=/home/kernel-opt-skill/.claude/skills/kernel-opt-skill
mkdir -p <OUTPUT_DIR>/v{n}
# Write the kernel to <OUTPUT_DIR>/v{n}/v{n}.cu
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o <OUTPUT_DIR>/v{n}/kernel.so <OUTPUT_DIR>/v{n}/v{n}.cu
python3 $SKILL_ROOT/profiling/script/correctness_check.py <OUTPUT_DIR>/v{n}/v{n}.cu --ref=<OUTPUT_DIR>/ref.py --output-dir=<OUTPUT_DIR>/v{n} <DIM_ARGS>
```
IMPORTANT: Extract parameter names and values from ref.py. Pass them as `--param_name=value` to correctness_check.py.
If correctness FAILS → fix the kernel and re-run. Do NOT proceed until it passes.

**C1. NCU Profiling**
```bash
python3 $SKILL_ROOT/profiling/script/ncu_profile.py <OUTPUT_DIR>/v{n}/v{n}.cu --output-dir=<OUTPUT_DIR>/v{n} <DIM_ARGS>
```

**C2. Bottleneck Classification**
Read `<OUTPUT_DIR>/v{n}/ncu_summary.md`.
Classify as:
- Memory-Bound: Memory Throughput % > 60% AND much higher than SM Throughput %
- Compute-Bound: SM Throughput % > 60% AND much higher than Memory Throughput %
- Latency-Bound: Both < 40%
- Occupancy-Bound: Achieved Occupancy << Theoretical

**C3. Query Experience**
```bash
python3 $SKILL_ROOT/experience/learned/scripts/experience_log.py recommend --kernel <kernel_type> --backend cuda --chip sm_90 --bottleneck <memory|compute|latency> 2>&1 || true
```

**C4. Formulate Hypothesis**
Write `<OUTPUT_DIR>/v{n}/hypothesis.txt`:
```
Hypothesis: <ONE specific change>
Rationale : <NCU metric that supports this>
Expected  : <what should improve>
```
ONE change per iteration only (one-variable rule).

**C5. Generate Next Version (skip for v5)**
Write `<OUTPUT_DIR>/v{n+1}/v{n+1}.cu` with exactly ONE change from v{n} based on the hypothesis.

**C6. Record Outcome** (after profiling v{n+1})
```bash
python3 $SKILL_ROOT/experience/learned/scripts/experience_log.py add --kernel <kernel_type> --backend cuda --chip sm_90 --version v{n+1} --result <speedup|regression|neutral> 2>&1 || true
```

### D. After All Iterations

**Benchmark best version** (the one with lowest latency):
```bash
cd <OUTPUT_DIR>
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o v{best}/kernel.so v{best}/v{best}.cu
python3 $SKILL_ROOT/benchmark/script/benchmark.py v{best}/v{best}.cu --ref=ref.py --output-dir=. <DIM_ARGS>
```

**Sync experience**:
```bash
python3 $SKILL_ROOT/experience/learned/scripts/experience_log.py sync 2>&1 || true
```

## CRITICAL RULES

1. **One variable per iteration** - test exactly ONE change at a time
2. **Correctness before profiling** - never profile a broken kernel
3. **Metrics drive decisions** - always read NCU output before hypothesizing
4. **Don't modify ref.py** - it's the ground truth
5. **Compile before correctness/profile** - always run nvcc first
6. **CUDA entry point**: `extern "C" void solve(...)` with pointer args for tensors and int/long args for dimensions
7. **Use sm_90** architecture (Hopper)
8. **O3 optimization** always

## CUDA CODE PATTERNS

Your .cu file should follow this structure:
```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Your kernel(s) here
__global__ void my_kernel(float* input, float* output, int N) { ... }

// Entry point called by the harness
extern "C" void solve(float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

For reduction operations, use shared memory reduction patterns.
For element-wise operations, use simple strided kernels.
For matrix operations, use tiled shared memory approach.

## FINAL REPORT

When complete, summarize:
- Best version (v0-v5)
- Speedup vs PyTorch reference
- Bottleneck type found
- Key optimizations that worked
- Any regressions encountered
