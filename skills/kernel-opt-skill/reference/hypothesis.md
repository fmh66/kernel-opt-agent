# Hypothesis Formulation

After bottleneck classification and metric analysis, compose a hypothesis **before writing any code**. This ensures every optimization decision has a clear rationale and expected outcome.

## One-variable rule

Each iteration tests **exactly one change** from the previous version. This makes it possible to attribute a latency change (good or bad) to a specific optimization. When a change doesn't yield improvement, revert it and test a different fix for the same bottleneck.

**Good hypothesis:** "Add shared memory tiling with tile-size=128 to reduce the Long Scoreboard stall seen in v1's NCU report."

**Bad hypothesis:** "Add tiling, vectorization, and unrolling." (Three variables — which one helped? Which one hurt?)

## Format

Write to `<output_dir>/v{n}/hypothesis.txt` using this format:

```
Hypothesis: <what optimization is being applied or changed>
Rationale : <which bottleneck + which NCU metric supports this>
Expected  : <what metrics should improve and by how much>
```

## Examples

### Memory-Bound kernel

```
Hypothesis: Shared memory tiling with tile-size=128 to reduce global memory latency
Rationale : v1 NCU shows Memory SOL=72%, Long Scoreboard stall=45% indicating memory-bound.
            Tiling reduces global memory round-trips and increases data reuse in shared memory.
Expected  : Memory SOL ↓ below 50%, Long Scoreboard stall ↓ below 20%, latency ↓ 20-30%.
```

### Compute-Bound kernel

```
Hypothesis: Replace scalar FMUL/FADD with Tensor Core WMMA to improve FP32 utilization
Rationale : v1 NCU shows SM SOL=68%, FP32 Pipe Utilization=85% but FFMA=0%.
            Tensor Core instructions can fuse mul+add into a single operation.
Expected  : SM SOL ↓ below 40%, FP32 Pipe Utilization stable, FFMA > 50%, latency ↓ 30-50%.
```

### Latency-Bound kernel

```
Hypothesis: Tune block size from 128 to 256 with __launch_bounds__ to hide memory latency
Rationale : v1 NCU shows Memory SOL=30%, SM SOL=25%, both below 40% → latency-bound.
            Occupancy=35% — larger block increases warps/SM to hide global memory latency.
Expected  : Achieved Occupancy ↑ above 50%, Long Scoreboard stall ↓, latency ↓ 15-25%.
```

### Occupancy-Bound kernel

```
Hypothesis: Reduce register usage from 80 to 64 per thread to raise occupancy
Rationale : v1 NCU shows Achieved Occupancy=25% vs Theoretical=50%, register spill > 0.
            --ptxas-options=-v shows 80 registers/thread. Reducing to 64 allows 2 blocks/SM.
Expected  : Achieved Occupancy ↑ above 40%, register spill → 0, latency ↓ 10-20%.
```
