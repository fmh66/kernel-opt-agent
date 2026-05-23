# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | kernel.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Buf/ptr** | 50331648 elems |
| **Tolerance** | atol=0.01  rtol=0.01 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | tensor[float16] | ✓ | 2.4414e-04 | 1.3142e-05 | 5.5837e-03 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.0162 | -0.0242 | -0.0161 | 0.0142 | 0.0038 | 0.0388 | 0.0281 | -0.0106 |
| ref    | -0.0162 | -0.0242 | -0.0161 | 0.0142 | 0.0038 | 0.0388 | 0.0280 | -0.0106 |
