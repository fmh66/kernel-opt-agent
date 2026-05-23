# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v1.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Buf/ptr** | 2098176 elems |
| **Tolerance** | atol=0.0001  rtol=0.001 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | tensor[float32] | ✓ | 1.9073e-06 | 8.9513e-09 | 1.3895e-08 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.1304 | -3.8921 | 0.1464 | -1.3603 | 0.8019 | 0.5981 | -0.4143 | -0.0351 |
| ref    | -0.1304 | -3.8921 | 0.1464 | -1.3603 | 0.8019 | 0.5981 | -0.4143 | -0.0351 |
