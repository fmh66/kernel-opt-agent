# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v4.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Buf/ptr** | 2099200 elems |
| **Tolerance** | atol=0.0001  rtol=0.001 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | tensor[float32] | ✓ | 1.4305e-06 | 1.6609e-08 | 1.0384e-07 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -1.3820 | -3.8869 | 0.0646 | 0.1190 | 0.3419 | 1.7379 | -0.1244 | 0.4914 |
| ref    | -1.3820 | -3.8869 | 0.0646 | 0.1190 | 0.3419 | 1.7379 | -0.1244 | 0.4914 |
