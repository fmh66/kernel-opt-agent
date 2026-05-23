# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | flash_attention.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 2048, 'd': 64} |
| **Buf/ptr** | 25165824 elems |
| **Tolerance** | atol=0.01  rtol=0.01 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | tensor[float16] | ✓ | 4.8828e-04 | 1.8197e-05 | 5.0056e-03 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.0524 | -0.0048 | -0.0291 | 0.0371 | -0.0111 | 0.0384 | -0.0404 | 0.0075 |
| ref    | -0.0524 | -0.0048 | -0.0291 | 0.0371 | -0.0111 | 0.0384 | -0.0404 | 0.0075 |
