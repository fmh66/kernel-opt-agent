# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v2.cu |
| **Backend** | cuda |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Buf/ptr** | 524288 elems |
| **Tolerance** | atol=0.001  rtol=0.01 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | float* | ✓ | 1.3411e-07 | 7.5783e-09 | 8.0487e-07 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.0627 | -0.0439 | -0.0270 | 0.0697 | -0.1398 | -0.1186 | 0.0207 | 0.2197 |
| ref    | -0.0627 | -0.0439 | -0.0270 | 0.0697 | -0.1398 | -0.1186 | 0.0207 | 0.2197 |
