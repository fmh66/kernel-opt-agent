# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v0.cu |
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
| output | float* | ✓ | 4.7684e-07 | 2.0627e-08 | 1.8181e-06 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.0627 | -0.0439 | -0.0270 | 0.0697 | -0.1398 | -0.1186 | 0.0207 | 0.2197 |
| ref    | -0.0627 | -0.0439 | -0.0270 | 0.0697 | -0.1398 | -0.1186 | 0.0207 | 0.2197 |
