# Environment Check

## Status
- ready: yes
- implementation backend ready: yes
- checked at: 2026-05-26T14:54:39
- python: /home/kernel-opt-agent/.venv/bin/python
- python version: 3.11.15 (main, Apr 14 2026, 14:28:36) [Clang 22.1.3 ]
- selected gpu index: 0

## Requirements

| Requirement | Status | Detail |
| --- | --- | --- |
| PyTorch import (yes) | ok | 2.12.0+cu126 |
| CUDA runtime (yes) | ok | torch CUDA 12.6 |
| GPU index 0 (yes) | ok | NVIDIA RTX A6000 (sm_86) |
| nvcc executable (yes) | ok | /usr/local/cuda-12.6/bin/nvcc |
| ncu executable (yes) | ok | /usr/local/cuda-12.6/bin/ncu |
| nsight-python package (yes) | ok | nsight 0.9.6 |
| triton package (implementation) | ok | triton 3.7.0 |
| cutlass python package (implementation) | ok | cutlass 4.5.2 (/home/kernel-opt-agent/.venv/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py) |
| cute-dsl python package (implementation) | ok | cutlass.cute unknown (/home/kernel-opt-agent/.venv/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py) |
| CUTLASS headers (implementation) | unavailable | set CUTLASS_PATH/CUTLASS_ROOT/CUTLASS_HOME to a CUTLASS source/install root containing include/cutlass and include/cute |

## Implementation Backends

| Implementation | Ready | Requirements | Detail |
| --- | --- | --- | --- |
| cuda-cpp | yes | nvcc executable | /usr/local/cuda-12.6/bin/nvcc |
| cute-dsl | yes | cutlass.cute python package | /home/kernel-opt-agent/.venv/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py |
| cutlass | no | nvcc executable, CUTLASS headers | set CUTLASS_PATH/CUTLASS_ROOT/CUTLASS_HOME to a CUTLASS source/install root containing include/cutlass and include/cute |
| triton | yes | triton package | triton 3.7.0 |

## GPU
- model: NVIDIA RTX A6000
- compute capability: 8.6
- sm: sm_86
- driver version: 575.57.08
- torch: 2.12.0+cu126
- torch cuda: 12.6
- device count: 2
- nvidia-smi: /usr/bin/nvidia-smi

## Tools
- nvcc: /usr/local/cuda-12.6/bin/nvcc
- nvcc version: nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
- ncu: /usr/local/cuda-12.6/bin/ncu
- ncu version: NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2024 NVIDIA Corporation
Version 2024.3.2.0 (build 34861637) (public-release)
- nsight-python: 0.9.6
- triton: 3.7.0
- cutlass python: 4.5.2
- cute-dsl python: unknown
- CUTLASS headers: not found

## Environment variables
- CUDA_PATH: (unset)
- CUDA_HOME: (unset)
- CUDA_ROOT: (unset)
- CUTLASS_PATH: (unset)
- CUTLASS_ROOT: (unset)
- CUTLASS_HOME: (unset)

## Errors
- none

## Warnings
- none