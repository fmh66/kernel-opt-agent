#!/usr/bin/env python3
"""Benchmark CUDA/CUTLASS/CuTe DSL/Triton kernel vs PyTorch reference.

Compares performance of a custom kernel against a PyTorch reference
using KernelBench-style CUDA event timing by default.

Reference .py must define `reference(**kwargs)` — in-place, PyTorch tensors.
CUDA-C++/CUTLASS solution: .cu exposing `extern "C" void solve(...)`.
CuTe DSL/Triton solution: .py defining `setup(**kwargs)` and `run_kernel(**kwargs)`.

Usage:
    python benchmark.py solution.cu --ref=ref.py --output-dir=./out --M=1024 --N=1024
    python benchmark.py solution.cu --ref=ref.py --output-dir=./out --M=4096 --N=4096
"""

import argparse
import copy
import ctypes
import importlib.util
import os
import re
import statistics
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Type tables for parsing extern "C" void solve(...)
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float*":         torch.float32,
    "double*":        torch.float64,
    "int*":           torch.int32,
    "long*":          torch.int64,
    "short*":         torch.int16,
    "char*":          torch.int8,
    "unsigned char*": torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*":  getattr(torch, "uint32", torch.int32),
}

_CTYPE_MAP = {
    "float*":          ctypes.c_void_p,
    "double*":         ctypes.c_void_p,
    "unsigned char*":  ctypes.c_void_p,
    "unsigned short*": ctypes.c_void_p,
    "unsigned int*":   ctypes.c_void_p,
    "char*":           ctypes.c_void_p,
    "short*":          ctypes.c_void_p,
    "long*":           ctypes.c_void_p,
    "int*":            ctypes.c_void_p,
    "int":             ctypes.c_int,
    "long":            ctypes.c_long,
    "size_t":          ctypes.c_size_t,
    "unsigned int":    ctypes.c_uint,
    "unsigned short":  ctypes.c_ushort,
    "unsigned char":   ctypes.c_ubyte,
    "char":            ctypes.c_char,
    "short":           ctypes.c_short,
}

_INT_TYPES = {"int", "long", "size_t", "unsigned int"}
SUPPORTED_IMPLEMENTATIONS = ("auto", "cuda-cpp", "cute-dsl", "cutlass", "triton")
CUDA_IMPLEMENTATIONS = {"cuda", "cuda-cpp", "cutlass"}
PYTHON_IMPLEMENTATIONS = {"cute-dsl", "triton"}


@dataclass
class BackendState:
    backend: str
    callable: object
    tensors: dict
    ref_inputs: dict
    output_names: list


@dataclass
class TimingConfig:
    method: str
    num_warmup: int
    num_trials: int
    discard_first: int
    prewarm_calls: int
    device: int


# ---------------------------------------------------------------------------
# Helpers (self-contained, no cross-skill imports)
# ---------------------------------------------------------------------------

def _detect_arch(device_index=0):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"sm_{major}{minor}"
    return "sm_80"


def _parse_signature(cu_file):
    with open(cu_file, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{', src)
    if not m:
        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')
    raw = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())
    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(_CTYPE_MAP, key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            hit = re.match(rf"({base})\s+(\w+)", clean)
            if hit:
                params.append((key, hit.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")
    return params


def _load_reference(ref_file):
    spec = importlib.util.spec_from_file_location("_ref", ref_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(f"'{ref_file}' must define reference(**kwargs)")
    return mod


def _load_python_module(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)


def _parse_dim_values(extra_args):
    dim_values = {}
    for item in extra_args:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{item}'", file=sys.stderr)
    return dim_values


def _prepare_reference_call_inputs(ref_inputs, output_names):
    """Allow references that treat output tensors as flat buffers."""
    call_inputs = dict(ref_inputs)
    for name in output_names:
        value = call_inputs.get(name)
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            call_inputs[name] = value.reshape(-1)
    return call_inputs


def _infer_backend(solution_file):
    return "triton" if os.path.splitext(solution_file)[1].lower() == ".py" else "cuda-cpp"


def _normalize_implementation(implementation):
    value = (implementation or "auto").lower()
    aliases = {
        "cuda": "cuda-cpp",
        "cpp": "cuda-cpp",
        "cute": "cute-dsl",
        "cutedsl": "cute-dsl",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_IMPLEMENTATIONS:
        allowed = ", ".join(SUPPORTED_IMPLEMENTATIONS)
        raise ValueError(f"Unsupported implementation: {implementation}. Expected one of: {allowed}")
    return value


# ---------------------------------------------------------------------------
# Solution setup (CUDA-C++ / CUTLASS / CuTe DSL / Triton)
# ---------------------------------------------------------------------------

def _setup_kernel(cu_file, dim_values, ptr_size_override, arch, seed):
    """Load pre-compiled .so, allocate CUDA buffers, return BackendState."""
    params = _parse_signature(cu_file)

    so_path = os.path.splitext(cu_file)[0] + (".dll" if os.name == "nt" else ".so")
    if not os.path.exists(so_path):
        sys.exit(f"[error] .so not found: {so_path}\n"
                 f"        Compile first: nvcc -shared -std=c++17 -arch={arch} "
                 f"-O3 -Xcompiler -fPIC -o {so_path} {cu_file}")
    lib = ctypes.CDLL(so_path)

    for ptype, pname, _ in params:
        if ptype in _INT_TYPES and pname not in dim_values:
            raise ValueError(f"Missing dimension --{pname}=<value>")

    int_vals = [dim_values[n] for t, n, _ in params if t in _INT_TYPES]
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(int_vals) == 0:
        ptr_elems = 1024 * 1024
    elif len(int_vals) == 1:
        ptr_elems = int_vals[0]
    else:
        sv = sorted(int_vals, reverse=True)
        ptr_elems = sv[0] * sv[1]
    ptr_elems = min(ptr_elems, 256 * 1024 * 1024)

    if seed is not None:
        torch.manual_seed(seed)

    tensors, ref_inputs, call_args, argtypes = {}, {}, [], []
    for ptype, pname, is_const in params:
        if ptype in _DTYPE_MAP:
            dtype = _DTYPE_MAP[ptype]
            t = (torch.randn(ptr_elems, device="cuda", dtype=dtype)
                 if dtype.is_floating_point
                 else torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_())
            tensors[pname] = t
            ref_inputs[pname] = t
            call_args.append(ctypes.c_void_p(t.data_ptr()))
            argtypes.append(ctypes.c_void_p)
        else:
            ctype = _CTYPE_MAP[ptype]
            val = dim_values[pname]
            ref_inputs[pname] = val
            call_args.append(ctype(val))
            argtypes.append(ctype)

    lib.solve.restype = None
    lib.solve.argtypes = argtypes

    return BackendState(
        backend="cuda-cpp",
        callable=lambda: lib.solve(*call_args),
        tensors=tensors,
        ref_inputs=ref_inputs,
        output_names=[n for t, n, c in params if t in _DTYPE_MAP and not c],
    )


def _setup_python_kernel(py_file, implementation, dim_values, seed):
    module = _load_python_module(py_file, f"_{implementation.replace('-', '_')}_kernel_module")
    if not hasattr(module, "setup"):
        raise AttributeError(f"'{py_file}' must define setup(**kwargs)")
    if not hasattr(module, "run_kernel"):
        raise AttributeError(f"'{py_file}' must define run_kernel(**kwargs)")

    if seed is not None:
        torch.manual_seed(seed)

    setup_kwargs = dict(dim_values)
    if "seed" not in setup_kwargs and seed is not None:
        setup_kwargs["seed"] = seed
    prepared = module.setup(**setup_kwargs)
    if not isinstance(prepared, dict):
        raise TypeError(f"{implementation} setup() must return dict with 'inputs' and 'outputs'")

    ref_inputs = prepared.get("inputs")
    outputs = prepared.get("outputs")
    if not isinstance(ref_inputs, dict):
        raise TypeError(f"{implementation} setup()['inputs'] must be a dict")
    if not isinstance(outputs, (list, tuple)):
        raise TypeError(f"{implementation} setup()['outputs'] must be a list/tuple")

    for name in outputs:
        if name not in ref_inputs:
            raise ValueError(f"{implementation} output '{name}' not found in setup()['inputs']")
        if not isinstance(ref_inputs[name], torch.Tensor):
            raise TypeError(f"{implementation} output '{name}' must be a torch.Tensor")

    tensors = {k: v for k, v in ref_inputs.items() if isinstance(v, torch.Tensor)}
    return BackendState(
        backend=implementation,
        callable=lambda: module.run_kernel(**ref_inputs),
        tensors=tensors,
        ref_inputs=ref_inputs,
        output_names=list(outputs),
    )


def _setup_solution(solution_file, implementation, dim_values, ptr_size_override, arch, seed):
    requested = _normalize_implementation(implementation)
    resolved = _infer_backend(solution_file) if requested == "auto" else requested
    if resolved in CUDA_IMPLEMENTATIONS:
        state = _setup_kernel(solution_file, dim_values, ptr_size_override, arch, seed)
        state.backend = resolved
        return state
    if resolved in PYTHON_IMPLEMENTATIONS:
        return _setup_python_kernel(solution_file, resolved, dim_values, seed)
    raise ValueError(f"Unsupported backend: {resolved}")


# ---------------------------------------------------------------------------
# Torch reference helpers
# ---------------------------------------------------------------------------

def _make_torch_fns(ref_fn, ref_inputs_snapshot, output_names):
    """Build PyTorch timing closures.

    The default eager/compiled closures reuse static CUDA tensors so their timing
    is comparable with custom kernels that run in-place on preallocated buffers.
    """
    compile_fn = torch.compile(ref_fn, dynamic=False)

    eager_static_inputs = {k: _clone_value(v) for k, v in ref_inputs_snapshot.items()}
    compile_static_inputs = {k: _clone_value(v) for k, v in ref_inputs_snapshot.items()}

    def eager_static():
        ref_fn(**_prepare_reference_call_inputs(eager_static_inputs, output_names))

    def compiled_static():
        compile_fn(**_prepare_reference_call_inputs(compile_static_inputs, output_names))

    return eager_static, compiled_static, compile_fn


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _summarize_times(times):
    if not times:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p20": 0.0,
            "p80": 0.0,
            "num_trials": 0,
        }
    vals = sorted(float(v) for v in times)

    def pct(p):
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
        "p20": pct(0.20),
        "p80": pct(0.80),
        "num_trials": len(vals),
    }


def _clear_l2_cache_torch(device):
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def _prewarm(fn, calls, device):
    if calls <= 0:
        return
    print(f"[prewarm] running {calls} untimed call(s) to trigger lazy init/JIT...")
    with torch.cuda.device(device):
        for _ in range(calls):
            fn()
        torch.cuda.synchronize(device=device)


def _bench_times_cuda_event(fn, cfg):
    device = torch.device(f"cuda:{cfg.device}")
    times = []
    with torch.cuda.device(device):
        for _ in range(cfg.num_warmup):
            fn()
            torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()

        for trial in range(cfg.num_trials + cfg.discard_first):
            torch.cuda.synchronize(device=device)
            _clear_l2_cache_torch(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device=device)
            if trial >= cfg.discard_first:
                times.append(float(start.elapsed_time(end)))
    return times


def _bench_times_host_time(fn, cfg):
    device = torch.device(f"cuda:{cfg.device}")
    times = []
    with torch.cuda.device(device):
        for _ in range(cfg.num_warmup):
            fn()
            torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()

        for trial in range(cfg.num_trials + cfg.discard_first):
            torch.cuda.synchronize(device=device)
            _clear_l2_cache_torch(device)
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize(device=device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if trial >= cfg.discard_first:
                times.append(float(elapsed_ms))
    return times


def _bench_times(fn, cfg):
    if cfg.method == "cuda_event":
        return _bench_times_cuda_event(fn, cfg)
    if cfg.method == "host_time":
        return _bench_times_host_time(fn, cfg)
    raise ValueError(f"Unsupported timing method: {cfg.method}")


def _run_variant(label, fn, cfg):
    """Time fn with the configured benchmark method."""
    detail = (
        f"warmup_calls={cfg.num_warmup}, trials={cfg.num_trials}, "
        f"discard_first={cfg.discard_first}, cache=torch_l2_thrash"
    )
    print(f"[timing] {label} ({cfg.method}, {detail})...")
    times = _bench_times(fn, cfg)
    stats = _summarize_times(times)
    ms, std = stats["mean"], stats["std"]
    print(
        f"[timing] {label} : mean {ms:.4f} ms ± {std:.4f} ms, "
        f"median {stats['median']:.4f} ms, min {stats['min']:.4f} ms, "
        f"max {stats['max']:.4f} ms, n={stats['num_trials']}"
    )
    if stats["num_trials"] < 3:
        print(
            f"[warn] {label} produced only {stats['num_trials']} timing sample(s); "
            "increase --num-trials for more stable statistics.",
            file=sys.stderr,
        )
    return ms, std, stats


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def _check_correctness(sol_tensors, ref_inputs_snapshot, ref_fn, output_names,
                       atol=1e-4, rtol=1e-3):
    """Run ref on cloned inputs and compare outputs with torch.allclose."""
    cloned = {k: _clone_value(v) for k, v in ref_inputs_snapshot.items()}
    ref_call_inputs = _prepare_reference_call_inputs(cloned, output_names)
    ref_fn(**ref_call_inputs)
    torch.cuda.synchronize()

    all_pass = True
    for name in output_names:
        if name not in cloned or not isinstance(cloned[name], torch.Tensor):
            continue
        ok = torch.allclose(sol_tensors[name].float(), cloned[name].float(),
                            atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False
    return all_pass


def _run_correctness(label, sol_tensors, ref_inputs_snapshot, ref_fn, compile_fn,
                     output_names, atol, rtol):
    """Check solution against PyTorch eager and compiled reference."""
    ok = True
    print(f"[correctness] checking {label} vs PyTorch eager...")
    ok = _check_correctness(sol_tensors, ref_inputs_snapshot, ref_fn,
                            output_names, atol, rtol)
    print(f"[correctness] checking {label} vs PyTorch compile...")
    ok = _check_correctness(sol_tensors, ref_inputs_snapshot, compile_fn,
                            output_names, atol, rtol) and ok
    print(f"[correctness] {'PASS' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(solution_file, ref_file, dim_values, arch,
                 sol_ms, sol_std,
                 torch_eager_ms, torch_eager_std,
                 torch_compile_ms, torch_compile_std,
                 correctness_pass,
                 sol_stats=None, torch_eager_stats=None, torch_compile_stats=None,
                 timing_config=None):
    gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    sol_stats = sol_stats or {"mean": sol_ms, "std": sol_std, "median": sol_ms, "min": sol_ms, "max": sol_ms}
    torch_eager_stats = torch_eager_stats or {
        "mean": torch_eager_ms,
        "std": torch_eager_std,
        "median": torch_eager_ms,
        "min": torch_eager_ms,
        "max": torch_eager_ms,
    }
    torch_compile_stats = torch_compile_stats or {
        "mean": torch_compile_ms,
        "std": torch_compile_std,
        "median": torch_compile_ms,
        "min": torch_compile_ms,
        "max": torch_compile_ms,
    }

    lines = [
        "# Benchmark Report",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Solution** | `{os.path.basename(solution_file)}` |",
        f"| **Reference** | `{os.path.basename(ref_file)}` |",
        f"| **GPU** | {gpu} |",
        f"| **Arch** | {arch} |",
        f"| **Dims** | {dim_values} |",
        f"| **Correctness** | {'PASS' if correctness_pass else 'FAIL'} |",
        f"| **Timing Method** | {timing_config.method if timing_config else 'cuda_event'} |",
        f"| **Prewarm Calls** | {timing_config.prewarm_calls if timing_config else 1} |",
        "| **Cache Mode** | torch_l2_thrash |",
        "| **Timing Scope** | Preallocated/static tensors; solution and PyTorch baselines exclude per-call input cloning. |",
        "",
        "## Timing",
        "",
        "| Metric | Solution | PyTorch Eager | PyTorch Compile |",
        "|--------|----------:|-------------:|----------------:|",
        f"| Mean Time (ms)      | {sol_stats['mean']:.4f} | {torch_eager_stats['mean']:.4f} | {torch_compile_stats['mean']:.4f} |",
        f"| Median Time (ms)    | {sol_stats['median']:.4f} | {torch_eager_stats['median']:.4f} | {torch_compile_stats['median']:.4f} |",
        f"| P20 Time (ms)       | {sol_stats['p20']:.4f} | {torch_eager_stats['p20']:.4f} | {torch_compile_stats['p20']:.4f} |",
        f"| P80 Time (ms)       | {sol_stats['p80']:.4f} | {torch_eager_stats['p80']:.4f} | {torch_compile_stats['p80']:.4f} |",
        f"| Min Time (ms)       | {sol_stats['min']:.4f} | {torch_eager_stats['min']:.4f} | {torch_compile_stats['min']:.4f} |",
        f"| Max Time (ms)       | {sol_stats['max']:.4f} | {torch_eager_stats['max']:.4f} | {torch_compile_stats['max']:.4f} |",
        f"| Std dev (ms)        | {sol_stats['std']:.4f} | {torch_eager_stats['std']:.4f} | {torch_compile_stats['std']:.4f} |",
        f"| Samples             | {sol_stats['num_trials']} | {torch_eager_stats['num_trials']} | {torch_compile_stats['num_trials']} |",
    ]
    for label, base_stats in (
        ("vs PyTorch Eager", torch_eager_stats),
        ("vs PyTorch Compile", torch_compile_stats),
    ):
        mean_speedup = base_stats["mean"] / sol_stats["mean"] if sol_stats["mean"] > 0 else float("inf")
        median_speedup = base_stats["median"] / sol_stats["median"] if sol_stats["median"] > 0 else float("inf")
        lines.append(f"| Speedup ({label}, mean)   | {mean_speedup:.2f}x | - | - |")
        lines.append(f"| Speedup ({label}, median) | {median_speedup:.2f}x | - | - |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA-C++/CUTLASS/CuTe DSL/Triton kernel vs PyTorch (eager + compile)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to solution file (.cu or .py)")
    parser.add_argument("--implementation", "--impl", "--backend", dest="backend", type=str,
                        default="auto",
                        choices=SUPPORTED_IMPLEMENTATIONS + ("cuda", "cute", "cutedsl", "cpp"),
                        help="Kernel implementation: auto/cuda-cpp/cute-dsl/cutlass/triton (default: auto)")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to reference .py defining reference(**kwargs)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output files")
    parser.add_argument("--timing-method", type=str, default="cuda_event",
                        choices=["cuda_event", "host_time"],
                        help="Timing method: cuda_event fixed trials or host_time end-to-end timing")
    parser.add_argument("--num-warmup", type=int, default=5,
                        help="Warmup calls for explicit timing methods (default: 5)")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="Measured trials for explicit timing methods (default: 100)")
    parser.add_argument("--discard-first", type=int, default=1,
                        help="Discard this many first measured trials for explicit methods (default: 1)")
    parser.add_argument("--prewarm-calls", type=int, default=1,
                        help="Untimed calls before correctness/timing to trigger lazy init/JIT (default: 1)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--atol", type=float, default=1e-4, help="Correctness atol (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Correctness rtol (default: 1e-3)")

    args, unknown = parser.parse_known_args()
    dim_values = _parse_dim_values(unknown)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else _detect_arch(args.gpu)
    solution_file = str(Path(args.solution_file).resolve())
    ref_file = str(Path(args.ref).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_cfg = TimingConfig(
        method=args.timing_method,
        num_warmup=args.num_warmup,
        num_trials=args.num_trials,
        discard_first=args.discard_first,
        prewarm_calls=args.prewarm_calls,
        device=args.gpu,
    )

    # --- setup ----------------------------------------------------------

    state = _setup_solution(solution_file, args.backend, dim_values,
                            args.ptr_size, arch, args.seed)
    ref_mod = _load_reference(ref_file)
    inputs_snapshot = {k: _clone_value(v) for k, v in state.ref_inputs.items()}

    sol_fn = state.callable
    (
        torch_eager_fn,
        torch_compile_fn_timing,
        torch_compile_ref,
    ) = _make_torch_fns(
        ref_mod.reference, inputs_snapshot, state.output_names)

    print(f"[benchmark] solution  : {solution_file}")
    print(f"[benchmark] backend   : {state.backend}")
    print(f"[benchmark] reference : {ref_file}")
    print(f"[benchmark] arch      : {arch}")
    print(f"[benchmark] dims      : {dim_values}")
    print()

    # CuTe DSL, Triton, and torch.compile may lazily initialize runtime state.
    # Keep one-time setup out of correctness and timing measurements.
    _prewarm(sol_fn, timing_cfg.prewarm_calls, args.gpu)

    # --- correctness ----------------------------------------------------

    ok = _run_correctness(
        "solution", state.tensors, inputs_snapshot,
        ref_mod.reference, torch_compile_ref,
        state.output_names, args.atol, args.rtol,
    )

    # --- timing ---------------------------------------------------------

    sol_ms, sol_std, sol_stats = _run_variant(
        "Solution", sol_fn, timing_cfg)
    torch_eager_ms, torch_eager_std, torch_eager_stats = _run_variant(
        "PyTorch eager", torch_eager_fn, timing_cfg)
    torch_compile_ms, torch_compile_std, torch_compile_stats = _run_variant(
        "PyTorch compile", torch_compile_fn_timing, timing_cfg)

    # --- report ---------------------------------------------------------

    report = build_report(
        solution_file=solution_file,
        ref_file=ref_file,
        dim_values=dim_values,
        arch=arch,
        sol_ms=sol_ms, sol_std=sol_std,
        torch_eager_ms=torch_eager_ms, torch_eager_std=torch_eager_std,
        torch_compile_ms=torch_compile_ms, torch_compile_std=torch_compile_std,
        correctness_pass=ok,
        sol_stats=sol_stats,
        torch_eager_stats=torch_eager_stats,
        torch_compile_stats=torch_compile_stats,
        timing_config=timing_cfg,
    )
    report_path = output_dir / "benchmark.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[benchmark] report    -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
