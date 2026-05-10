#!/usr/bin/env python3
"""Benchmark CUDA/Triton kernel vs PyTorch reference (eager + compile).

Compares performance of a custom kernel against a PyTorch reference
using CUDA event timing and nsight-python hardware metrics.

Reference .py must define `reference(**kwargs)` — in-place, PyTorch tensors.
CUDA solution: .cu exposing `extern "C" void solve(...)`.
Triton solution: .py defining `setup(**kwargs)` and `run_kernel(**kwargs)`.

Usage:
    python benchmark.py solution.cu --ref=ref.py --output-dir=./out --M=1024 --N=1024
    python benchmark.py solution.cu --ref=ref.py --output-dir=./out --skip-nsight --M=4096 --N=4096
"""

import argparse
import copy
import ctypes
import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path
from dataclasses import dataclass

import torch

import nsight

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


@dataclass
class BackendState:
    backend: str
    callable: object
    tensors: dict
    ref_inputs: dict
    output_names: list

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
    return "triton" if os.path.splitext(solution_file)[1].lower() == ".py" else "cuda"


# ---------------------------------------------------------------------------
# Solution setup (CUDA / Triton)
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
        backend="cuda",
        callable=lambda: lib.solve(*call_args),
        tensors=tensors,
        ref_inputs=ref_inputs,
        output_names=[n for t, n, c in params if t in _DTYPE_MAP and not c],
    )


def _setup_triton(py_file, dim_values, seed):
    module = _load_python_module(py_file, "_triton_kernel_module")
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
        raise TypeError("Triton setup() must return dict with 'inputs' and 'outputs'")

    ref_inputs = prepared.get("inputs")
    outputs = prepared.get("outputs")
    if not isinstance(ref_inputs, dict):
        raise TypeError("Triton setup()['inputs'] must be a dict")
    if not isinstance(outputs, (list, tuple)):
        raise TypeError("Triton setup()['outputs'] must be a list/tuple")

    for name in outputs:
        if name not in ref_inputs:
            raise ValueError(f"Triton output '{name}' not found in setup()['inputs']")
        if not isinstance(ref_inputs[name], torch.Tensor):
            raise TypeError(f"Triton output '{name}' must be a torch.Tensor")

    tensors = {k: v for k, v in ref_inputs.items() if isinstance(v, torch.Tensor)}
    return BackendState(
        backend="triton",
        callable=lambda: module.run_kernel(**ref_inputs),
        tensors=tensors,
        ref_inputs=ref_inputs,
        output_names=list(outputs),
    )


def _setup_solution(solution_file, backend, dim_values, ptr_size_override, arch, seed):
    resolved = backend if backend != "auto" else _infer_backend(solution_file)
    if resolved == "cuda":
        return _setup_kernel(solution_file, dim_values, ptr_size_override, arch, seed)
    if resolved == "triton":
        return _setup_triton(solution_file, dim_values, seed)
    raise ValueError(f"Unsupported backend: {resolved}")


# ---------------------------------------------------------------------------
# Torch reference helpers
# ---------------------------------------------------------------------------

def _make_torch_fns(ref_fn, ref_inputs_snapshot, output_names):
    """Build eager/compile timing closures that clone inputs each call."""
    compile_fn = torch.compile(ref_fn, dynamic=False)

    def eager():
        cloned = {k: _clone_value(v) for k, v in ref_inputs_snapshot.items()}
        ref_fn(**_prepare_reference_call_inputs(cloned, output_names))

    def compiled():
        cloned = {k: _clone_value(v) for k, v in ref_inputs_snapshot.items()}
        compile_fn(**_prepare_reference_call_inputs(cloned, output_names))

    return eager, compiled, compile_fn


# ---------------------------------------------------------------------------
# NSight metrics
# ---------------------------------------------------------------------------

BENCH_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

METRIC_LABELS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":  "SM Throughput (% peak)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":"Memory Throughput (% peak)",
    "dram__bytes.sum.per_second":                        "DRAM Bandwidth (bytes/s)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy (%)",
}


def profile_nsight(fn, warmup, replay_mode="kernel"):
    metrics = list(dict.fromkeys(BENCH_METRICS))

    @nsight.analyze.kernel(
        metrics=metrics,
        runs=1,
        output="quiet",
        output_csv=False,
        clock_control="none",
        cache_control="all",
        replay_mode=replay_mode,
    )
    def _run(n_warmup):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        with nsight.annotate("benchmark"):
            fn()

    return _run(warmup).to_dataframe()


def _get_metric(df, name):
    for _, row in df.iterrows():
        if row.get("Metric") == name:
            return row.get("AvgValue")
    return None


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2e}" if abs(v) > 1e6 else f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Timing & profiling
# ---------------------------------------------------------------------------

def time_kernel(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0), times


def _run_variant(label, fn, warmup, iters, skip_nsight, under_nsight, replay_mode="kernel"):
    """Time fn and optionally collect nsight metrics. Returns (ms, std, df)."""
    ms, std = 0.0, 0.0
    if not under_nsight:
        print(f"[timing] {label} ({warmup} warmup, {iters} iters)...")
        ms, std, _ = time_kernel(fn, warmup, iters)
        print(f"[timing] {label} : {ms:.4f} ms ± {std:.4f} ms")
    df = None
    if not skip_nsight:
        if not under_nsight:
            print(f"[nsight] profiling {label}...")
        try:
            df = profile_nsight(fn, warmup, replay_mode=replay_mode)
        except Exception as exc:
            print(f"[nsight] {label} profiling failed: {exc}", file=sys.stderr)
    return ms, std, df


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
                 sol_ms, sol_std, sol_df,
                 torch_eager_ms, torch_eager_std, torch_eager_df,
                 torch_compile_ms, torch_compile_std, torch_compile_df,
                 correctness_pass):
    gpu = torch.cuda.get_device_name(torch.cuda.current_device())

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
        "",
        "## Timing (CUDA Events)",
        "",
        "| Metric | Solution | PyTorch Eager | PyTorch Compile |",
        "|--------|----------:|-------------:|----------------:|",
        f"| Execution Time (ms) | {sol_ms:.4f} | {torch_eager_ms:.4f} | {torch_compile_ms:.4f} |",
        f"| Std dev (ms)        | {sol_std:.4f} | {torch_eager_std:.4f} | {torch_compile_std:.4f} |",
    ]
    for label, base_ms in (("vs PyTorch Eager", torch_eager_ms), ("vs PyTorch Compile", torch_compile_ms)):
        speedup = base_ms / sol_ms if sol_ms > 0 else float("inf")
        lines.append(f"| Speedup ({label})    | {speedup:.2f}x | — | — |")
    lines.append("")

    dfs = {
        "Solution": sol_df,
        "PyTorch Eager": torch_eager_df,
        "PyTorch Compile": torch_compile_df,
    }
    available = {k: v for k, v in dfs.items() if v is not None}
    if available:
        lines += [
            "## Hardware Metrics (nsight-python)",
            "",
            "| Metric | " + " | ".join(available.keys()) + " |",
            "|--------|" + "|".join(["----------:" for _ in available]) + "|",
        ]
        for metric, label in METRIC_LABELS.items():
            vals = [_fmt(_get_metric(df, metric)) for df in available.values()]
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA/Triton kernel vs PyTorch (eager + compile)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to solution file (.cu or .py)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "triton"],
                        help="Backend type for solution file (default: auto)")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to reference .py defining reference(**kwargs)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output files")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations (default: 20)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timing iterations for CUDA event measurement (default: 100)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--atol", type=float, default=1e-4, help="Correctness atol (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Correctness rtol (default: 1e-3)")
    parser.add_argument("--skip-nsight", action="store_true",
                        help="Skip nsight hardware profiling; only run CUDA event timing")

    args, unknown = parser.parse_known_args()
    dim_values = _parse_dim_values(unknown)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else _detect_arch(args.gpu)
    solution_file = str(Path(args.solution_file).resolve())
    ref_file = str(Path(args.ref).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # nsight re-launches the whole process to inject the profiler;
    # lightweight logging-only work in those child processes
    under_nsight = (
        "CUDA_INJECTION64_PATH" in os.environ
        or "NV_NSIGHT_INJECTION64_PATH" in os.environ
        or any(k.startswith("NV_NSIGHT_") for k in os.environ)
    )

    # --- setup ----------------------------------------------------------

    state = _setup_solution(solution_file, args.backend, dim_values,
                            args.ptr_size, arch, args.seed)
    ref_mod = _load_reference(ref_file)
    inputs_snapshot = {k: _clone_value(v) for k, v in state.ref_inputs.items()}

    sol_fn = state.callable
    torch_eager_fn, torch_compile_fn_timing, torch_compile_ref = _make_torch_fns(
        ref_mod.reference, inputs_snapshot, state.output_names)

    if not under_nsight:
        print(f"[benchmark] solution  : {solution_file}")
        print(f"[benchmark] backend   : {state.backend}")
        print(f"[benchmark] reference : {ref_file}")
        print(f"[benchmark] arch      : {arch}")
        print(f"[benchmark] dims      : {dim_values}")
        print()

    # --- correctness ----------------------------------------------------

    ok = True
    if not under_nsight:
        sol_fn()
        torch.cuda.synchronize()
        ok = _run_correctness(
            "solution", state.tensors, inputs_snapshot,
            ref_mod.reference, torch_compile_ref,
            state.output_names, args.atol, args.rtol,
        )

    # --- timing & nsight ------------------------------------------------

    sol_ms, sol_std, sol_df = _run_variant(
        "Solution", sol_fn, args.warmup, args.iters, args.skip_nsight, under_nsight)
    torch_eager_ms, torch_eager_std, torch_eager_df = _run_variant(
        "PyTorch eager", torch_eager_fn, args.warmup, args.iters, args.skip_nsight, under_nsight)
    torch_compile_ms, torch_compile_std, torch_compile_df = _run_variant(
        "PyTorch compile", torch_compile_fn_timing, args.warmup, args.iters, args.skip_nsight, under_nsight)

    if under_nsight:
        return 0

    # --- report ---------------------------------------------------------

    report = build_report(
        solution_file=solution_file,
        ref_file=ref_file,
        dim_values=dim_values,
        arch=arch,
        sol_ms=sol_ms, sol_std=sol_std, sol_df=sol_df,
        torch_eager_ms=torch_eager_ms, torch_eager_std=torch_eager_std,
        torch_eager_df=torch_eager_df,
        torch_compile_ms=torch_compile_ms, torch_compile_std=torch_compile_std,
        torch_compile_df=torch_compile_df,
        correctness_pass=ok,
    )
    report_path = output_dir / "benchmark.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[benchmark] report    -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
