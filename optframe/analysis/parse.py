"""解析 TraceLens 报表 CSV，产出与框架无关的中间结构。

只读 `unified_perf_summary.csv`（唯一同时带 roofline + kernel 时间 + shape 的表）
与 `gpu_timeline.csv`（拿总 GPU 时间做分母）。不解析 xlsx，避免 openpyxl 版本差异。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

#: TraceLens 的 dtype 字符串 -> 统一写法
_DTYPE_MAP = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    "qint8": "int8",
    "char": "int8",
    "int8": "int8",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "float8_e4m3fnuz": "fp8",
    "float8_e5m2fnuz": "fp8",
    "scalar": "scalar",
    "bool": "bool",
}

_KERNEL_NAME_RE = re.compile(r"'name':\s*'((?:[^'\\]|\\.)*)'")

#: trace 里的包装前缀：cudaLaunchKernelExC->foo / cudaGraphLaunch->foo
_WRAPPER_PREFIX = re.compile(r"^(?:cudaLaunchKernel(?:ExC)?|cudaGraphLaunch|cudaLaunchCooperativeKernel)->")

#: 名字里出现这些词就归为 GEMM 类（即使没有 shape 也能让人看懂）
_GEMM_HINTS = re.compile(r"(marlin|gemm|cublas|cutlass|matmul|wgmma|tcgen05)", re.IGNORECASE)


@dataclass
class OpRow:
    """报表里的一行算子（同 op 不同 shape 视为不同行）。"""

    name: str
    category: str
    shape: str
    dtype: str
    gpu_time_us: float
    count: int
    bound_raw: Optional[str]
    sol_pct_raw: Optional[float]
    tflops: Optional[float]
    data_moved_mb: Optional[float]
    kernel: Optional[str]
    mean_kernel_us: Optional[float]
    pct_reported: Optional[float]


@dataclass
class Timeline:
    """gpu_timeline.csv 的解析结果（单位 ms）。"""

    computation_ms: float
    idle_ms: float
    total_ms: float
    exposed_comm_ms: float

    @property
    def gpu_busy_pct(self) -> float:
        return _safe_div(self.computation_ms * 100.0, self.total_ms)

    @property
    def idle_pct(self) -> float:
        return _safe_div(self.idle_ms * 100.0, self.total_ms)

    @property
    def exposed_comm_pct(self) -> float:
        return _safe_div(self.exposed_comm_ms * 100.0, self.total_ms)


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _is_nan(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except (TypeError, ValueError):
        return True


def normalize_dtype(raw: str) -> str:
    """把 `('c10::BFloat16', 'c10::BFloat16')` 归一化为 `bf16`。"""
    if not raw or _is_nan(raw):
        return "unknown"
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(raw))
    seen: List[str] = []
    for tok in tokens:
        key = tok.lower()
        mapped = _DTYPE_MAP.get(key)
        if mapped and mapped not in seen:
            seen.append(mapped)
    if not seen:
        return "unknown"
    return seen[0] if len(seen) == 1 else "+".join(seen)


def normalize_op_name(name: str) -> str:
    """去掉 trace 的启动包装前缀，让名字可读。"""
    return _WRAPPER_PREFIX.sub("", (name or "").strip())


def _extract_kernel(raw) -> Optional[str]:
    """从 kernel_details_summary 的 python-repr 里抠出第一个 kernel 名。"""
    if raw is None or _is_nan(raw):
        return None
    match = _KERNEL_NAME_RE.search(str(raw))
    if not match:
        return None
    return match.group(1).replace("\\'", "'")


def load_timeline(csv_dir: Path) -> Timeline:
    """读取 gpu_timeline.csv。缺文件时抛 FileNotFoundError。"""
    path = Path(csv_dir) / "gpu_timeline.csv"
    if not path.exists():
        raise FileNotFoundError(f"gpu_timeline.csv not found in {csv_dir}")
    df = pd.read_csv(path)
    values: Dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row["type"]).strip()
        try:
            values[key] = float(row["time ms"])
        except (TypeError, ValueError):
            continue
    return Timeline(
        computation_ms=values.get("computation_time", 0.0),
        idle_ms=values.get("idle_time", 0.0),
        total_ms=values.get("total_time", 0.0),
        exposed_comm_ms=values.get("exposed_comm_time", 0.0),
    )


def load_ops(csv_dir: Path) -> List[OpRow]:
    """读取 unified_perf_summary.csv，按行返回算子。"""
    path = Path(csv_dir) / "unified_perf_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"unified_perf_summary.csv not found in {csv_dir}")
    df = pd.read_csv(path)

    required = {"name", "op category", "Kernel Time (µs)_sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"unified_perf_summary.csv missing columns: {sorted(missing)}")

    rows: List[OpRow] = []
    for _, row in df.iterrows():
        gpu_us = _to_float(row.get("Kernel Time (µs)_sum"))
        if gpu_us is None:
            gpu_us = _to_float(row.get("total_duration_us")) or 0.0
        if gpu_us <= 0:
            continue

        bound_raw = row.get("Roofline Bound")
        bound_raw = None if _is_nan(bound_raw) else str(bound_raw).strip()
        sol = _to_float(row.get("Pct Roofline_mean"))
        count = int(_to_float(row.get("operation_count")) or 0)
        mean_us = _to_float(row.get("Kernel Time (µs)_mean"))

        raw_name = str(row["name"]).strip()
        name = normalize_op_name(raw_name)
        category = str(row.get("op category", "other")).strip()
        if category in ("other", "") and _GEMM_HINTS.search(name):
            category = "GEMM"

        rows.append(
            OpRow(
                name=name,
                category=category,
                shape=str(row.get("Input Dims", "")).strip(),
                dtype=normalize_dtype(row.get("Input type", "")),
                gpu_time_us=gpu_us,
                count=count,
                bound_raw=bound_raw,
                sol_pct_raw=sol,
                tflops=_to_float(row.get("TFLOPS/s_mean")),
                data_moved_mb=_to_float(row.get("Data Moved (MB)")),
                kernel=_extract_kernel(row.get("kernel_details_summary")),
                mean_kernel_us=mean_us,
                pct_reported=_to_float(row.get("Percentage (%)")),
            )
        )
    rows.sort(key=lambda r: r.gpu_time_us, reverse=True)
    return rows


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def normalize_bound(raw: Optional[str]) -> str:
    """TraceLens 的 bound 字符串 -> schema 的 bound_type。"""
    if not raw:
        return "unknown"
    key = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if "COMPUTE" in key:
        return "compute"
    if "MEMORY" in key or "DRAM" in key or "BANDWIDTH" in key:
        return "memory"
    if "LATENCY" in key:
        return "latency"
    if "LAUNCH" in key:
        return "launch"
    if "COMM" in key:
        return "comm"
    return "unknown"
