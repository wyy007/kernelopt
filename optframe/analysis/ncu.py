"""可选：把 ncu 硬件计数器结果并入瓶颈（总体方案 §三 步骤4）。

没有 ncu 数据时，所有 finding 只能是 derived；只有真的拿到硬件计数器，
才允许把 confidence 升到 measured —— 这是硬约束，不是建议。

支持两种 ncu CSV：
  1. 长表（默认 `ncu --csv`）：每行一个 metric，列为 Metric Name / Metric Value；
  2. 宽表（`ncu --csv --page raw`）：每个 metric 一列。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

#: 关注的三类指标（ncu 原始 metric 名）
METRIC_SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_DRAM = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"
METRIC_DURATION = "gpu__time_duration.sum"

#: ncu 判定 bound 的阈值（百分点差）
_BOUND_MARGIN = 10.0


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "inf"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def load_ncu_csv(path: str | Path) -> Dict[str, Dict[str, object]]:
    """解析 ncu CSV，返回 {kernel_name: {metric_name: value}}。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ncu csv not found: {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    # ncu 会在 CSV 前打印 `==PROF== ...` 日志行，定位真正的表头再解析
    header_idx = next(
        (i for i, line in enumerate(lines) if "Kernel Name" in line), None
    )
    if header_idx is None:
        return {}

    rows = list(csv.DictReader(lines[header_idx:]))

    if not rows:
        return {}
    headers = [_norm_key(h) for h in rows[0].keys()]
    long_format = "metric_name" in headers and "metric_value" in headers

    result: Dict[str, Dict[str, object]] = {}
    if long_format:
        for row in rows:
            name = _pick(row, "kernel_name")
            metric = _pick(row, "metric_name")
            value = _to_float(_pick(row, "metric_value"))
            if not name or not metric:
                continue
            result.setdefault(name, {})[_norm_key(metric)] = value
    else:
        for row in rows:
            name = _pick(row, "kernel_name")
            if not name:
                continue
            entry: Dict[str, object] = {}
            for raw_key, raw_val in row.items():
                key = _norm_key(raw_key or "")
                if key in {"kernel_name", "id", "process_id", "process_name", "host_name"}:
                    continue
                value = _to_float(raw_val)
                if value is not None:
                    entry[key] = value
            if entry:
                result.setdefault(name, {}).update(entry)

    return result


def _pick(row: Dict[str, str], wanted: str) -> Optional[str]:
    for key, value in row.items():
        if key and _norm_key(key) == wanted:
            return value
    return None


def _find_metrics(entry: Dict[str, object]) -> Dict[str, Optional[float]]:
    """从一条 kernel 记录里取出三类核心指标（兼容长表/宽表的命名差异）。"""
    out: Dict[str, Optional[float]] = {"sm_pct": None, "dram_pct": None,
                                       "occupancy_pct": None, "duration_us": None}
    for key, value in entry.items():
        if not isinstance(value, (int, float)):
            continue
        if key.startswith(_norm_key(METRIC_SM)):
            out["sm_pct"] = float(value)
        elif key.startswith(_norm_key(METRIC_DRAM)):
            out["dram_pct"] = float(value)
        elif key.startswith(_norm_key(METRIC_OCCUPANCY)):
            out["occupancy_pct"] = float(value)
        elif key.startswith(_norm_key(METRIC_DURATION)):
            out["duration_us"] = float(value)
    return out


def _signature(name: str) -> str:
    """取最能区分 kernel 的长标识符（通常是模板参数）。"""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{11,}", name or "")
    if not tokens:
        return ""
    return max(tokens, key=len)


def match_kernel(kernel: Optional[str], op: str, table: Dict[str, Dict[str, object]]) -> Optional[str]:
    """把 finding 的 kernel/op 名匹配到 ncu 里的 kernel 名。

    同一内核在不同工具里名字不一致（ncu 丢命名空间、TraceLens 带完整参数列表），
    所以依次尝试：完全相等 -> 去命名空间/参数后的子串 -> 最长标识符（模板参数）匹配。
    """
    if not table:
        return None
    candidates: List[str] = []
    if kernel:
        candidates.append(kernel)
    candidates.append(op)

    for candidate in candidates:
        if candidate in table:
            return candidate

    def _clean(text: str) -> str:
        text = re.sub(r"\b(void|static|__global__)\b", " ", text)
        text = re.sub(r"\b[a-z_][a-z0-9_]*::", "", text)  # 去命名空间
        text = re.sub(r"\(.*\)$", "", text.strip())     # 去参数列表
        return re.sub(r"\s+", "", text)

    for candidate in candidates:
        probe = _clean(candidate)
        if not probe or len(probe) < 8:
            continue
        for name in table:
            if probe in _clean(name) or _clean(name) in probe:
                return name

    for candidate in candidates:
        sig = _signature(candidate)
        if not sig:
            continue
        for name in table:
            if sig in name or _signature(name) == sig:
                return name
    return None


def apply_to_finding(
    finding,
    table: Dict[str, Dict[str, object]],
    *,
    near_roofline_pct: float = 85.0,
    min_pct_e2e: float = 2.0,
) -> None:
    """把 ncu 证据写进 finding，并按需修正 bound_type / confidence。"""
    exact = finding.kernel if finding.kernel in table else None
    if exact is None:
        # 模板实例无法唯一对应：记录"有 ncu 数据但没匹配上"，不给结论
        candidates = [n for n in table if finding.kernel and finding.kernel in n]
        finding.evidence["ncu_match"] = {
            "matched": False,
            "candidates": len(candidates),
            "reason": "kernel 名在 trace 与 ncu 间不唯一",
        }
        if candidates:
            finding.evidence.setdefault("notes", []).append(
                f"ncu 有 {len(candidates)} 个同名模板实例，无法唯一定位，未采用其计数器"
            )
        return
    name = exact

    metrics = _find_metrics(table[name])
    if not any(v is not None for v in metrics.values()):
        return

    finding.evidence["ncu"] = {
        "kernel": name,
        "sm_throughput_pct": metrics["sm_pct"],
        "dram_throughput_pct": metrics["dram_pct"],
        "occupancy_pct": metrics["occupancy_pct"],
        "duration_us": metrics["duration_us"],
    }

    sm, dram = metrics["sm_pct"], metrics["dram_pct"]
    if sm is not None and dram is not None:
        ncu_bound = "compute" if sm >= dram else "memory"
        if abs(sm - dram) >= _BOUND_MARGIN and ncu_bound != finding.bound_type:
            conflict = f"trace 判定 {finding.bound_type}，ncu 判定 {ncu_bound}"
            finding.evidence.setdefault("notes", []).append(conflict)
            finding.bound_type = ncu_bound

    # 有硬件计数器才算 measured
    finding.confidence = "measured"

    # 用实测利用率作为 SOL：compute 看 SM，memory 看 DRAM；latency 不适用
    measured = None
    if finding.bound_type == "compute":
        measured = sm
    elif finding.bound_type == "memory":
        measured = dram
    if measured is not None:
        finding.sol_pct = float(min(measured, 100.0))
        finding.sol_pct_raw = float(measured)
        finding.evidence.setdefault("notes", []).append(
            f"SOL 取自 ncu 实测（{finding.bound_type} → {finding.sol_pct:.1f}%）"
        )

    # 时间来自 profiler 窗口的排除项，在有实测计数器后可以推翻
    if finding.excluded_reason == "profiled_timing":
        finding.excluded_reason = None
        finding.eligible = False
        sol = finding.sol_pct or 0.0
        if sol >= near_roofline_pct:
            finding.excluded_reason = "near_roofline"
            finding.recommended_action = "tune"
            finding.estimated_gain_pct = 0.0
            finding.impact_score = 0.0
        elif finding.pct_e2e < min_pct_e2e:
            finding.excluded_reason = "below_amdahl_threshold"
            finding.recommended_action = "tune"
            finding.estimated_gain_pct = 0.0
            finding.impact_score = 0.0
        else:
            finding.recommended_action = "tune"
            finding.suggested_owner = "kernel"
            finding.eligible = True
            gain = finding.pct_e2e * (1.0 - sol / 100.0)
            finding.estimated_gain_pct = gain
            finding.impact_score = gain
