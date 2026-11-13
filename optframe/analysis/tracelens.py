"""TraceLens 报表生成的外部进程封装。

只做三件事：调 CLI、收集产物路径、校验 sanity check。
不做任何解析（解析在 parse.py），保证"分析层只诊断不改代码"的定位。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CLI_NAME = "TraceLens_generate_perf_report_pytorch_inference"


class TraceLensError(RuntimeError):
    """TraceLens 执行失败或 sanity check 不通过。"""


@dataclass
class ReportArtifacts:
    """一次报表生成的产物。"""

    xlsx: Path
    csv_dir: Path
    stdout: str = ""
    sanity_lines: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)


def find_cli(explicit: Optional[str] = None) -> str:
    """定位 TraceLens CLI。

    顺序：显式参数 -> 环境变量 TRACELENS_CLI -> 当前解释器同目录 -> PATH。
    """
    if explicit:
        return explicit
    env = os.environ.get("TRACELENS_CLI")
    if env:
        return env

    sibling = Path(sys.executable).parent / CLI_NAME
    if sibling.exists():
        return str(sibling)

    found = shutil.which(CLI_NAME)
    if found:
        return found
    raise TraceLensError(
        f"{CLI_NAME} not found; install TraceLens or set TRACELENS_CLI"
    )


def tracelens_commit(source_dir: Optional[Path] = None) -> Optional[str]:
    """读取 TraceLens 源码目录的 commit（用于留痕）。"""
    if source_dir is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


def _parse_us(stdout: str, label: str) -> Optional[float]:
    match = re.search(rf"{re.escape(label)}\s*:?\s*([\d.]+)\s*µs", stdout)
    return float(match.group(1)) if match else None


def _sanity_verdict(stdout: str) -> tuple[bool, list[str], dict]:
    """按"时间覆盖率"判定，而不是死抠事件计数。

    真实 vLLM trace（含 CUDA graph）会少量 kernel 无法挂到算子树上：
    实测 12056 个 GPU 事件 / 11728 个已挂账（2.7%）。只要 kernel 时间
    覆盖了 computation_time，报表就是可用的；计数差超过 20% 才认为不可信。
    """
    warnings: list[str] = []
    stats: dict = {}

    comp = _parse_us(stdout, "computation_time (gpu_timeline)")
    ktime = _parse_us(stdout, "Kernel time (kernel_launchers)")
    ute = _parse_us(stdout, "Kernel time (unified_perf)")
    total = re.search(r"Total GPU events in tree\s*:?\s*(\d+)", stdout)
    accounted = re.search(r"Kernels accounted\s*:?\s*(\d+)", stdout)

    stats.update({
        "computation_us": comp,
        "kernel_launchers_us": ktime,
        "unified_perf_us": ute,
    })

    hard_fail = False
    if comp and ktime is not None and ktime + 1e-6 < comp:
        hard_fail = True
        warnings.append(f"kernel_launchers 时间({ktime:.0f}µs) < computation_time({comp:.0f}µs)")
    if comp and ute is not None and ute + 1e-6 < comp:
        hard_fail = True
        warnings.append(f"unified_perf 时间({ute:.0f}µs) < computation_time({comp:.0f}µs)")

    if total and accounted:
        total_n, acc_n = int(total.group(1)), int(accounted.group(1))
        missing = max(total_n - acc_n, 0)
        frac = missing / total_n if total_n else 0.0
        stats.update({"gpu_events": total_n, "accounted": acc_n,
                      "unaccounted_pct": round(frac * 100, 2)})
        if frac > 0.20:
            hard_fail = True
            warnings.append(f"未挂账 GPU 事件 {missing}/{total_n}（{frac*100:.1f}%）超过 20%")
        elif missing:
            warnings.append(f"未挂账 GPU 事件 {missing}/{total_n}（{frac*100:.1f}%），不影响时间覆盖")

    return (not hard_fail), warnings, stats


def run_perf_report(
    trace_path: str | Path,
    arch_json: str | Path,
    workdir: str | Path,
    *,
    cli: Optional[str] = None,
    timeout_s: int = 3600,
    extra_args: Optional[List[str]] = None,
) -> ReportArtifacts:
    """生成 TraceLens 性能报表，失败抛 TraceLensError。"""
    trace_path = Path(trace_path).expanduser().resolve()
    arch_json = Path(arch_json).expanduser().resolve()
    workdir = Path(workdir).expanduser().resolve()

    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")
    if not arch_json.exists():
        raise FileNotFoundError(f"arch json not found: {arch_json}")

    csv_dir = workdir / "csvs"
    xlsx = workdir / "perf_report.xlsx"
    csv_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        find_cli(cli),
        "--profile_json_path", str(trace_path),
        "--gpu_arch_json_path", str(arch_json),
        "--output_xlsx_path", str(xlsx),
        "--output_csvs_dir", str(csv_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TraceLensError(f"TraceLens timed out after {timeout_s}s") from exc

    stdout = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(stdout.strip().splitlines()[-15:])
        raise TraceLensError(f"TraceLens exited {proc.returncode}\n{tail}")

    sanity = [ln.strip() for ln in stdout.splitlines() if "PASS" in ln or "FAIL" in ln]
    ok, warnings, stats = _sanity_verdict(stdout)
    if not ok:
        raise TraceLensError("TraceLens sanity check failed:\n" + "\n".join(warnings))

    if not csv_dir.exists() or not any(csv_dir.glob("*.csv")):
        raise TraceLensError(f"TraceLens produced no CSV in {csv_dir}")

    return ReportArtifacts(xlsx=xlsx, csv_dir=csv_dir, stdout=stdout,
                           sanity_lines=sanity, warnings=warnings, stats=stats)
