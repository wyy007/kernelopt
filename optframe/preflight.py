"""换卡/换环境自检：`python -m optframe.preflight`。

回答一个问题：**这台机器现在能不能跑，哪些结论已经过期需要重跑。**
把它做成命令，是因为上一轮换卡（L20 sm_89 → RTX PRO 5000 sm_120）的代价
是"内核静默不执行、输出全 0、没有任何报错"——靠编译报错试错太贵。

判定口径：
    PASS  可以直接用
    WARN  能跑，但数据/知识需要重测（结论可信度下降）
    FAIL  会跑错或跑不了（必须处理）
    SKIP  缺少前置条件，未检查
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts.cuda_loader import _cc, device_arch, find_torch_cudart

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

ARCH_JSON_DEFAULT = "arch/RTX_PRO_5000.json"

_GENCODE = re.compile(r"-gencode[= ]arch=compute_(\d+)")
_SM_TAG = re.compile(r"code=sm_(\d+)")


@dataclass
class Finding:
    check: str
    status: str
    detail: str
    hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "status": self.status,
                "detail": self.detail, "hint": self.hint}


@dataclass
class Report:
    device: Dict[str, Any] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)

    def add(self, check: str, status: str, detail: str, hint: str = "") -> None:
        self.findings.append(Finding(check, status, detail, hint))

    @property
    def worst(self) -> str:
        for level in (FAIL, WARN):
            if any(f.status == level for f in self.findings):
                return level
        return PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "worst": self.worst,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------- 探测

def parse_version(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """`12.8` / `release 13.0, V13.0.88` -> (12, 8) / (13, 0)。"""
    if not text:
        return None
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def nvcc_version(nvcc: str = "nvcc") -> Optional[Tuple[int, int]]:
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                             timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_version(out.stdout)


def probe_device() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"torch import failed: {exc}"
        return info
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        info["error"] = "CUDA unavailable"
        return info
    info.update({
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "arch": device_arch(),
        "memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
        ),
    })
    try:
        import triton

        target = triton.runtime.driver.active.get_current_target()
        info["triton_arch"] = f"sm{target.arch}"
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------- 各项检查

def check_device(info: Dict[str, Any], report: Report) -> None:
    if not info.get("available"):
        report.add("device", FAIL, info.get("error", "CUDA unavailable"),
                   "确认驱动与 GPU 可见性（nvidia-smi）")
        return
    report.add(
        "device", PASS,
        f"{info['name']} | {info.get('arch')} | {info.get('memory_gib')} GiB | "
        f"torch {info.get('torch')} (cu{info.get('torch_cuda')})",
    )


def check_toolkit(info: Dict[str, Any], report: Report) -> None:
    torch_cuda = parse_version(str(info.get("torch_cuda") or ""))
    nvcc = nvcc_version()
    if nvcc is None or torch_cuda is None:
        report.add("toolkit", SKIP, "nvcc 或 torch CUDA 版本未知",
                   "确认 nvcc 在 PATH、torch 带 CUDA")
        return
    if nvcc[0] == torch_cuda[0]:
        report.add("toolkit", PASS, f"nvcc {nvcc[0]}.{nvcc[1]} 与 torch cu{torch_cuda[0]}.{torch_cuda[1]} 同大版本")
        return
    cudart = find_torch_cudart()
    if cudart is None:
        report.add(
            "toolkit", FAIL,
            f"nvcc 是 CUDA {nvcc[0]}.x，torch 是 cu{torch_cuda[0]}.x，且找不到 torch 自带 cudart",
            "内核会出现 'no kernel image' 或静默不执行；装匹配的 toolkit",
        )
        return
    report.add(
        "toolkit", PASS,
        f"nvcc {nvcc[0]}.{nvcc[1]} vs torch cu{torch_cuda[0]}.{torch_cuda[1]}（跨大版本），"
        f"将用 torch 自带 cudart 做兼容链接",
        "cuda_loader 已自动处理；若手工编译需自己处理双 runtime",
    )


def check_triton(info: Dict[str, Any], report: Report) -> None:
    if not info.get("available"):
        report.add("triton", SKIP, "无可用设备")
        return
    if "triton_arch" not in info:
        report.add("triton", WARN, "triton 不可用或无法探测目标架构",
                   "简单档需要 triton")
        return
    if info["triton_arch"] != info.get("arch"):
        report.add("triton", WARN,
                   f"triton 目标 {info['triton_arch']} 与设备 {info.get('arch')} 不一致",
                   "检查 triton 版本")
        return
    report.add("triton", PASS, f"triton {info.get('triton')} 目标 {info['triton_arch']}")


def scan_arch_jsons(arch_dir: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    if not arch_dir.exists():
        return out
    for path in sorted(arch_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append((path, str(data.get("name") or "unknown")))
    return out


def check_arch_baseline(info: Dict[str, Any], arch_dir: Path, report: Report) -> None:
    if not info.get("available"):
        report.add("arch.json", SKIP, "无可用设备")
        return
    entries = scan_arch_jsons(arch_dir)
    if not entries:
        report.add("arch.json", FAIL, f"{arch_dir} 下没有 arch.json",
                   "先跑 microbench 生成当前卡的 roofline 基线")
        return

    device_name = str(info.get("name") or "")
    matched = [p for p, name in entries if name and name.split()[0] in device_name]
    if matched:
        report.add("arch.json", PASS, f"基线匹配: {matched[-1].name}")
        return

    names = ", ".join(f"{p.name}={name}" for p, name in entries)
    report.add(
        "arch.json", WARN,
        f"现有基线 {names} 与当前设备 {device_name} 不匹配",
        "阶段2/3 的 roofline SOL 会失真，需重跑 microbench",
    )


def iter_task_tomls(dirs: Sequence[Path]) -> Iterable[Path]:
    seen = set()
    for base in dirs:
        if not base.exists():
            continue
        for path in base.rglob("task.toml"):
            if path in seen:
                continue
            seen.add(path)
            yield path


def task_arch(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*arch\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
    return match.group(1) if match else None


def check_tasks(info: Dict[str, Any], task_dirs: Sequence[Path], report: Report) -> None:
    arch = info.get("arch")
    paths = list(iter_task_tomls(task_dirs))
    if not paths:
        report.add("tasks", SKIP, "没有找到 task.toml")
        return
    if not arch:
        report.add("tasks", SKIP, "无可用设备，跳过架构比对")
        return

    from collections import Counter

    counts = Counter()
    mismatched: List[str] = []
    for path in paths:
        declared = task_arch(path)
        counts[declared or "unknown"] += 1
        if declared and declared != arch:
            mismatched.append(f"{path.parent.name}({declared})")

    if not mismatched:
        report.add("tasks", PASS, f"{len(paths)} 个任务，arch 与设备一致（{arch}）")
        return

    sample = ", ".join(mismatched[:3])
    report.add(
        "tasks", WARN,
        f"{len(mismatched)}/{len(paths)} 个任务 arch 与设备 {arch} 不一致：{sample}",
        "评测会按实际设备编译（已自动覆盖），但 SOL 排序仍是旧卡的，建议重生成任务",
    )


def scan_hardcoded_gencode(paths: Sequence[Path], limit: int = 400) -> List[Tuple[str, str]]:
    """找出源码里写死的 -gencode / sm_XX。返回 [(文件, 架构码)]。"""
    found: List[Tuple[str, str]] = []
    seen = 0
    for base in paths:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if seen >= limit:
                return found
            if not path.is_file() or path.suffix not in {".cu", ".py", ".toml", ".sh"}:
                continue
            seen += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in _GENCODE.finditer(text):
                found.append((str(path), match.group(1)))
            for match in _SM_TAG.finditer(text):
                found.append((str(path), match.group(1)))
    return found


def check_hardcoded(info: Dict[str, Any], paths: Sequence[Path], report: Report) -> None:
    arch = info.get("arch")
    hits = scan_hardcoded_gencode(paths)
    if not hits:
        report.add("hardcoded_gencode", PASS, "未发现写死的 -gencode/sm_XX")
        return
    if not arch:
        report.add("hardcoded_gencode", SKIP, f"发现 {len(hits)} 处硬编码，但无设备可比对")
        return

    cc = _cc(arch).replace(".", "")
    wrong = [(p, code) for p, code in hits if code != cc]
    if not wrong:
        report.add("hardcoded_gencode", PASS, f"{len(hits)} 处硬编码与设备 {arch} 一致")
        return
    sample = ", ".join(sorted({f"{Path(p).name}→sm{code}" for p, code in wrong})[:4])
    report.add(
        "hardcoded_gencode", WARN,
        f"{len(wrong)} 处硬编码与设备 {arch} 不一致：{sample}",
        "cuda_loader 会剥离后按设备重编；手工编译需自行处理",
    )


def check_hw_kb(info: Dict[str, Any], forge_root: Optional[Path], report: Report) -> None:
    arch = info.get("arch")
    if forge_root is None or not forge_root.exists():
        report.add("hw_kb", SKIP, "未提供 ForgeKernel 根目录")
        return
    kb = forge_root / "hw_kb" / "cuda"
    if not kb.exists():
        report.add("hw_kb", SKIP, f"{kb} 不存在")
        return
    if not arch:
        report.add("hw_kb", SKIP, "无可用设备")
        return
    target = kb / arch.replace("sm_", "sm")
    if target.exists():
        report.add("hw_kb", PASS, f"复杂档知识库覆盖 {target.name}")
        return
    available = ", ".join(sorted(p.name for p in kb.iterdir() if p.is_dir()))
    report.add(
        "hw_kb", WARN,
        f"复杂档缺 {arch} 的知识库（现有: {available}）",
        "复杂档 agent 会现场摸索新架构，耗时与质量都会下降",
    )


# ---------------------------------------------------------------- 主流程


def probe_live_peak_tflops(size: int = 8192, reps: int = 20) -> Optional[float]:
    """实测当前条件下的 cuBLAS bf16 GEMM 峰值（TFLOPS）。

    arch.json 记的是"空闲时可达峰值"，排队/验收却常在争用条件下做——实测曾差 2.4 倍
    （218 vs 90），导致 SOL 被系统性低估、排出一堆打不动的任务。
    """
    import torch

    if not torch.cuda.is_available():
        return None
    torch.manual_seed(0)
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)

    def run():
        return a @ b

    for _ in range(5):
        run()
    torch.cuda.synchronize()

    best = 0.0
    for _ in range(max(reps, 1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if ms > 0:
            best = max(best, 2.0 * size ** 3 / (ms / 1000.0) / 1e12)
    return round(best, 1)


def check_peak_baseline(info: Dict[str, Any], report: Report, *, skip: bool = False,
                        arch_json: str = ARCH_JSON_DEFAULT) -> None:
    """对比 arch.json 峰值与当前实测峰值，偏差过大就告警。"""
    if skip:
        report.add("peak_baseline", SKIP, "--skip-peak-probe")
        return
    if not info.get("available"):
        report.add("peak_baseline", SKIP, "无可用设备")
        return

    live = probe_live_peak_tflops()
    if live is None or live <= 0:
        report.add("peak_baseline", SKIP, "实测失败（torch/cuda 不可用）")
        return

    try:
        data = json.loads(Path(arch_json).read_text(encoding="utf-8"))
        recorded = float(data["max_achievable_tflops"]["matrix_bf16"])
    except Exception:  # noqa: BLE001
        report.add("peak_baseline", SKIP, f"{arch_json} 不可读或缺 matrix_bf16")
        return

    ratio = recorded / live if live else float("inf")
    detail = f"arch.json {recorded:.1f} vs 实测 {live:.1f} TFLOPS（{ratio:.2f}×）"
    if ratio > 1.3:
        report.add("peak_baseline", WARN, detail,
                   "SOL 被系统性低估；排队/验收请改用同条件 cuBLAS 基线")
    elif ratio < 0.77:
        report.add("peak_baseline", WARN, detail, "实测高于记录值，arch.json 需要更新")
    else:
        report.add("peak_baseline", PASS, detail)


def run_checks(
    *,
    arch_dir: Path = Path("arch"),
    task_dirs: Optional[Sequence[Path]] = None,
    forge_root: Optional[Path] = Path("/model/wyy/ka_eval/ForgeKernel"),
    scan_dirs: Optional[Sequence[Path]] = None,
    skip_peak_probe: bool = False,
    arch_json: str = ARCH_JSON_DEFAULT,
) -> Report:
    info = probe_device()
    report = Report(device=info)

    check_device(info, report)
    check_toolkit(info, report)
    check_triton(info, report)
    check_arch_baseline(info, arch_dir, report)
    check_peak_baseline(info, report, skip=skip_peak_probe, arch_json=arch_json)

    task_dirs = list(task_dirs or [Path("runs/stage3_tasks")])
    check_tasks(info, task_dirs, report)

    scan = list(scan_dirs or task_dirs)
    if forge_root and forge_root.exists():
        scan.append(forge_root / "tasks")
    check_hardcoded(info, scan, report)
    check_hw_kb(info, forge_root, report)

    return report


def render(report: Report) -> str:
    icons = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "⏭ "}
    lines = [
        "preflight — 环境与基线自检",
        "=" * 60,
    ]
    for finding in report.findings:
        lines.append(f"{icons.get(finding.status, '?')} [{finding.check}] {finding.detail}")
        if finding.hint:
            lines.append(f"        → {finding.hint}")
    lines.append("=" * 60)
    lines.append(f"结论: {report.worst}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="optframe.preflight")
    parser.add_argument("--arch-dir", default="arch", help="roofline 基线目录")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="任务目录（可多个，默认 runs/stage3_tasks）")
    parser.add_argument("--forge-root", default="/model/wyy/ka_eval/ForgeKernel")
    parser.add_argument("--json", default=None, help="把结果写成 JSON")
    parser.add_argument("--strict", action="store_true", help="WARN 也返回非零")
    parser.add_argument("--skip-peak-probe", action="store_true",
                        help="跳过峰值实测（默认会测一次 cuBLAS GEMM）")
    parser.add_argument("--arch-json", default=ARCH_JSON_DEFAULT,
                        help="峰值基线文件")
    args = parser.parse_args(argv)

    report = run_checks(
        arch_dir=Path(args.arch_dir),
        task_dirs=[Path(p) for p in args.tasks] if args.tasks else None,
        forge_root=Path(args.forge_root) if args.forge_root else None,
        skip_peak_probe=args.skip_peak_probe,
        arch_json=args.arch_json,
    )
    print(render(report))

    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if report.worst == FAIL:
        return 1
    if args.strict and report.worst == WARN:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
