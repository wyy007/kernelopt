"""统一评测：正确性 -> device 计时 -> 比值（双档方案 §4、§5）。

铁律：任何档位自报的性能数字都不采信，一律走这里复测。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import torch

from .schema import Candidate, Result
from .source_guard import check_candidate
from .cuda_loader import compile_cuda, device_arch, validate_cuda_source
from .task import Task, load_module, resolve_callable
from .timing import bootstrap_median_ci, time_ms, time_pair_samples


@dataclass
class Baseline:
    outputs: List[Any] = field(default_factory=list)
    ms: float = 0.0
    #: 基线可调用对象，用于交错 A/B 计时（可选）
    fn: Optional[Any] = None


def measure_baseline(
    task: Task,
    inputs: List[torch.Tensor],
    *,
    warmup: int = 10,
    repeat: int = 50,
) -> Baseline:
    """以 reference(PyTorch) 为基线，产出真值输出与中位耗时。"""
    fn = resolve_callable(task.reference_module, task.reference_spec)
    outputs = fn(*inputs)
    ms = time_ms(lambda: fn(*inputs), warmup=warmup, repeat=repeat)
    return Baseline(outputs=outputs, ms=ms, fn=lambda: fn(*inputs))


def evaluate_candidate(
    task: Task,
    candidate: Candidate,
    baseline: Baseline,
    inputs: List[torch.Tensor],
    workdir: str | Path,
    *,
    warmup: int = 10,
    repeat: int = 50,
    forbid_torch: bool = True,
) -> Result:
    """完整评测一个候选，返回统一口径的 Result。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if candidate.language == "cuda":
        verdict = validate_cuda_source(candidate.source)
    else:
        verdict = check_candidate(candidate.source, candidate.language, forbid_torch)
    if verdict:
        return Result(status="rejected", error=verdict,
                      ref_ms=baseline.ms,
                      artifacts={"candidate_sha": candidate.sha})

    if candidate.language == "cuda":
        build_dir = workdir / "cuda_build"
        # 以实际设备为准：任务里的 arch 是生成时的基线，换卡后不一定适用
        arch = device_arch() or task.arch
        notes = []
        if arch != task.arch:
            notes.append(f"task.arch={task.arch} 与当前设备 {arch} 不一致，按设备编译")
        module, err = compile_cuda(candidate.source, arch, build_dir)
        if module is None:
            return Result(status="compile_error", error=(err or "compile failed")[-3000:],
                          ref_ms=baseline.ms,
                          artifacts={"candidate_sha": candidate.sha,
                                     "build_dir": str(build_dir), "arch": arch})
        path = build_dir / f"fkcuda_{candidate.sha[:12]}.cu"
    else:
        path = workdir / f"cand_{candidate.sha[:12]}.py"
        path.write_text(candidate.source, encoding="utf-8")

        # 每次评测都重新加载，避免 sys.modules 缓存旧版本。
        # 候选模块在 import 阶段就可能抛任何异常（缺依赖、语法通过但执行失败），
        # 必须归为 import_error，不能让它掀掉整个进程。
        try:
            module = load_module(path, alias="cand")
        except Exception as exc:  # noqa: BLE001
            return Result(
                status="import_error",
                error=f"{type(exc).__name__}: {exc}",
                ref_ms=baseline.ms,
                artifacts={
                    "candidate_sha": candidate.sha,
                    "source": str(path),
                    "traceback": traceback.format_exc()[-1200:],
                },
            )

    run = getattr(module, "run", None)
    if not callable(run):
        return Result(status="import_error", error="candidate has no callable run()",
                      ref_ms=baseline.ms,
                      artifacts={"candidate_sha": candidate.sha, "source": str(path)})

    # 正确性：用输入副本，防止候选原地修改污染后续计时
    try:
        cand_out = run(*[t.clone() for t in inputs])
    except Exception as exc:  # noqa: BLE001 - 候选任意异常都要记录
        return Result(
            status="runtime_error",
            error=f"{type(exc).__name__}: {exc}",
            ref_ms=baseline.ms,
            artifacts={"candidate_sha": candidate.sha, "source": str(path),
                       "traceback": traceback.format_exc()[-1500:]},
        )

    try:
        check = task.judge_check(baseline.outputs, cand_out)
    except Exception as exc:  # noqa: BLE001
        return Result(status="runtime_error", error=f"judge failed: {exc}",
                      ref_ms=baseline.ms,
                      artifacts={"candidate_sha": candidate.sha, "source": str(path)})

    if not check.get("ok", False):
        return Result(
            correct=False,
            status="incorrect",
            error=str(check.get("error", "judge mismatch")),
            max_abs_err=check.get("max_abs_err"),
            ref_ms=baseline.ms,
            artifacts={"candidate_sha": candidate.sha, "source": str(path)},
        )

    try:
        if baseline.fn is not None:
            ref_samples, cand_samples = time_pair_samples(
                baseline.fn, lambda: run(*inputs), warmup=warmup, repeat=repeat
            )
            ref_ms = float(median(ref_samples))
            ms = float(median(cand_samples))
            paired = [
                r / c for r, c in zip(ref_samples, cand_samples) if c > 0
            ]
            ratio_low, ratio_high = (
                bootstrap_median_ci(paired) if len(paired) >= 10 else (None, None)
            )
            pairs = len(paired)
            # 点估计也取配对比值的中位数，与 CI 同口径
            ratio_point = float(median(paired)) if paired else None
        else:
            ref_ms = baseline.ms
            ms = time_ms(lambda: run(*inputs), warmup=warmup, repeat=repeat)
            ratio_low, ratio_high, pairs = None, None, 0
            ratio_point = None
    except Exception as exc:  # noqa: BLE001
        return Result(status="runtime_error", error=f"timing failed: {exc}",
                      correct=True, ref_ms=baseline.ms,
                      artifacts={"candidate_sha": candidate.sha, "source": str(path)})

    ratio = ratio_point if ratio_point is not None else ((ref_ms / ms) if ms > 0 else None)
    return Result(
        correct=True,
        device_ms=ms,
        ref_ms=ref_ms,
        ratio=ratio,
        ratio_low=ratio_low,
        ratio_high=ratio_high,
        pairs=pairs,
        max_abs_err=check.get("max_abs_err"),
        status="ok",
        artifacts={
            "candidate_sha": candidate.sha,
            "source": str(path),
            "arch": task.arch if candidate.language != "cuda" else arch,
            "arch_notes": notes if candidate.language == "cuda" else [],
        },
    )
