"""阶段3 主体：amdahl.json -> task.toml + reference.py + judge.py。

用法：
    python -m optframe.inventory.trace_to_tasks \
        --amdahl runs/stage3_amdahl/amdahl.json \
        --outdir runs/stage3_tasks

产物：
    runs/stage3_tasks/
      index.json
      <op>__<shape>/task.toml, reference.py, judge.py, notes.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .codegen import SUPPORTED_DTYPES, judge_source, lookup, normalize_op, reference_source
from .escalation import decide, reason_text
from .taskspec import (
    ARCH_BY_HARDWARE,
    TaskSpec,
    arch_for_hardware,
    parse_shapes,
    sanitize_name,
    shape_digest,
)


class GenerationError(RuntimeError):
    """生成失败（不是"跳过"）。"""


def _notes_markdown(task: Dict[str, Any], spec: TaskSpec, comments: List[str]) -> str:
    lines = [
        f"# 任务说明：{spec.name}",
        "",
        f"- 原始算子：`{spec.op}`",
        f"- 类别：{spec.category} ｜ 动作：{spec.action}",
        f"- 形状：`{spec.shapes}` ｜ dtype：{spec.dtype} ｜ arch：{spec.arch}",
        f"- 目标加速比：{spec.target_ratio:.3f}×（依据：{task.get('speedup_basis')}）",
        f"- 端到端占比：{task.get('pct_e2e')}% ｜ roofline SOL：{task.get('sol_pct')}",
        f"- 允许升级复杂档：{'是' if spec.allow_complex else '否'}",
    ]
    if spec.escalation_reasons:
        lines.append("- 升级原因：")
        for reason in spec.escalation_reasons:
            lines.append(f"  - `{reason}` — {reason_text(reason)}")
    lines.extend([
        "",
        "## 验收口径",
        "",
        "- 正确性：`judge.py` 对拍 `reference.py`，容差按 dtype；",
        "- 性能：device 计时，warmup 10 / repeat 50 取中位数；",
        "- 目标：相对 `baseline=torch` 达到 `target_ratio`；低于 1.02× 视为噪声。",
        "",
        "## 生成备注",
        "",
    ])
    lines.extend(f"- {c}" for c in comments) if comments else lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def build_spec(
    task: Dict[str, Any],
    *,
    hardware: Optional[str],
    arch_override: Optional[str],
) -> tuple[Optional[TaskSpec], List[str], Optional[str]]:
    """把一条 amdahl task 转成 TaskSpec。

    返回 (spec, comments, skip_reason)。skip_reason 非空表示这条任务不生成。
    """
    op = str(task.get("op", ""))
    shapes = parse_shapes(str(task.get("shape", "")))
    comments: List[str] = []

    if not shapes:
        return None, comments, "shape_unparsable"

    # 两输入 GEMM：看 K/N（输出/归约维），不看 M（batch）；K 或 N 太小属 launch-bound
    if len(shapes) >= 2 and min(shapes[-1]) < 128:
        return None, comments, "tiny_shape"

    template = lookup(op)
    if template is None:
        return None, comments, "unsupported_op"
    if len(shapes) < template.n_inputs:
        return None, comments, "missing_inputs"

    dtype = str(task.get("dtype") or "unknown")
    if dtype == "unknown":
        return None, comments, "unknown_dtype"
    if "+" in dtype:
        comments.append(f"复合 dtype `{dtype}` 取第一段作为计算 dtype")
        dtype = dtype.split("+")[0]
    if dtype not in SUPPORTED_DTYPES:
        return None, comments, "unsupported_dtype"

    arch = arch_override
    if arch is None:
        # 优先用实际设备：换卡后任务里的 arch 会过期，硬编码映射表也会漏新卡
        try:
            from ..contracts.cuda_loader import device_arch

            detected = device_arch()
        except Exception:  # noqa: BLE001 - 无 CUDA 时退回映射表
            detected = None
        if detected:
            arch = detected
            comments.append(f"arch 取自当前设备: {arch}")
        else:
            arch = arch_for_hardware(hardware)
    if arch_override is None and hardware and not any(
        hardware.upper().replace("NVIDIA", "").strip().startswith(name.upper())
        for name in ARCH_BY_HARDWARE
    ):
        comments.append(f"硬件 `{hardware}` 不在映射表中，arch={arch}，请人工确认")

    name = f"{sanitize_name(op)}__{shape_digest(shapes)}"
    decision = decide(task)

    spec = TaskSpec(
        name=name,
        op=op,
        arch=arch,
        dtype=dtype,
        shapes=shapes,
        target_ratio=float(task.get("target_ratio") or task.get("target_speedup") or 1.0),
        category=str(task.get("category") or "other"),
        action=str(task.get("action") or "tune"),
        sol_pct=task.get("sol_pct"),
        pct_e2e=task.get("pct_e2e"),
        allow_complex=decision.allow_complex,
        escalation_reasons=decision.reasons,
        runtime_rule=decision.runtime_rule,
        meta={
            "source_run_id": task.get("_source_run_id"),
            "speedup_basis": task.get("speedup_basis"),
            "rank": task.get("rank"),
        },
    )
    return spec, comments, None


def write_task(
    task: Dict[str, Any],
    spec: TaskSpec,
    outdir: Path,
    comments: List[str],
) -> Path:
    task_dir = outdir / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)

    ref = reference_source(spec.op, spec.shapes, spec.dtype)
    if ref is None:
        raise GenerationError(f"reference 生成失败: {spec.op}")

    (task_dir / "task.toml").write_text(spec.to_toml(), encoding="utf-8")
    (task_dir / "reference.py").write_text(ref, encoding="utf-8")
    (task_dir / "judge.py").write_text(judge_source(spec.dtype), encoding="utf-8")
    (task_dir / "notes.md").write_text(
        _notes_markdown(task, spec, comments), encoding="utf-8"
    )
    return task_dir


def self_check(task_dir: Path, timeout_s: int = 300) -> Dict[str, Any]:
    """在 CPU 上跑一遍 judge 自检，证明生成的代码真的能执行。"""
    proc = subprocess.run(
        [sys.executable, "judge.py"],
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip()[-500:],
        "stderr": (proc.stderr or "").strip()[-800:],
    }


def run(
    amdahl_path: Path,
    outdir: Path,
    *,
    hardware_override: Optional[str] = None,
    arch_override: Optional[str] = None,
    do_self_check: bool = True,
) -> Dict[str, Any]:
    payload = json.loads(amdahl_path.read_text(encoding="utf-8"))
    tasks: List[Dict[str, Any]] = payload.get("tasks") or []
    hardware = hardware_override or (payload.get("target") or {}).get("hardware")
    run_id = payload.get("source_run_id")

    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for task in tasks:
        task = dict(task)
        task["_source_run_id"] = run_id
        spec, comments, skip_reason = build_spec(
            task, hardware=hardware, arch_override=arch_override
        )
        if spec is None:
            skipped.append({
                "op": task.get("op"),
                "shape": task.get("shape"),
                "reason": skip_reason,
            })
            continue

        try:
            task_dir = write_task(task, spec, outdir, comments)
        except GenerationError as exc:
            failures.append({"op": spec.op, "error": str(exc)})
            continue

        entry = spec.to_dict()
        entry["dir"] = str(task_dir)
        entry["comments"] = comments
        if do_self_check:
            check = self_check(task_dir)
            entry["self_check"] = {"ok": check["ok"], "returncode": check["returncode"]}
            if not check["ok"]:
                entry["self_check"]["stderr"] = check["stderr"]
                failures.append({
                    "op": spec.op,
                    "error": "self_check_failed",
                    "stderr": check["stderr"],
                })
        written.append(entry)

    index = {
        "schema_version": "1.0",
        "source_amdahl": str(amdahl_path.resolve()),
        "source_run_id": run_id,
        "target": payload.get("target") or {},
        "tasks": written,
        "skipped": skipped,
        "failures": failures,
    }
    (outdir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="optframe.inventory.trace_to_tasks",
        description="amdahl.json -> task.toml + reference.py + judge.py",
    )
    parser.add_argument("--amdahl", required=True, help="阶段3 Amdahl 输出的 amdahl.json")
    parser.add_argument("--outdir", default="runs/stage3_tasks", help="输出目录")
    parser.add_argument("--hardware", default=None, help="覆盖硬件名（默认取 amdahl.json）")
    parser.add_argument("--arch", default=None, help="直接指定 arch（如 sm89），优先于硬件映射")
    parser.add_argument("--no-self-check", action="store_true", help="跳过 CPU 自检")
    args = parser.parse_args(argv)

    amdahl_path = Path(args.amdahl)
    if not amdahl_path.exists():
        print(f"[FAIL] 找不到 {amdahl_path}", file=sys.stderr)
        return 2

    index = run(
        amdahl_path,
        Path(args.outdir),
        hardware_override=args.hardware,
        arch_override=args.arch,
        do_self_check=not args.no_self_check,
    )

    print(f"[1/2] 生成 {len(index['tasks'])} 个任务，跳过 {len(index['skipped'])}，"
          f"失败 {len(index['failures'])}")
    for entry in index["tasks"]:
        flag = "✅" if entry.get("self_check", {}).get("ok", True) else "❌"
        complex_mark = "复杂档可用" if entry["allow_complex"] else "仅简单档"
        print(f"      {flag} {entry['name']}  target={entry['target_ratio']}×  ({complex_mark})")
    for item in index["skipped"]:
        print(f"      ⏭  {item['op']} ({item['shape']}): {item['reason']}")
    for item in index["failures"]:
        print(f"      ❌ {item['op']}: {item['error']}", file=sys.stderr)

    print(f"[2/2] 产物: {Path(args.outdir) / 'index.json'}")
    return 0 if not index["failures"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
