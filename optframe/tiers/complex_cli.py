"""复杂档 CLI：prepare -> run -> collect -> reevaluate。

示例：
  # 只做任务翻译（不跑 ForgeKernel，验证格式）
  python -m optframe.tiers.complex_cli --task runs/stage3_tasks/silu__2048x2048 --prepare-only

  # 复测已有 run 的产物（不重跑生成）
  python -m optframe.tiers.complex_cli --task ... --artifact .artifacts/silu_xxx --skip-run

  # 完整外挂执行（需要 ForgeKernel 的 agent 桥可用，预算内可能跑很久）
  python -m optframe.tiers.complex_cli --task ... --run --budget-hours 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..contracts.evaluator import measure_baseline
from ..contracts.task import load_task
from .complex_forge import (
    ForgeConfig,
    ForgeError,
    collect,
    prepare,
    reevaluate,
    run,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="optframe.tiers.complex_cli")
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--forge-root", default="/model/wyy/ka_eval/ForgeKernel")
    parser.add_argument("--python", default="/model/wyy/forge-venv/bin/python")
    parser.add_argument("--budget-hours", type=float, default=0.2)
    parser.add_argument("--max-meta", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="真正执行 forge run")
    parser.add_argument("--artifact", default=None, help="已有 run 目录，跳过创建")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--stub-cmd", default=None,
                        help="测试用：JSON 数组形式的替代命令")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args(argv)

    task = load_task(Path(args.task))
    workspace = Path(args.workspace) if args.workspace else Path("runs") / "stage4_complex" / task.name
    config = ForgeConfig(
        forge_root=Path(args.forge_root),
        python=args.python,
        budget_hours=args.budget_hours,
        max_meta=args.max_meta,
        create_run=args.run and not args.artifact,
    )

    prepared = prepare(task, workspace, config)
    print(f"[1/4] prepared: {prepared.task_dir}")
    if args.prepare_only:
        return 0

    artifact = Path(args.artifact) if args.artifact else None
    forge_run = None
    if not args.skip_run:
        stub = json.loads(args.stub_cmd) if args.stub_cmd else None
        forge_run = run(prepared, config, artifact_dir=artifact, stub_cmd=stub)
        print(f"[2/4] run: ok={forge_run.ok} rc={forge_run.returncode} "
              f"artifact={forge_run.artifact_dir} {forge_run.elapsed_s:.0f}s")
        if not forge_run.ok and forge_run.returncode != 2:
            print(forge_run.log_tail[-1200:], file=sys.stderr)
            return 3
        artifact = forge_run.artifact_dir or artifact
    if artifact is None:
        print("[FAIL] 没有可用的 artifact 目录", file=sys.stderr)
        return 2

    try:
        candidate = collect(artifact)
    except ForgeError as exc:
        print(f"[FAIL] collect: {exc}", file=sys.stderr)
        return 3
    print(f"[3/4] collected: {candidate.meta.get('kernel_path')} "
          f"forge_ratio={candidate.meta.get('forge_ratio')}")

    inputs = task.make_inputs(device="cuda")
    baseline = measure_baseline(task, inputs, warmup=args.warmup, repeat=args.repeat)
    result = reevaluate(task, candidate, baseline, inputs, workspace / "reeval",
                        warmup=args.warmup, repeat=args.repeat)

    print(f"[4/4] reevaluated: status={result.status} correct={result.correct} "
          f"ratio={result.ratio} ci_low={result.ratio_low} "
          f"forge_claimed={candidate.meta.get('forge_ratio')}")
    (workspace / "complex_result.json").write_text(
        json.dumps({
            "task": task.name,
            "run": forge_run.to_dict() if forge_run else None,
            "candidate": candidate.to_dict(),
            "result": result.to_dict(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if result.correct else 4


if __name__ == "__main__":
    raise SystemExit(main())
