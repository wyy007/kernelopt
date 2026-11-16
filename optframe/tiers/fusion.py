"""融合档编排（当前只支持简单档）。

职责边界（见 docs/modes.md）：
- 本模块不含任何单算子逻辑，也不被 `tiers/simple.py` 引用；
- 复用 `run_simple_tier` 的 best-of-N / 重试 / 双门槛 / CI 判定，只替换"任务语义"和收益归因；
- 复杂档（CUDA/CuTe）暂不支持；遇到 `allow_complex=True` 的融合任务直接拒绝，避免半成品路径。

收益口径（与单算子一致）：

    贡献(% e2e) = 覆盖份额 × (1 − 1 / ratio)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts.evaluator import Baseline
from ..contracts.task import Task, load_module, load_task, resolve_callable
from ..contracts.timing import time_ms
from ..generators.fixture import FixtureGenerator
from ..generators.kernelgen_local import KernelgenLocal
from .simple import TierOutcome, run_simple_tier


@dataclass
class FusionOutcome:
    outcome: TierOutcome
    region: str
    ops: List[str]
    coverage_pct: float
    contribution_pct: float

    def to_dict(self) -> Dict[str, Any]:
        d = self.outcome.to_dict()
        d.update({
            "kind": "fused",
            "region": self.region,
            "ops": self.ops,
            "coverage_pct": round(self.coverage_pct, 3),
            "contribution_pct": round(self.contribution_pct, 4),
        })
        return d


def _as_task(task_or_dir: Task | str | Path) -> Task:
    return load_task(task_or_dir) if isinstance(task_or_dir, (str, Path)) else task_or_dir


def run_fusion_tier(
    task_or_dir: Task | str | Path,
    generator,
    **kwargs,
) -> FusionOutcome:
    """跑一个融合任务。参数与 `run_simple_tier` 相同（n/rounds/workdir/evaluate_fn/...）。"""
    task = _as_task(task_or_dir)
    if not task.is_fused:
        raise ValueError(
            f"task kind={task.kind!r}：融合档只接受 kind='fused' 的任务，"
            "单算子请用 tiers/simple.py"
        )
    if task.allow_complex:
        raise ValueError("融合档目前只支持简单档；请把 escalation.allow_complex 设为 false")

    outcome = run_simple_tier(task, generator, **kwargs)

    coverage = task.coverage_pct
    ratio = outcome.best_ratio
    contribution = 0.0
    if outcome.verdict == "KEEP" and ratio and ratio > 0:
        contribution = coverage * (1.0 - 1.0 / ratio)

    return FusionOutcome(
        outcome=outcome,
        region=task.region or "",
        ops=list(task.ops),
        coverage_pct=coverage,
        contribution_pct=contribution,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="optframe.tiers.fusion")
    ap.add_argument("--task", required=True, help="融合任务目录（task.toml 里 kind='fused'）")
    ap.add_argument("--generator", choices=["fixture", "llm"], default="fixture")
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--base-url", default="http://10.51.198.60:10152/v1-openai")
    ap.add_argument("--model", default="ds-v4")
    ap.add_argument("--prompt-mode", choices=["default", "strict"], default="default")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--api-key-env", default="GPUSTACK_API_KEY")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--noise-floor", type=float, default=1.02)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--baseline-module", default=None,
        help="性能基线实现（file.py:run）。融合任务的判定必须对标'服务同款'实现"
             "（如 mm + 单个 act_and_mul kernel），否则会拿 torch 参考实现当基线，结论失真。")
    args = ap.parse_args(argv)

    task_dir = Path(args.task)
    if not task_dir.exists():
        print(f"[FAIL] task dir not found: {task_dir}")
        return 2

    if args.generator == "fixture":
        if not args.fixture_dir:
            print("[FAIL] fixture 模式需要 --fixture-dir（融合任务暂无内置 fixture）")
            return 2
        generator = FixtureGenerator(args.fixture_dir)
    else:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            print(f"[FAIL] env {args.api_key_env} is empty")
            return 2
        generator = KernelgenLocal(
            base_url=args.base_url, model=args.model, api_key=api_key,
            prompt_mode=args.prompt_mode, max_tokens=args.max_tokens,
        )

    baseline_fn = None
    if args.baseline_module:
        mod_path, _, _attr = args.baseline_module.partition(":")
        module = load_module(Path(mod_path), alias="fusion_baseline")
        fn = resolve_callable(module, args.baseline_module)

        def baseline_fn(task, inputs, warmup=10, repeat=50, flush_cache=True):  # noqa: F811
            outputs = fn(*inputs)
            ms = time_ms(lambda: fn(*inputs), warmup=warmup, repeat=repeat,
                         flush_cache=flush_cache)
            return Baseline(outputs=outputs, ms=ms, fn=lambda: fn(*inputs))

    try:
        result = run_fusion_tier(
            task_dir, generator,
            n=args.n, rounds=args.rounds, noise_floor=args.noise_floor,
            workdir=args.workdir, warmup=args.warmup, repeat=args.repeat,
            baseline_fn=baseline_fn,
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    o = result.outcome
    print(f"task={o.task} kind=fused region={result.region} coverage={result.coverage_pct}%")
    print(f"baseline={o.baseline_ms*1000:.1f}us verdict={o.verdict} "
          f"ratio={o.best_ratio} ci_low={o.best_ratio_low}")
    print(f"contribution={result.contribution_pct:.4f}% e2e")
    for rec in o.rounds:
        print(f"  round {rec.round}: cand={rec.candidates} correct={rec.correct} best={rec.best_ratio}")
    print(f"ledger: {o.ledger_path}")
    print("JSON " + json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if o.verdict == "KEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
