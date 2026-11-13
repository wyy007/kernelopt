"""简单档 CLI。

用法：
  python -m optframe.tiers.simple_cli --task runs/stage3_tasks/silu__2048x2048 \
      --generator fixture --fixture-dir fixtures/silu

  python -m optframe.tiers.simple_cli --task ... --generator llm \
      --base-url http://10.51.198.60:10152/v1-openai --model qwen3.5-9b

  # 用真实生产实现做性能基线（reference 仍用于正确性）
  python -m optframe.tiers.simple_cli --task ... \
      --baseline-module /model/wyy/flagos-opt/prod_silu.py:run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from ..contracts.ledger import Ledger
from ..contracts.evaluator import Baseline
from ..contracts.task import load_module, resolve_callable
from ..contracts.timing import time_ms
from ..generators.fixture import FixtureGenerator
from ..generators.kernelgen_local import KernelgenLocal
from .simple import run_simple_tier


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="optframe.tiers.simple_cli")
    parser.add_argument("--task", required=True, help="任务目录（含 task.toml）")
    parser.add_argument("--generator", choices=["fixture", "llm"], default="fixture")
    parser.add_argument("--fixture-dir", default=None)
    parser.add_argument("--base-url", default="http://10.51.198.60:10152/v1-openai")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--prompt-mode", choices=["default", "strict"], default="default",
                        help="strict: 给弱模型一个可填空骨架")
    parser.add_argument("--max-tokens", type=int, default=3000,
                        help="生成上限；matmul 等长代码需要更大值")
    parser.add_argument("--api-key-env", default="GPUSTACK_API_KEY")
    parser.add_argument("--n", type=int, default=3, help="每轮候选数（best-of-N）")
    parser.add_argument("--rounds", type=int, default=2, help="最多轮次（含错误重试）")
    parser.add_argument("--noise-floor", type=float, default=1.02)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--baseline-module", default=None,
        help="性能基线实现（file.py:run）。省略则用 reference（torch）做基线；"
             "正确性永远以 reference 为准。",
    )
    parser.add_argument("--allow-noisy-keep", action="store_true",
        help="关闭配对 bootstrap 置信区间要求（仅快速调试用，验收不要开）")
    args = parser.parse_args(argv)

    task_dir = Path(args.task)
    if not task_dir.exists():
        print(f"[FAIL] task dir not found: {task_dir}")
        return 2

    if args.generator == "fixture":
        if not args.fixture_dir:
            fixture_dir = task_dir.parent.parent / "fixtures" / task_dir.name.split("__")[0]
            args.fixture_dir = str(fixture_dir)
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

    workdir = Path(args.workdir) if args.workdir else None

    baseline_fn = None
    if args.baseline_module:
        mod_path, _, _attr = args.baseline_module.partition(":")
        module = load_module(Path(mod_path), alias="prod_baseline")
        fn = resolve_callable(module, args.baseline_module)

        def baseline_fn(task, inputs, warmup=10, repeat=50):  # noqa: F811
            outputs = fn(*inputs)
            ms = time_ms(lambda: fn(*inputs), warmup=warmup, repeat=repeat)
            # 带上可调用对象，让 evaluator 能做交错 A/B 与置信区间
            return Baseline(outputs=outputs, ms=ms, fn=lambda: fn(*inputs))

    outcome = run_simple_tier(
        task_dir,
        generator,
        n=args.n,
        rounds=args.rounds,
        noise_floor=args.noise_floor,
        workdir=workdir,
        baseline_fn=baseline_fn,
        warmup=args.warmup,
        repeat=args.repeat,
        require_ci=not args.allow_noisy_keep,
    )

    print(f"task={outcome.task} baseline={outcome.baseline_ms*1000:.1f}us "
          f"verdict={outcome.verdict} best_ratio={outcome.best_ratio} "
          f"ci_low={outcome.best_ratio_low} escalate={outcome.escalate}")
    for rec in outcome.rounds:
        print(f"  round {rec.round}: cand={rec.candidates} correct={rec.correct} "
              f"best={rec.best_ratio}")
    print(f"ledger: {outcome.ledger_path}")
    return 0 if outcome.verdict == "KEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
