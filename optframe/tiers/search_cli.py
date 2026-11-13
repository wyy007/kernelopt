"""配置搜索 CLI：python -m optframe.tiers.search_cli。

示例：
  python -m optframe.tiers.search_cli \
    --task runs/stage3_tasks/silu__2048x2048 \
    --source runs/stage4_dsv4_silu2/cand_xxx.py \
    --workdir runs/search_silu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ..contracts.evaluator import measure_baseline
from ..contracts.task import load_task
from .search import DEFAULT_BLOCKS, DEFAULT_WARPS, run_search


def _ints(text: str) -> List[int]:
    return [int(x) for x in text.replace(",", " ").split()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="optframe.tiers.search_cli")
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", required=True, help="已通过正确性的候选源码文件（.py）")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--blocks", default=" ".join(str(b) for b in DEFAULT_BLOCKS))
    parser.add_argument("--warps", default=" ".join(str(w) for w in DEFAULT_WARPS))
    parser.add_argument("--matmul", action="store_true",
                        help="按 BLOCK_M/N/K × num_warps 扫描（matmul 任务）")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args(argv)

    task = load_task(Path(args.task))
    source = Path(args.source).read_text(encoding="utf-8")
    workdir = Path(args.workdir) if args.workdir else Path("runs") / f"search_{task.name}"

    inputs = task.make_inputs(device="cuda")
    baseline = measure_baseline(task, inputs, warmup=args.warmup, repeat=args.repeat)

    outcome = run_search(
        task, source,
        baseline=baseline,
        inputs=inputs,
        workdir=workdir,
        matmul=args.matmul,
        blocks=_ints(args.blocks),
        warps=_ints(args.warps),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    print(f"task={outcome.task} variants={len(outcome.variants)} kept={outcome.kept}")
    if outcome.best and outcome.best.result:
        r = outcome.best.result
        print(f"best: block={outcome.best.block} warps={outcome.best.warps} "
              f"ratio={r.ratio} ci_low={r.ratio_low}")
    else:
        print("best: 无有效候选")
    print(f"ledger: {outcome.ledger_path}")
    return 0 if outcome.kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
