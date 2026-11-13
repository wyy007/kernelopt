"""简单档编排：best-of-N + 错误反馈重试 + 噪声门限 + 升级判定。

流程（双档方案 §5.3）：
    task -> propose(n) -> evaluate（正确性 -> 计时）
         -> 失败/不达标：带错误重试（最多 rounds-1 次）
         -> 达标：KEEP，停止
         -> 全轮不达标：按运行期规则判定是否升级复杂档
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..contracts.evaluator import Baseline, evaluate_candidate, measure_baseline
from ..contracts.ledger import Ledger
from ..contracts.schema import Candidate, Result, Verdict
from ..contracts.task import Task, load_task
from ..generators.base import GenContext, Generator
from ..inventory.escalation import should_escalate_after_best_of_three

#: 小于该倍数视为噪声，不允许 KEEP（方案 §4.3）
DEFAULT_NOISE_FLOOR = 1.02


@dataclass
class RoundRecord:
    round: int
    candidates: int = 0
    best_ratio: Optional[float] = None
    correct: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "candidates": self.candidates,
            "correct": self.correct,
            "best_ratio": None if self.best_ratio is None else round(self.best_ratio, 4),
            "errors": self.errors,
        }


@dataclass
class TierOutcome:
    task: str
    baseline_ms: float
    verdict: str
    best_ratio: Optional[float]
    best_source: Optional[str]
    rounds: List[RoundRecord]
    escalate: bool
    ledger_path: Optional[str]
    best_ratio_low: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "baseline_ms": round(self.baseline_ms, 6),
            "verdict": self.verdict,
            "best_ratio": None if self.best_ratio is None else round(self.best_ratio, 4),
            "best_ratio_low": (
                None if self.best_ratio_low is None else round(self.best_ratio_low, 4)
            ),
            "escalate": self.escalate,
            "rounds": [r.to_dict() for r in self.rounds],
            "ledger": self.ledger_path,
            "best_source": self.best_source,
        }


def run_simple_tier(
    task_or_dir: Task | str | Path,
    generator: Generator,
    *,
    n: int = 3,
    rounds: int = 2,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    workdir: Optional[str | Path] = None,
    ledger: Optional[Ledger] = None,
    evaluate_fn: Optional[Callable[..., Result]] = None,
    baseline_fn: Optional[Callable[..., Baseline]] = None,
    inputs: Optional[List[Any]] = None,
    warmup: int = 10,
    repeat: int = 50,
    device: str = "cuda",
    require_ci: bool = True,
) -> TierOutcome:
    task = load_task(task_or_dir) if isinstance(task_or_dir, (str, Path)) else task_or_dir
    workdir = Path(workdir) if workdir else (task.dir / "runs" / "simple")
    workdir.mkdir(parents=True, exist_ok=True)

    if inputs is None:
        inputs = task.make_inputs(device=device)
    baseline_fn = baseline_fn or measure_baseline
    baseline = baseline_fn(task, inputs, warmup=warmup, repeat=repeat)

    ledger = ledger or Ledger(workdir / "ledger.jsonl", run_id=task.name)
    ledger.append(
        stage="baseline",
        verdict=Verdict.BASELINE.value,
        task=task.name,
        generator="torch-reference",
        payload={"ref_ms": round(baseline.ms, 6)},
    )

    evaluate_fn = evaluate_fn or evaluate_candidate
    records: List[RoundRecord] = []
    errors: List[str] = []
    best: Optional[Result] = None
    best_source: Optional[str] = None

    def _passes(result: Result) -> bool:
        """双重门槛：中位比值超噪声下限，且配对 bootstrap 下界不低于 1.0。

        只看点估计会在共享 GPU 上反复翻盘（实测同一候选 1.036/1.000/1.014）。
        """
        if not result.correct or result.ratio is None or result.ratio < noise_floor:
            return False
        if not require_ci:
            return True
        if result.ratio_low is None:
            return False
        # 严格大于 1.0：置信区间下界等于 1.0 说明"分不出胜负"，不能算赢
        return result.ratio_low > 1.0

    for round_idx in range(max(rounds, 1)):
        context = GenContext(round=round_idx, errors=list(errors))
        candidates: List[Candidate] = list(generator.propose(task, context, n=n))
        record = RoundRecord(round=round_idx, candidates=len(candidates))

        for cand in candidates:
            if not cand.source.strip():
                err = cand.meta.get("error", "empty candidate")
                record.errors.append(err[:400])
                errors.append(err[:400])
                ledger.append(
                    stage="candidate",
                    verdict=Verdict.INVALID.value,
                    task=task.name,
                    generator=cand.generator,
                    candidate_sha=cand.sha,
                    payload={"error": err[:400]},
                )
                continue

            result = evaluate_fn(
                task, cand, baseline, inputs, workdir,
                warmup=warmup, repeat=repeat,
            )
            passed = _passes(result)
            ledger.append(
                stage="candidate",
                verdict=Verdict.KEEP.value if passed else Verdict.REVERT.value,
                task=task.name,
                generator=cand.generator,
                candidate_sha=cand.sha,
                payload=result.to_dict(),
            )

            if result.correct:
                record.correct += 1
            else:
                errors.append(f"{cand.generator}: {result.status} - {result.error}"[:600])
                record.errors.append(f"{result.status}: {result.error}"[:400])

            if result.correct and result.ratio is not None:
                if record.best_ratio is None or result.ratio > record.best_ratio:
                    record.best_ratio = result.ratio
                if best is None or (result.ratio or 0) > (best.ratio or 0):
                    best = result
                    best_source = cand.source

        records.append(record)

        if best is not None and _passes(best):
            break

    best_ratio = max((r.best_ratio for r in records if r.best_ratio is not None), default=None)
    kept = best is not None and _passes(best)
    escalate = bool(
        not kept
        and task.allow_complex
        and should_escalate_after_best_of_three(best_ratio)
    )

    outcome = TierOutcome(
        task=task.name,
        baseline_ms=baseline.ms,
        verdict=Verdict.KEEP.value if kept else Verdict.REVERT.value,
        best_ratio=best_ratio,
        best_source=best_source if kept else None,
        rounds=records,
        escalate=escalate,
        ledger_path=str(ledger.path),
        best_ratio_low=best.ratio_low if best is not None else None,
    )
    (workdir / "simple_result.json").write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outcome
