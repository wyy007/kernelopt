"""配置搜索：固定一份已通过正确性的 kernel，扫 BLOCK × num_warps。

为什么需要它（2026-11-10 的实测）：
    ds-v4 生成的 silu 6/6 正确，但 ratio 全是 1.000 —— 模型会写 kernel，不会调参。
    best-of-N 只是换温度重采样，不是搜索；这里补上真正的搜索。

前提：候选源码里 BLOCK / num_warps 是可识别的字面量（多数 Triton 代码满足）。
做法：正则改写这两个字面量生成变体，逐个走共享 evaluator（正确性 + 交错 A/B），
      按配对比值选优；每个变体都记入账本。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..contracts.evaluator import Baseline, evaluate_candidate
from ..contracts.ledger import Ledger
from ..contracts.schema import Candidate, Result, Verdict

_BLOCK_ASSIGN = re.compile(r"(?P<lhs>\bBLOCK(?:_SIZE)?\s*[:=]\s*)(?P<val>\d+)")
_WARPS_ASSIGN = re.compile(r"(?P<lhs>\bnum_warps\s*=\s*)(?P<val>\d+)")
#: matmul 的 tile 参数（BLOCK_M / BLOCK_N / BLOCK_K）
_TILE_ASSIGN = re.compile(r"(?P<lhs>\bBLOCK_[MNK]\s*=\s*)(?P<val>\d+)")
#: Triton kernel 启动调用：`kernel[grid](args...)`
_LAUNCH_CALL = re.compile(r"(\]\s*\()([^()]*)(\))")

DEFAULT_BLOCKS = (256, 512, 1024, 2048, 4096)
DEFAULT_WARPS = (1, 2, 4, 8)


@dataclass
class Variant:
    block: Optional[int]
    warps: Optional[int]
    source: str
    result: Optional[Result] = None
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block": self.block,
            "label": self.label,
            "warps": self.warps,
            "ratio": None if not self.result else self.result.ratio,
            "ratio_low": None if not self.result else self.result.ratio_low,
            "correct": None if not self.result else self.result.correct,
            "status": None if not self.result else self.result.status,
            "error": "" if not self.result else self.result.error[:200],
        }


@dataclass
class SearchOutcome:
    task: str
    base_ratio: Optional[float]
    best: Optional[Variant]
    variants: List[Variant] = field(default_factory=list)
    ledger_path: Optional[str] = None

    @property
    def kept(self) -> bool:
        if self.best is None or self.best.result is None:
            return False
        r = self.best.result
        return bool(r.correct and r.ratio and r.ratio >= 1.02
                    and (r.ratio_low is None or r.ratio_low > 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "base_ratio": self.base_ratio,
            "kept": self.kept,
            "best": None if self.best is None else self.best.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "ledger": self.ledger_path,
        }


def rewrite_config(source: str, block: Optional[int], warps: Optional[int]) -> str:
    """把源码里的 BLOCK / num_warps 改成指定值；没有 num_warps 时注入到启动调用。"""
    out = source
    if block is not None:
        out = _BLOCK_ASSIGN.sub(lambda m: f"{m.group('lhs')}{block}", out)
    if warps is not None:
        if _WARPS_ASSIGN.search(out):
            out = _WARPS_ASSIGN.sub(lambda m: f"{m.group('lhs')}{warps}", out)
        else:
            # 候选没写 num_warps（走默认值）；注入到第一个 kernel[grid](...) 调用
            def _inject(match: re.Match) -> str:
                args = match.group(2).strip()
                if "num_warps" in args:
                    return match.group(0)
                new_args = f"{args}, num_warps={warps}" if args else f"num_warps={warps}"
                return f"{match.group(1)}{new_args}{match.group(3)}"

            out = _LAUNCH_CALL.sub(_inject, out, count=1)
    return out


def has_config_knobs(source: str) -> Tuple[bool, bool]:
    """返回 (可扫 BLOCK, 可扫 num_warps)。

    num_warps 没有写也能扫：注入到第一个 `kernel[grid](...)` 调用即可。
    """
    return (
        bool(_BLOCK_ASSIGN.search(source)),
        bool(_WARPS_ASSIGN.search(source)) or bool(_LAUNCH_CALL.search(source)),
    )


def build_variants(
    source: str,
    blocks: Sequence[int] = DEFAULT_BLOCKS,
    warps: Sequence[int] = DEFAULT_WARPS,
) -> List[Variant]:
    """按网格生成变体；缺省维度用 None 表示"不扫这一维"。"""
    has_block, has_warps = has_config_knobs(source)
    block_values: Iterable[Optional[int]] = blocks if has_block else (None,)
    warp_values: Iterable[Optional[int]] = warps if has_warps else (None,)

    out: List[Variant] = []
    seen_sources = set()
    for block, warp in product(block_values, warp_values):
        src = rewrite_config(source, block, warp)
        if src in seen_sources:
            continue
        seen_sources.add(src)
        out.append(Variant(block=block, warps=warp, source=src))
    return out


def run_search(
    task,
    source: str,
    *,
    baseline: Baseline,
    inputs: List[Any],
    workdir: str | Path,
    blocks: Sequence[int] = DEFAULT_BLOCKS,
    warps: Sequence[int] = DEFAULT_WARPS,
    matmul: bool = False,
    evaluate_fn=evaluate_candidate,
    warmup: int = 10,
    repeat: int = 50,
) -> SearchOutcome:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(workdir / "ledger.jsonl", run_id=f"{task.name}_search")
    variants = (
        build_matmul_variants(source, warps=warps) if matmul
        else build_variants(source, blocks, warps)
    )
    best: Optional[Variant] = None

    for variant in variants:
        cand = Candidate(
            source=variant.source,
            language="triton",
            generator=f"search:{variant.label or f'block{variant.block}_warps{variant.warps}'}",
        )
        result = evaluate_fn(task, cand, baseline, inputs, workdir,
                             warmup=warmup, repeat=repeat)
        variant.result = result

        passed = bool(result.correct and result.ratio and result.ratio >= 1.02
                      and (result.ratio_low is None or result.ratio_low > 1.0))
        ledger.append(
            stage="candidate",
            verdict=Verdict.KEEP.value if passed else Verdict.REVERT.value,
            task=task.name,
            generator=cand.generator,
            candidate_sha=cand.sha,
            payload={**result.to_dict(), "block": variant.block, "warps": variant.warps},
        )

        if result.correct and result.ratio is not None:
            if best is None or (result.ratio or 0) > (best.result.ratio or 0):
                best = variant

    outcome = SearchOutcome(
        task=task.name,
        base_ratio=None,
        best=best,
        variants=variants,
        ledger_path=str(ledger.path),
    )
    (workdir / "search_result.json").write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outcome


def build_matmul_variants(
    source: str,
    blocks_m: Sequence[int] = (64, 128),
    blocks_n: Sequence[int] = (64, 128),
    blocks_k: Sequence[int] = (32, 64),
    warps: Sequence[int] = (4, 8),
) -> List[Variant]:
    """针对 matmul 的 BLOCK_M/N/K × num_warps 网格。"""
    if not _TILE_ASSIGN.search(source):
        return build_variants(source)

    out: List[Variant] = []
    seen = set()
    for bm, bn, bk, wp in product(blocks_m, blocks_n, blocks_k, warps):
        src = source
        src = re.sub(r"(\bBLOCK_M\s*=\s*)\d+", rf"\g<1>{bm}", src)
        src = re.sub(r"(\bBLOCK_N\s*=\s*)\d+", rf"\g<1>{bn}", src)
        src = re.sub(r"(\bBLOCK_K\s*=\s*)\d+", rf"\g<1>{bk}", src)
        src = rewrite_config(src, None, wp)
        if src in seen:
            continue
        seen.add(src)
        out.append(Variant(block=None, warps=wp, source=src,
                           label=f"M{bm}N{bn}K{bk}w{wp}"))
    return out
