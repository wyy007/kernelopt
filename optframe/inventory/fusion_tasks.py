"""阶段3（融合模式）：把 trace 里"相邻且可融"的算子聚成 region，生成任务包。

当前只实现一个模式：**GEMM + gated-SiLU**（SwiGLU），也就是 8B 上占 34% GPU 时间的那个形状。
其余模式（GEMM+cast、elementwise 链）留扩展位，不在本轮范围。

输入：阶段2 的 `bottlenecks.json`（有 op / shape / pct_e2e）
输出：任务目录，`task.toml` 里 `kind="fused"`，含组合 reference 与 judge

设计原则（见 docs/modes.md）：
- 不碰 `tiers/simple.py`，产物由 `contracts/task.py` 统一加载；
- region 独占：同一 region 内的算子不应再进单算子队列（由调用方按 region 过滤）；
- `pct_e2e` = region 内各算子之和，用于统一收益口径。
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .codegen import judge_source
from .taskspec import TaskSpec

#: 识别 act_and_mul 类算子的名字片段
_ACT_HINTS = ("_and_mul", "_run_activation", "silu_and_mul", "gelu_and_mul")
#: 识别 GEMM 的名字片段
_MM_HINTS = ("aten::mm", "aten::matmul", "aten::addmm")


@dataclass
class Region:
    """一个融合候选：若干算子 + 组合后的形状与份额。"""

    pattern: str
    ops: List[str]
    shapes: List[List[int]]
    pct_e2e: float
    gpu_time_ms: float
    region: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.shapes[0][0])

    @property
    def k(self) -> int:
        return int(self.shapes[0][1])

    @property
    def n(self) -> int:
        return int(self.shapes[1][1])


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

def _parse_shape(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def _is_mm(op: str) -> bool:
    return any(h in op for h in _MM_HINTS)


def _is_act(op: str) -> bool:
    return any(h in op for h in _ACT_HINTS)


def _mm_dims(shape: Any) -> Optional[tuple]:
    """((M, K), (K, N)) -> (M, K, N)"""
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return None
    a, b = shape
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return None
    if len(a) != 2 or len(b) != 2:
        return None
    m, k1 = int(a[0]), int(a[1])
    k2, n = int(b[0]), int(b[1])
    if k1 != k2:
        return None
    return m, k1, n


def _act_dims(shape: Any) -> Optional[tuple]:
    """((), (M, 2H), (M, H)) -> (M, 2H, H)"""
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return None
    _, inp, out = shape
    if not isinstance(inp, (list, tuple)) or not isinstance(out, (list, tuple)):
        return None
    if len(inp) != 2 or len(out) != 2:
        return None
    return int(inp[0]), int(inp[1]), int(out[1])


# --------------------------------------------------------------------------- #
# 检测
# --------------------------------------------------------------------------- #

def detect_swiglu(findings: Sequence[Dict[str, Any]], *, min_share: float = 0.0) -> List[Region]:
    """GEMM (M,K)x(K,2H) + act_and_mul -> SwiGLU region。"""
    mms: List[tuple] = []
    acts: List[tuple] = []
    for f in findings:
        op = str(f.get("op") or "")
        dims = _parse_shape(f.get("shape"))
        if _is_mm(op):
            d = _mm_dims(dims)
            if d:
                mms.append((f, d))
        elif _is_act(op):
            d = _act_dims(dims)
            if d:
                acts.append((f, d))

    # act 索引：(M, 2H) -> finding
    act_index: Dict[tuple, Dict[str, Any]] = {}
    for f, (m, two_h, _h) in acts:
        act_index[(m, two_h)] = f

    regions: List[Region] = []
    for f, (m, k, n) in mms:
        act = act_index.get((m, n))
        if act is None:
            continue
        share = float(f.get("pct_e2e") or 0.0) + float(act.get("pct_e2e") or 0.0)
        if share < min_share:
            continue
        regions.append(Region(
            pattern="swiglu",
            ops=[str(f.get("op")), str(act.get("op"))],
            shapes=[[m, k], [k, n]],
            pct_e2e=round(share, 3),
            gpu_time_ms=round(float(f.get("gpu_time_ms") or 0) + float(act.get("gpu_time_ms") or 0), 3),
            region=f"swiglu_m{m}_k{k}_n{n}",
            detail={"mm_share": f.get("pct_e2e"), "act_share": act.get("pct_e2e")},
        ))

    regions.sort(key=lambda r: -r.pct_e2e)
    return regions


PATTERNS = {"swiglu": detect_swiglu}


# --------------------------------------------------------------------------- #
# 代码生成
# --------------------------------------------------------------------------- #

_SWIGLU_REF = '''"""融合 region 的参考实现（自动生成，请勿手改）。

region: {region}
算子序列: {ops}
形状: {shapes}
说明: 参考实现可以用任意 torch 算子；候选实现必须用一次 kernel 完成等价语义。
"""

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
SHAPES = {shapes}


def make_inputs(device="cuda", dtype=DTYPE, seed=0):
    torch.manual_seed(seed)
    return [
        torch.randn(*SHAPES[0], device=device, dtype=dtype),
        torch.randn(*SHAPES[1], device=device, dtype=dtype),
    ]


def run(x, w):
    """SwiGLU: silu(gate) * up，其中 [gate | up] = x @ w。"""
    full = torch.mm(x, w)
    half = full.shape[1] // 2
    gate, up = full[:, :half], full[:, half:]
    return F.silu(gate) * up
'''


def _write_task(region: Region, outdir: Path, *, arch: str, dtype: str,
                target_ratio: float, language: str) -> Path:
    spec = TaskSpec(
        name=f"fused__{region.pattern}__{region.m}x{region.k}x{region.n}",
        op=f"fused::{region.pattern}",
        arch=arch,
        dtype=dtype,
        shapes=region.shapes,
        language=language,
        target_ratio=target_ratio,
        category="FUSED",
        action="tune",
        pct_e2e=region.pct_e2e,
        kind="fused",
        ops=list(region.ops),
        region=region.region,
        meta={"pattern": region.pattern, "gpu_time_ms": region.gpu_time_ms},
    )
    task_dir = outdir / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "task.toml").write_text(spec.to_toml(), encoding="utf-8")
    (task_dir / "reference.py").write_text(
        _SWIGLU_REF.format(region=region.region, ops=region.ops, shapes=region.shapes),
        encoding="utf-8",
    )
    (task_dir / "judge.py").write_text(judge_source(dtype), encoding="utf-8")
    (task_dir / "notes.md").write_text(
        "\n".join([
            f"# 融合任务 {spec.name}",
            "",
            f"- region: `{region.region}`",
            f"- 算子序列: {' -> '.join(region.ops)}",
            f"- 形状: {region.shapes}（输出 {region.m}x{region.n // 2}）",
            f"- 占 e2e: {region.pct_e2e}%（mm {region.detail.get('mm_share')}% + act {region.detail.get('act_share')}%）",
            f"- GPU 时间: {region.gpu_time_ms} ms",
            "",
            "候选只需实现 `run(x, w) -> Tensor(M, H)`，由 evaluator 统一复测。",
        ]) + "\n",
        encoding="utf-8",
    )
    return task_dir


def build(report_path: str | Path, outdir: str | Path, *, arch: str, dtype: str,
          pattern: str = "swiglu", min_share: float = 0.0, target_ratio: float = 1.05,
          language: str = "triton") -> Dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    findings = report.get("findings") or report.get("bottlenecks") or []
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern!r}")
    regions = PATTERNS[pattern](findings, min_share=min_share)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for r in regions:
        d = _write_task(r, outdir, arch=arch, dtype=dtype,
                        target_ratio=target_ratio, language=language)
        written.append({"name": d.name, "dir": str(d), "pct_e2e": r.pct_e2e,
                        "ops": r.ops, "region": r.region})

    index = {
        "schema_version": "1.0",
        "kind": "fused",
        "pattern": pattern,
        "source_report": str(report_path),
        "source_run_id": report.get("run_id"),
        "target": report.get("target"),
        "tasks": written,
        "total_pct_e2e": round(sum(w["pct_e2e"] for w in written), 3),
    }
    (outdir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="bottlenecks.json -> 融合任务包")
    ap.add_argument("--report", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arch", default="sm_120")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--pattern", default="swiglu", choices=sorted(PATTERNS))
    ap.add_argument("--min-share", type=float, default=0.0, help="region 最小 e2e 份额（%%）")
    ap.add_argument("--target-ratio", type=float, default=1.05)
    args = ap.parse_args(argv)

    index = build(args.report, args.outdir, arch=args.arch, dtype=args.dtype,
                  pattern=args.pattern, min_share=args.min_share,
                  target_ratio=args.target_ratio)
    print(f"[1/2] 生成 {len(index['tasks'])} 个融合任务（pattern={args.pattern}）")
    for t in index["tasks"]:
        print(f"      ✅ {t['name']}  region={t['region']}  占 e2e {t['pct_e2e']}%")
    print(f"[2/2] 覆盖合计 {index['total_pct_e2e']}% e2e -> {Path(args.outdir) / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
