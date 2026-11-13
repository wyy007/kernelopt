"""确定性生成器：从目录读取预置候选。

用途：
1. 在没有可用 LLM（队列被占、离线、配额）时，仍能验证 evaluator/ledger/tier 全链路；
2. 回归测试：固定候选 => 固定判定，可当作 CI 用例。

它不能替代真实生成器 —— 它不产出新代码，只回放。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..contracts.schema import Candidate
from ..contracts.task import Task
from .base import GenContext


class FixtureGenerator:
    name = "fixture"

    def __init__(self, fixture_dir: str | Path):
        self.dir = Path(fixture_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"fixture dir not found: {self.dir}")
        self._files = sorted(self.dir.glob("*.py"))
        if not self._files:
            raise FileNotFoundError(f"no .py candidates in {self.dir}")
        self._cursor = 0

    def propose(self, task: Task, context: GenContext, n: int = 1) -> List[Candidate]:
        out: List[Candidate] = []
        for _ in range(n):
            if self._cursor >= len(self._files):
                break
            path = self._files[self._cursor]
            self._cursor += 1
            out.append(
                Candidate(
                    source=path.read_text(encoding="utf-8"),
                    language="triton",
                    generator=f"fixture:{path.name}",
                    meta={"path": str(path)},
                )
            )
        return out
