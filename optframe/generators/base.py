"""生成器协议：两档都只实现 propose()，不参与评测。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol

from ..contracts.schema import Candidate
from ..contracts.task import Task


@dataclass
class GenContext:
    """跨轮次上下文：错误反馈是"重试"能否有效的前提。"""

    round: int = 0
    errors: List[str] = field(default_factory=list)
    previous: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class Generator(Protocol):
    name: str

    def propose(self, task: Task, context: GenContext, n: int = 1) -> List[Candidate]:
        ...
