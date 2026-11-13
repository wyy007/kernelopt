"""阶段3：瓶颈 -> 标准 Task。

已实现：Amdahl 过滤（amdahl.py）。
后续：trace_to_tasks.py（reference.py / judge.py / task.toml 生成）、档位升级判定。
"""

from .amdahl import AmdahlParams, AmdahlResult, select

__all__ = ["AmdahlParams", "AmdahlResult", "select"]
