"""阶段2：TraceLens 分析层。

职责边界（总体方案 §九 关键约束1）：
    只做诊断，不修改任何算子代码；输出 bottlenecks.json 交给阶段3。
"""

from .schema import Finding, Report, validate

__all__ = ["Finding", "Report", "validate"]
