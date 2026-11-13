"""optframe — 全链路推理算子自动化优化框架。

阶段划分（见 doc/全链路推理算子自动化优化完整方案.md）：
    阶段2  analysis    trace -> bottlenecks.json（本包已实现）
    阶段3  inventory   bottlenecks.json -> task.toml
    阶段4  tiers       简单档 Triton / 复杂档 ForgeKernel
    阶段5  deliver     回灌 + 闭环回归
"""

__version__ = "0.1.0"
