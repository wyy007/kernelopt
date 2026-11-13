# kernelopt — 全链路推理算子自动化优化框架

把「真实推理 trace → 瓶颈归因 → 算子优化任务 → 候选生成/评测 → 回灌闭环」串成一条可执行、可复现的流水线。
目标不是让模型"写出能跑的 kernel"，而是**在不确定的 LLM 生成上给出可信的加速结论**：正确性优先、冷缓存交错 A/B、bootstrap CI、账本留痕。

```
trace ─(阶段2)─> bottlenecks.json ─(阶段3)─> amdahl.json ─> task.toml
                                                              │
                                          (阶段4) 简单档 Triton / 复杂档 ForgeKernel
                                                              │
                                                    ledger + 复测 → (阶段5) 回灌服务
```

## 五个阶段

| 阶段 | 职责 | 入口 |
| --- | --- | --- |
| 1 采集 | 起服务、压测、profiler 采 trace、切稳态窗口、质量校验 | `tools/smoke_trace.py`（合成）+ 外部 profiler |
| 2 分析 | TraceLens 报表 → roofline/SOL 判定 → `bottlenecks.json` | `python -m optframe.analysis.cli` |
| 3 转换 | Amdahl 两层过滤 → 任务包（`task.toml` + `reference.py` + `judge.py`） | `python -m optframe.inventory.cli` / `trace_to_tasks` |
| 4 优化 | 共享合同层 + 简单档（Triton，best-of-N/重试/双门槛）+ 复杂档（ForgeKernel 适配） | `python -m optframe.tiers.simple_cli` / `complex_cli` |
| 5 回灌 | 进程内注入 KEEP kernel，按 `(K,N)` 注册、`min_m` 批次门槛，再测端到端 | `delivery/sitecustomize.py` |

## 目录

```
optframe/
  analysis/     TraceLens 报表解析、roofline/SOL、过滤规则、可选 ncu 合并
  inventory/    Amdahl 过滤、任务生成、静态/运行期升级条件
  contracts/    任务加载、计时、judge、账本、CUDA 加载、源码守卫（反作弊）
  generators/   Triton 候选生成（LLM 驱动 / fixture 对照）
  tiers/        简单档编排与配置搜索、复杂档 ForgeKernel 四段适配
  preflight.py  换卡/换环境自检（device / toolkit / triton / arch / tasks / hw_kb）
tests/          98 个用例（多数不需要 GPU）
tools/          辅助脚本（合成 trace、arch 合并、ncu 选取等）
arch/           目标 GPU 的 roofline 基线（L20 / RTX PRO 5000 等）
fixtures/       手写对照 kernel（good / slow / wrong）
delivery/       阶段5 回灌注入（sitecustomize + 按形状注册）
hw_kb_sm120_index.md   sm_120 架构知识库索引（复杂档用）
```

## 快速开始

```bash
# 0) 环境自检：换卡后先跑这个，避免内核静默不执行、输出全 0
python -m optframe.preflight

# 1) 分析 trace，产出瓶颈清单
python -m optframe.analysis.cli \
  --trace runs/smoke/trace.json --arch arch/RTX_PRO_5000.json \
  --workdir runs/smoke_stage2 --framework sglang --model qwen3.5-9b

# 2) Amdahl 过滤 + 生成任务包
python -m optframe.inventory.cli --report runs/smoke_stage2/bottlenecks.json --workdir runs/stage3
python -m optframe.inventory.trace_to_tasks --amdahl runs/stage3/amdahl.json --outdir runs/tasks

# 3) 简单档：生成候选并评测（best-of-N + CI 双门槛）
python -m optframe.tiers.simple_cli --task runs/tasks/silu__x --generator fixture --fixture-dir fixtures/silu

# 4) 跑测试
python -m pytest tests -q
```

## 设计要点

- **不信任自报数字**：所有候选都过「正确性 → 冷缓存交错 A/B → bootstrap CI → 账本」四道关，KEEP 需要收益下界显著大于 1。
- **换卡安全**：`preflight` 用实测 GEMM 峰值与 `arch.json` 对账，架构/工具链不一致直接 FAIL 或 WARN。
- **反作弊**：源码守卫拒绝改计时、改 judge、绕过输入的"优化"。
- **批次门槛**：回灌 kernel 只在注册时声明的 `min_m` 以上生效，避免把 prefill 定参的 kernel 塞给 decode 小 batch。

## 当前状态（2026-11-13）

- 阶段 1–5 全链路已跑通；`tests/` 98 用例全过。
- 算子级收益可复现（1.02× / 1.04×，配对 A/B + CI 下界 > 1）。
- 端到端收益仍在噪声内：prefill 形状覆盖率约 15.5%，理论效应 ~0.39%，低于当前 ±1.5% 测量噪声。
  下一步方向：融合链（多算子 DAG）任务模型、提高形状覆盖、换标准算子占比更高的 workload。

## 许可

见 `LICENSE`。
