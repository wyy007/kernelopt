# optframe — 全链路推理算子自动化优化框架

代码位置：`/model/wyy/optframe`（`/model` 是持久盘）。
方案文档：`doc/全链路推理算子自动化优化完整方案.md`。

## 链路与入口

```
trace ─(阶段2)─> bottlenecks.json ─(阶段3)─> amdahl.json ─> task.toml
                                                              │
                                          (阶段4) 简单档 Triton / 复杂档 ForgeKernel
                                                              │
                                                      ledger + 复测结果
```

| 阶段 | 命令 |
| --- | --- |
| 2 分析 | `python -m optframe.analysis.cli --trace T --arch arch/X.json --workdir runs/x` |
| 3 Amdahl | `python -m optframe.inventory.cli --report .../bottlenecks.json --workdir runs/y` |
| 3 任务 | `python -m optframe.inventory.trace_to_tasks --amdahl .../amdahl.json --outdir runs/tasks` |
| 4 简单档 | `python -m optframe.tiers.simple_cli --task runs/tasks/silu__x --generator fixture --fixture-dir fixtures/silu` |
| 4 复杂档 | `python -m optframe.tiers.complex_cli --task ... --artifact .artifacts/<run> --skip-run` |
| 环境自检 | `python -m optframe.preflight` |


## 模型角色（别混）

| 角色 | 模型 | 用途 |
| --- | --- | --- |
| **驱动（driver）** | `ds-v4` | 生成/改写 kernel 的 LLM |
| **被优化对象（target）** | `qwen3.5-9b-test` | 要采集 trace、找瓶颈、优化其推理服务的模型 |
| 冒烟/回归 | `qwen3-0.8b` | 只验证链路通不通，不参与质量评估 |

> 2026-11-10 曾把 `qwen3.5-9b-test` 误当驱动模型跑了一轮生成测试（结果 0/6 与 3/6），
> 那组数据只说明"弱模型当驱动时的表现"，**不是本项目的驱动基线**。驱动一律用 `ds-v4`。

## 目录

| 目录 | 职责 |
| --- | --- |
| `contracts/` | 任务加载、计时、judge、账本、CUDA 加载、源码守卫 |
| `analysis/` | TraceLens 报表解析 → `bottlenecks.json` |
| `inventory/` | Amdahl 过滤、任务生成（task.toml/reference/judge） |
| `generators/` | Triton 候选生成（LLM / fixture） |
| `tiers/` | 简单档编排、复杂档 ForgeKernel 适配器 |
| `tests/` | 90 个用例（大部分不需要 GPU） |

## preflight：换卡/换环境先跑这个

```bash
cd /model/wyy/optframe
.venv/bin/python -m optframe.preflight --tasks runs/stage3_tasks runs/stage3_tasks_97 \
  --json runs/preflight.json
```

检查项与判定：

| 检查 | FAIL 条件 | WARN 条件 |
| --- | --- | --- |
| `device` | 无 CUDA / 看不到卡 | — |
| `toolkit` | nvcc 与 torch CUDA 跨大版本且找不到兼容 cudart | — |
| `triton` | — | triton 目标架构 ≠ 设备 |
| `arch.json` | 目录里没有基线 | 基线名字与设备不匹配（SOL 会失真） |
| `tasks` | — | 任务的 `arch` 与设备不一致（评测会自动覆盖，但排序过期） |
| `hardcoded_gencode` | — | 源码里写死的 `-gencode/sm_XX` 与设备不符 |
| `hw_kb` | — | 复杂档缺当前架构的知识库 |

退出码：有 FAIL → 1；`--strict` 时 WARN 也返回 1。

为什么需要它：换卡后最危险的不是报错，而是**内核静默不执行、输出全 0**。
2026-11-09 从 L20（sm_89）换到 RTX PRO 5000（sm_120）就踩过这个坑。
