# 阶段2：TraceLens 分析层 → bottlenecks.json

只做诊断，不改任何算子代码。输入 trace + arch.json，输出机器可读的瓶颈清单，供阶段3（`trace_to_tasks.py`）消费。

## 用法

```bash
cd /model/wyy/optframe

# 完整链路：跑 TraceLens 报表 + 导出 bottlenecks.json
.venv/bin/python -m optframe.analysis.cli \
  --trace runs/smoke/trace.json \
  --arch arch/L20_full.json \
  --workdir runs/smoke_stage2 \
  --framework sglang --model qwen3.5-9b

# 复用已有报表（调试解析逻辑时用，不重跑 TraceLens）
.venv/bin/python -m optframe.analysis.cli \
  --trace runs/smoke/trace.json \
  --arch arch/L20_full.json \
  --csv-dir runs/smoke/csvs_full \
  --workdir runs/smoke_stage2_replay

# 有 ncu 数据时，置信度可升到 measured
.venv/bin/python -m optframe.analysis.cli ... --ncu runs/smoke/ncu_topk.csv
```

产物（都在 `--workdir` 下）：

| 文件 | 用途 |
| --- | --- |
| `bottlenecks.json` | 机器可读，阶段3 的输入 |
| `topk.csv` | 人工快速扫读 |
| `analysis_deterministic.md` | 无大模型依赖的确定性摘要（不是方案里的 agent `analysis.md`） |
| `csvs/` `perf_report.xlsx` | TraceLens 原始报表，留痕用 |

## bottlenecks.json 契约

顶层：`schema_version / run_id / generated_at / source / target / params / summary / findings`

每条 finding 关键字段：

| 字段 | 说明 |
| --- | --- |
| `op` / `kernel` / `shape` / `dtype` | 定位到算子与 shape |
| `gpu_time_ms` / `pct_e2e` | 影响面（`pct_e2e` 分母 = `gpu_timeline.computation_time`） |
| `bound_type` | compute / memory / latency / launch / comm / unknown |
| `sol_pct` | roofline 达成率，>100% 会被截断并留痕 |
| `confidence` | measured / derived / heuristic / inconclusive |
| `recommended_action` | tune / fuse / backend / precision / investigate |
| `suggested_owner` | kernel / framework / serving |
| `eligible` / `excluded_reason` | 是否进入自动优化，以及为什么被排除 |
| `estimated_gain_pct` | `pct_e2e × (1 − SOL%)`，即 Amdahl 上限 |
| `evidence` | trace / csv / ncu / notes，**没有 trace 的结论不输出** |

JSON Schema：`optframe/analysis/bottlenecks.schema.json`

## 过滤规则（对应总体方案 §三）

| 规则 | 结果 |
| --- | --- |
| NCCL / collective / c10d | `excluded_reason=system_collective` |
| FlashMLA / DeepEP | `excluded_reason=hand_optimized_kernel` |
| memcpy / memset | `excluded_reason=system_memcpy` |
| `pct_e2e < --min-pct`（默认 2%） | `below_amdahl_threshold` |
| 无 roofline（bound=unknown） | `insufficient_evidence`，动作 `investigate` |
| `sol_pct ≥ --near-roofline`（默认 85%） | `near_roofline`，收益记 0 |

## 置信度规则（对应总体方案 §五 5.2）

- `measured`：**必须有 ncu 硬件计数器**，否则 `validate()` 直接报错；
- `derived`：trace + roofline（有 arch.json 且能算出 SOL）；
- `heuristic`：有耗时但无 roofline；
- `inconclusive`：证据不足。

> 注：`doc/自动算子优化系统-双档方案-20261105.md` 把"TraceLens 实测耗时 + arch spec"也算 `measured`，
> 与总体方案冲突。本实现按**总体方案**执行（更严格），因为 arch spec 是 microbench 拟合值，不等于硬件计数器。

## 已知限制

1. 只读 `unified_perf_summary.csv` + `gpu_timeline.csv`，不解析 xlsx；
2. 单卡 trace 没有 comm 分析，`exposed_comm_pct` 恒为 0；
3. `pct_e2e` 分母是 GPU 忙碌时间，不是 wall time —— 与报表自带 `Percentage (%)` 可能差几个点，差异会在 `evidence.notes` 里标注；
4. `sol_pct > 100%` 说明 TraceLens 对这类算子（尤其是 elementwise 的读写字节数）建模有误，
   实现上**直接丢弃该达成率并降级为 heuristic**，而不是截断成 100% —— 否则会把真实候选误判成"已贴顶"而漏掉。
   原始值保留在 `sol_pct_raw`，原因写进 `evidence.notes`。
