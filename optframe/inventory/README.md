# 阶段3：瓶颈 → 标准任务

## 已完成：Amdahl 过滤（`amdahl.py`）

阶段2 的"占比 ≥ 2%"只是粗筛。阶段3 要回答的是：
**这个算子就算优化到目标 SOL，端到端到底还能快多少？**

```bash
cd /model/wyy/optframe
.venv/bin/python -m optframe.inventory.cli \
  --report runs/stage2_final/bottlenecks.json \
  --workdir runs/stage3_amdahl
```

产物：`amdahl.json`（机器读，含 `target_ratio`）+ `amdahl.md`（人读）。

## 模型

```
T' = T − t + t/s          # t = p·T
e2e 加速比 = 1 / (1 − p·(1 − 1/s))
Amdahl 上限 = 1 / (1 − p)   # s → ∞
```

多个任务按**可加性**累加（GPU 时间串行），不做重叠假设 —— 宁可低估，不可虚高。

加速比 `s` 的来源：

| 情形 | s |
| --- | --- |
| 有 roofline 且 `sol < target_sol` | `target_sol / sol` |
| 有 roofline 且 `sol ≥ target_sol` | `1.0`（无可优化空间 → 噪声门限拦掉） |
| 无 roofline，action=fuse | 1.30（保守假设） |
| 无 roofline，action=tune | 1.20（保守假设） |

## 过滤顺序

| 顺序 | 规则 | reason |
| --- | --- | --- |
| 1 | 阶段2 已排除 | `stage2_excluded` |
| 2 | 动作不是 tune/fuse（backend/precision/investigate） | `action_not_optimizable` |
| 3 | `s < noise_floor`（默认 1.02） | `below_noise_floor` |
| 4 | 单任务 e2e 收益 `< min_e2e_gain`（默认 0.5%） | `below_e2e_gain_threshold` |
| 5 | 按收益降序后超过 `max_tasks` 或累计收益超过上限 | `budget_exceeded` |
| 6 | 计入前序任务后边际 `< min_marginal`（默认 0.1%） | `diminishing_returns` |

## 已完成：任务生成（`trace_to_tasks.py`）

```bash
.venv/bin/python -m optframe.inventory.trace_to_tasks \
  --amdahl runs/stage3_amdahl/amdahl.json \
  --outdir runs/stage3_tasks
```

每个任务一个目录：`<op>__<shape>/`，内含

| 文件 | 内容 |
| --- | --- |
| `task.toml` | name/op/arch/dtype/shapes/target_ratio/reference/judge/timing/escalation |
| `reference.py` | 可执行的 PyTorch 真值：`make_inputs()` + `run()` |
| `judge.py` | 按 dtype 容差对拍，返回 `{ok, atol, rtol, max_abs_err, max_rel_err}` |
| `notes.md` | 为什么选它、验收口径、升级原因 |

`index.json` 汇总生成/跳过/失败；**生成后会默认在 CPU 上跑一遍 `judge.py` 自检**，
代码不能执行就判失败（退出码 4），不会把坏任务交给下游。

### 只支持语义明确的算子

目前覆盖：`silu/gelu/relu/sigmoid/tanh/exp/neg`、`mul/add/sub/div/maximum`、
`mm/bmm/matmul`、`softmax/log_softmax/layer_norm/rms_norm/sum`。
其余 op **拒绝生成**（`unsupported_op`）——错误 reference 会让正确性校验整体失效，
比不生成任务坏得多。

### 档位升级：静态 / 运行期分离

| 类型 | 规则 | 实现 |
| --- | --- | --- |
| 静态 | attention/MoE/conv 或 fuse 任务 | `allow_complex=true` |
| 静态 | `sol_pct < 60` | `allow_complex=true` |
| 静态 | trace 显式建议 | `allow_complex=true` |
| **运行期** | best-of-3 提升 `< 1.05×` | `should_escalate_after_best_of_three()` |

运行期规则不做事前代理判断。`task.toml` 里写 `runtime_rule = "best_of_3_lt_1.05"`，
由 tiers 跑完 best-of-3 后调用判定。

## 已知限制

- **融合任务被低估**：`silu`+`mul` 这类应当合成一个 kernel 的候选，当前按单算子 1.3× 估算，
  实际融合收益更高。要修得先拿到 trace 里的执行顺序/数据依赖，属于后续增强；
  在此之前宁可用保守值，也不虚报。
- 无 roofline 的算子只能给默认加速比；接上 ncu 后可用 `sm/dram` 实测改进目标值。
- 多卡场景要额外考虑通信重叠，当前模型不覆盖。
