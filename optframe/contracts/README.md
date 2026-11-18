# 计时与评测契约（contracts）

这个目录管的是**"数字可不可信"**：任务加载、计时、judge、账本、CUDA 加载、源码守卫。
其中最容易出错的不是代码，而是**测量口径**。这份文档把它固定下来，任何评测结论都要按这里写。

---

## 1. 为什么要有这条约定

2026-11-13 对同一个交付 kernel（`delivery/optframe_delivery/kernel_mm_5104x1024_7168.py`）做了两组测量：

| 口径 | 5104×1024×7168 | 5104×1024×5120 |
| --- | --- | --- |
| 热缓存（不冲 L2） | **0.959×**（比 cuBLAS 慢 4%） | **0.956×** |
| 冷缓存（每次调用前冲 256MB） | **1.018×**（快 1.8%） | **1.043×**（快 4.3%） |

**同一份代码、同一张卡、同一个形状，结论方向相反，差 8 个百分点——比收益本身还大。**

原因：B 矩阵 14.7MB 可以落在 L2 里，热缓存时 cuBLAS 吃 L2 更快；冷缓存时两者都从 HBM 流式取权重，Triton 的 `BLOCK 128×128×32 + GROUP_M=8` 更省带宽。

所以：**不声明缓存口径的加速比，等于没有结论。**

---

## 2. 三条铁律

### 2.1 必须声明口径

任何 ratio 必须附带三件事：`cache_policy`、`n`（配对数）、`CI`（置信区间）。
缺任意一项，结论只能进"观察"，不能进 KEEP。

### 2.2 必须交错 A/B

共享 GPU 上"先全测 A、再全测 B"会被环境漂移污染（历史实测出现过 15% 的跳变）。
正确做法是逐对交替：A→B→A→B…，每对求比值，再对**比值序列**取中位数。

### 2.3 必须用非 profiler 计时

torch profiler 的绝对时间不可用于验收：

| kernel | profiler | CUDA event | 放大 |
| --- | --- | --- | --- |
| FP8 分块 GEMM（M=80） | 1.029 ms | 0.049 ms | **21×** |
| MoE 融合核（M=80） | 0.910 ms | 0.637 ms | 1.4× |

profiler 只用于**相对占比**（"时间去哪了"），绝对时间一律用 `torch.cuda.Event`。

---

## 3. 代码里怎么落实

`optframe/contracts/timing.py` 已经内置了这套口径：

```python
from optframe.contracts.timing import time_pair_samples, bootstrap_median_ci

ref_s, cand_s = time_pair_samples(fn_ref, fn_cand, warmup=10, repeat=50, flush_cache=True)
ratios = [r / c for r, c in zip(ref_s, cand_s)]
lo, hi = bootstrap_median_ci(ratios)
```

要点：

- `flush_cache=True`（默认）：每次迭代前 `_flush_cache()` 写 256MB 缓冲区，覆盖 L20 的 48MB / Blackwell 的 L2；
- `time_pair_samples` 是**逐对交错**，返回同一时间窗内的两组样本；
- `bootstrap_median_ci` 对配对比值做 bootstrap，给出 95% 区间。

### KEEP 判定

```
KEEP     ⇔ CI 下界 > 1 + noise_floor        # 默认 noise_floor = 0.02
REVERT   ⇔ CI 上界 < 1 + noise_floor
INCONCLUSIVE ⇔ 区间跨越门限
```

只看点估计（比如 1.018）不看 CI 下界（1.012）就判 KEEP，等于把噪声当收益。

---

## 4. 报告必须写的字段

| 字段 | 说明 |
| --- | --- |
| `cache_policy` | `cold`（冲 L2）/ `warm`（不冲）——**必填** |
| `n` | 配对数，建议 ≥ 30 |
| `ref_ms` / `cand_ms` | 各自中位数（同口径） |
| `ratio` + `ci` | 配对比值中位数 + 95% bootstrap CI |
| `verdict` | KEEP / REVERT / INCONCLUSIVE |
| `timer` | `cuda_event` / `profiler`（profiler 仅限占比分析） |

---

## 5. 什么时候可以用热缓存

只有一种情况：**你就是要测"权重已在 L2"的场景**（例如连续调用同一层、权重常驻）。
这时必须在结论里显式写 `cache_policy=warm`，并说明为什么这个场景成立。

否则一律 `cold`。

---

## 6. 已知的边界

1. **冷缓存不等于真实 serving**：真实推理里权重被多请求轮换，介于冷热之间。冷缓存是**上界口径**（对 Triton 有利），报告时要说清楚；
2. **小 batch 会被固定开销淹没**：GDN decode 在 B≤16 时耗时 77–79µs 不随 batch 变化，属于延迟受限，此时 SOL 低不代表有空间；
3. **纯访存算子按带宽判**：`kernel_shapes.enrich` 对 FLOPs=0 且 bytes>0 的算子走纯带宽分支（2026-11-13 修复），不再跳过；两侧都缺才跳过；
4. **profiler 窗口的时间会被放大**：只有占比可用，且需要在报告里注明。

---

## 7. 与其他契约的关系

- `evaluator.py`：正确性 → 冷缓存交错 A/B → bootstrap CI → 判定；
- `ledger.py`：每条记录都带 `cache_policy`（`cold` / `warm` / `profiler` / `n/a`，默认 `cold`；`n/a` 表示未计时的记录，如静态检查；非法值直接抛错）与 `timer`（`cuda_event` / `profiler`）；口径由 `evaluator` 的 `flush_cache` 参数产生，不靠调用方记；
- `source_guard.py`：拒绝"改计时/改 judge/绕输入"的伪优化。

**核心原则：宁可报 INCONCLUSIVE，也不要报一个不带口径的加速比。**

---

## 8. 静态检查（lint）与见证式反馈

上 GPU 之前先跑 `contracts/lint.py`：能在编译期判定的问题不该花一次评测的成本，而且反馈要能定位。

### 检查项

| code | 级别 | 判据 |
| --- | --- | --- |
| `syntax_error` | error | `ast.parse` 失败 |
| `run_missing` / `run_no_return` | error | 没有 `run(*args)`，或有但没有 return |
| `run_arity` | error | `run` 位置参数个数与任务输入张量数不一致（有 `*args` 时不判） |
| `import_*_missing` | error | 缺 `torch` / `triton` / `triton.language as tl` |
| `jit_missing` | error | 没有 `@triton.jit` kernel |
| `mask_missing` | error | 形状不被 BLOCK 整除，但源码里没有任何 `mask=` |
| `smem_overflow` | error | 估算 `(BM·BK + BK·BN)·2·stages` 超过 227KB |
| `warps_invalid` | error | `num_warps ∉ {1,2,4,8,16,32}` |
| `torch_compute` | error | 调用了 `torch.mm` 等计算算子（复用 `source_guard`） |
| `acc_dtype_unclear` | warning | 用了 `tl.dot` 但没有 `tl.float32` 累加 |
| `block_not_power_of_two` | warning | BLOCK 不是 2 的幂 |

### 见证格式

失败不是一句话，而是可定位的反例，直接进重试 prompt：

```
[mask_missing] 第 42 行: kernel 里没有 `mask=`，而 M=7835 不是 BLOCK_M=128 的整数倍，最后一块会越界读/写
  见证: M=7835, BLOCK_M=128, 余数=27, 越界行区间=[7808, 7963)
```

judge 侧同理：`allclose` 失败时返回首个最大误差的坐标与两侧数值（`index` 字段），不再只有 `allclose failed`。

### 落点

- `tiers/simple.py`：候选进 `evaluate_fn` 之前先 lint；`errors` 非空 → 记 `INVALID`（`phase=lint`, `cache_policy=n/a`）并进重试队列，不占 GPU；
- `warnings` 单独记一条 `LINT_WARN`，只留痕不阻断；
- `run_simple_tier(..., lint=False)` 可关闭（供编排测试使用）。

---

## 9. 端到端 A/B（e2e）判定口径

§3 判的是 kernel 冷缓存加速比；e2e 判的是"开/关回灌之后的请求延迟比"。两者的噪声量级不同，
门限不能共用：kernel 级噪声 2%（`noise_floor_speedup = 1.02`），e2e 端到端的块间漂移是 1.5%–2.0%。

### 量怎么算

- `lat_ratio = p50(off) / p50(on)`，> 1 表示开回灌更快；配对数 `n` = 有效对（on 块融合命中 > 0 且两侧 p50 有效）；
- 计时是 HTTP 请求墙钟（不是 CUDA event）；`--flush` 每块前冲 KV/radix cache，prompt 注入随机 nonce 强制 cache miss；
- 每块固定时长打流，按 `(off_i, on_i)` 配对求比值，对**比值序列**取中位数 + 95% bootstrap CI。

### 门限

```
min_gain = 1 + noise_floor_e2e        # 默认 noise_floor_e2e = 0.015，即 --min-gain 1.015
```

依据（2026-11-18，6 块 × 20s）：块间漂移实测 1.5%–2.0%（off 1825→1862ms、on 1816→1852ms，各 +2.0%）。
配对能抵消同步漂移，但抵消不掉的是"每对里 on 总在第二个"的顺序偏倚，量级也在 1% 以上。
n=3 时 bootstrap CI 半宽会缩到 0.02%–0.4%（2026-11-18 CI [1.0050, 1.0054]、2026-11-17 第 3 轮 CI [0.9995, 1.0073]），
**CI 半宽是"配对内的精度"，不是"结论的可信度"，所以门限必须外部给定，不能由 CI 自己决定。**

### 判定

| verdict | 条件 | 含义 |
| --- | --- | --- |
| `KEEP` | `n ≥ 2` 且 `CI 下界 > min_gain` | 噪声之上确有收益，可以默认开启 |
| `REVERT` | `CI 上界 < 1.0` | 确定比基线慢，不要开 |
| `INCONCLUSIVE` | 其余 | 判不了，维持现状 |

`INCONCLUSIVE` 必须带 `verdict_reason`：

| reason | 含义 |
| --- | --- |
| `insufficient_pairs` | `n < 2`，单对没有区间可言 |
| `no_pairs` | 没有有效配对（on 块无命中/数据缺失） |
| `below_noise_floor` | `CI 上界 ≤ min_gain`：连收益上限都没到门限（典型 0.5%） |
| `ci_spans_threshold` | 下界 ≥ 1.0 但上界 > `min_gain`：CI 跨过门限，方向对但样本/漂移不足 |
| `ci_spans_baseline` | 下界 < 1.0 且上界 > `min_gain`：区间同时跨基线与门限，方向不明 |

### 与 §3 的差别（为什么 REVERT 的含义不同）

§3 的 `REVERT ⇔ CI 上界 < 1 + noise_floor`，因为那边 `REVERT` 表示"丢掉这个候选"；
e2e 的 `REVERT` 只表示"确定比基线慢"，因为它的用途是**决定要不要默认开启**，而"略好但低于噪声"
既不该说成收益，也不该说成回退——那种情况记 `INCONCLUSIVE(below_noise_floor)`，
报告里写"方向一致、量级不足"。

### 顺序要求

- `--order alternate`（默认）off→on→off→on，配对是相邻块，但每对里 on 总在第二个；
- `--order abba` off→on→on→off，配对自然变成 (b0,b1)、(b3,b2)…，on 块在一半配对里排在第一个；
  **验“位置偏倦”时必须用这个**，块数要是 4 的倍数（8 块 = 4 对），产物 `pairs[]` 带
  `off_block`/`on_block`/`on_first` 可核对。

### 落点

- `tools/e2e_ab.py`：`decide()` / `summarize_pairs()` 是唯一判定实现，`--min-gain` 默认 1.015；
- `optframe/webui/e2e.py`：透传 `--min-gain`，**只读** `e2e_ab.json` 里的 `verdict`，不自己再判一遍；
  退出码 `KEEP=0`、其余 `1`（动作的 `ok_codes` 已把 1 当成功，不代表执行失败）；
- 两个产物文件都必须带 `min_gain` / `noise_floor` / `n_pairs` / `verdict` / `verdict_reason`；
- 测试：`tests/test_e2e_ab.py` 覆盖五个分支（含 `CI [1.005, 1.005]` 在 `--min-gain 1.015` 下必须是 `INCONCLUSIVE`）。
