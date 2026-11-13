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
3. **纯访存算子目前判不了**：`kernel_shapes.enrich` 里有 `if hint.avg_flops <= 0: continue`，FLOPs 为 0 的拷贝类算子会被跳过（待修）；
4. **profiler 窗口的时间会被放大**：只有占比可用，且需要在报告里注明。

---

## 7. 与其他契约的关系

- `evaluator.py`：正确性 → 冷缓存交错 A/B → bootstrap CI → 判定；
- `ledger.py`：每条记录都带 `cache_policy`（`cold` / `warm` / `profiler`，默认 `cold`，非法值直接抛错）；
- `source_guard.py`：拒绝"改计时/改 judge/绕输入"的伪优化。

**核心原则：宁可报 INCONCLUSIVE，也不要报一个不带口径的加速比。**
