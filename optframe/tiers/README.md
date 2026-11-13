# 阶段4 简单档（Triton）

## 用法

```bash
cd /model/wyy/optframe

# 预置候选（无 LLM 也能跑通全链路）
.venv/bin/python -m optframe.tiers.simple_cli \
  --task runs/stage3_tasks/silu__2048x2048 \
  --generator fixture --fixture-dir fixtures/silu --n 3 --rounds 2

# LLM 生成（GPUStack / OpenAI 兼容）
export GPUSTACK_API_KEY=...
.venv/bin/python -m optframe.tiers.simple_cli \
  --task runs/stage3_tasks/silu__2048x2048 --generator llm \
  --base-url http://10.51.198.60:10152/v1-openai --model qwen3.5-9b
```

## 判定口径（两道门槛）

1. **正确性**：`judge.py` 对拍 reference，容差按 dtype；不过直接 REVERT。
2. **性能**：`ratio = ref_ms / cand_ms`
   - 必须 `ratio >= 1.02`（方案规定的噪声门限）；
   - 必须通过**配对 bootstrap 95% 置信区间下界 > 1.0**。

第 2 条是实测逼出来的：同一候选在共享 GPU 上三次跑出 1.036 / 1.000 / 1.014，
判定在 KEEP/REVERT 之间反复翻转。只加点估计门槛等于掷硬币。

## 计时口径

- 交错 A/B：每轮先测基线再测候选，两侧经历同一段负载；
- 每次迭代前 flush 256MB，强制冷缓存（方案要求）；
- 取中位数，warmup 10 / repeat 50；
- 逐次比值再算 bootstrap CI（1000 次重采样）。

## 反作弊

候选源码在运行前做静态检查：

| 检查 | 处理 |
| --- | --- |
| 语法错误 / 缺 `run()` / 缺 `@triton.jit` | REVERT（不执行） |
| `torch.mm/softmax/silu/...` 等计算调用 | REVERT |
| `torch.nn` / `F.*` | REVERT |
| `torch.empty_like` / `x.numel()` / `x.shape` | 允许 |

原因：不拦的话，模型可以写 `return torch.mm(x, y)` 轻松"赢"，整条链路失效。

## 实测（L20，与 qwen3.5-9b 共享 GPU）

| 场景 | baseline | 候选 | ratio | CI 下界 | verdict |
| --- | --- | --- | --- | --- | --- |
| torch silu | 29.70 µs | 28.67 µs | 1.036 | 1.033 | KEEP |
| 慢实现做基线（仅验证链路） | 41.0 µs | 29.7 µs | 1.411 | 1.378 | KEEP |
| 故意写错数学 | 29.7 µs | — | — | — | REVERT（allclose failed） |
| 小 block（BLOCK=64, 1 warp） | 29.7 µs | 42.0 µs | 0.707 | — | REVERT |

注意：这里的 29.7µs 与 trace 里的 9.8µs 不矛盾 —— trace 是 L2 命中（8MB 张量 < 48MB L2），
bench 每次 flush 冷缓存。两侧同口径，比值仍有效，但**绝对数字不能和 trace 混用**。

## 已知限制

1. 候选在**同进程**执行；恶意/崩溃代码会带走整个进程。复杂档必须独立进程。
2. 共享 GPU 上 CI 更宽；验收应在独占 GPU 跑。
3. LLM 生成器当前受 GPUStack 队列影响（Waiting 10+，~1 tok/s），生产使用需排队或换端点。
4. `baseline` 默认是 reference（torch）。真实验收应传 `--baseline-module` 指向线上实现
   （如 FlagGems/SGLang 的融合算子），否则会和"更弱的基线"比出虚高收益。
