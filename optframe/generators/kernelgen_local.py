"""简单档生成器：调用 OpenAI 兼容接口产出 Triton 候选。

接口只依赖标准库 urllib，避免在目标环境为 HTTP 客户端再加依赖。
生产环境走 GPUStack（`/v1-openai/chat/completions`），也兼容任何 OpenAI 兼容端点。

重要：生成器只负责"提出候选"，性能数字一律由 evaluator 复测。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional

from ..contracts.schema import Candidate
from ..contracts.task import Task
from .base import GenContext
from .parsing import extract_code

SYSTEM_PROMPT = """你是 GPU kernel 工程师，只输出可执行的 Triton 代码。

硬性约束：
1. 只输出一个 ```python 代码块，不要解释、不要测试代码。
2. 代码块的第一行必须是 `import torch`；随后是 `import triton` 和 `import triton.language as tl`。
   缺任何一条都会在运行时报 NameError，直接算失败。
3. 必须有模块级函数 `run(*args)`，返回输出张量（与 reference 同形状/同 dtype）。
4. 计算必须由 @triton.jit kernel 完成。
5. 禁止调用任何 torch 计算算子（如 torch.mm / torch.softmax / torch.nn.functional）；
   `torch.empty_like`、`torch.empty`、`x.numel()`、`x.shape` 允许。
6. 输入张量在 CUDA 上、连续布局。
"""

#: 严格模式：给弱模型一个可填空的骨架，只让它改计算那一行
SKELETON = '''import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x  # <<< TODO: 在这里实现算子，例如 silu 就是 x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


def run(*args):
    x = args[0]
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    _kernel[grid](x, out, n_elements, BLOCK=BLOCK, num_warps=4)
    return out
'''


def build_prompt(task: Task, context: GenContext, prompt_mode: str = "default") -> str:
    shape_desc = "; ".join(
        f"args[{i}]: shape={dims}, dtype={task.dtype_name}" for i, dims in enumerate(task.shapes)
    )
    lines = [
        f"算子: {task.op}",
        f"目标 GPU: {task.arch}",
        f"输入: {shape_desc}",
        "",
        "请实现与 PyTorch 语义等价的 Triton kernel，并暴露 run(*args)。",
    ]
    if prompt_mode == "strict":
        lines.extend([
            "",
            "严格模式：只输出下面这个骨架，把 TODO 那一行改成正确计算。",
            "不要改函数签名，不要删任何行，不要加别的代码。",
            "```python",
            SKELETON.rstrip(),
            "```",
        ])
    if context.errors:
        lines.append("")
        lines.append("上一轮候选失败，错误如下（必须修复）：")
        for err in context.errors[-3:]:
            lines.append(f"- {err[:600]}")
    if context.notes:
        lines.append("")
        lines.append("补充要求：")
        lines.extend(f"- {n}" for n in context.notes[-3:])
    return "\n".join(lines)


def _decode_response(raw: str) -> dict:
    """解析聊天响应：兼容标准 JSON 与被包成 SSE 的 JSON。

    ds-v4 上游会把非流式响应包成 `data: {...}`（Content-Type 却写 application/json），
    直接 json.loads 会报 JSONDecodeError。这里两种都接。
    """
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    content_parts: list[str] = []
    last_obj = None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        last_obj = obj
        try:
            delta = obj["choices"][0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
        except (KeyError, IndexError, TypeError):
            pass

    if content_parts:
        return {"choices": [{"message": {"content": "".join(content_parts)}}]}
    if last_obj is not None:
        return last_obj
    raise json.JSONDecodeError("not JSON and not SSE", raw[:200], 0)


class KernelgenLocal:
    """OpenAI 兼容端点上的本地生成器。"""

    name = "kernelgen_local"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        *,
        timeout_s: int = 600,
        temperatures: tuple = (0.2, 0.6, 1.0),
        prompt_mode: str = "default",
        max_tokens: int = 3000,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("GPUSTACK_API_KEY", "")
        self.timeout_s = timeout_s
        self.temperatures = temperatures
        self.prompt_mode = prompt_mode
        self.max_tokens = max_tokens

    def _chat(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                # GPUStack 新版只认 X-API-Key（Bearer 会 401）；两个都发，兼容不同版本
                "X-API-Key": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8", "replace")
        body = _decode_response(raw)
        return body["choices"][0]["message"]["content"]

    def propose(self, task: Task, context: GenContext, n: int = 1) -> List[Candidate]:
        prompt = build_prompt(task, context, self.prompt_mode)
        out: List[Candidate] = []
        for i in range(n):
            temperature = self.temperatures[i % len(self.temperatures)]
            started = time.time()
            try:
                text = self._chat(prompt, temperature)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                out.append(
                    Candidate(
                        source="",
                        language="triton",
                        generator=f"{self.name}:error",
                        meta={"error": f"{type(exc).__name__}: {exc}", "temperature": temperature},
                    )
                )
                continue

            code = extract_code(text, prefer="run")
            if not code:
                out.append(
                    Candidate(
                        source="",
                        language="triton",
                        generator=f"{self.name}:no_code",
                        meta={"temperature": temperature, "latency_s": round(time.time() - started, 2)},
                    )
                )
                continue

            out.append(
                Candidate(
                    source=code,
                    language="triton",
                    generator=self.name,
                    meta={
                        "model": self.model,
                        "temperature": temperature,
                        "latency_s": round(time.time() - started, 2),
                    },
                )
            )
        return out
