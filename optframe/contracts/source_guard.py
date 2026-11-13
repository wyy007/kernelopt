"""候选源码的静态检查：结构合法性与"禁止偷用 PyTorch 计算"。

对齐 KernelAgent 的规则（不直接 import 它的 worker，避免把它的重依赖拖进来），
但契约不同：我们的候选必须暴露模块级 `run(*args)`，而不是 `kernel_function`。
"""

from __future__ import annotations

import ast
import re
from typing import Optional

#: 允许的 torch 用法（分配/元信息），其余计算类调用一律禁止
_ALLOWED_TORCH = {
    "empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like",
    "cuda", "device", "dtype", "tensor", "Tensor", "no_grad", "is_grad_enabled",
}

_TORCH_CALL = re.compile(r"\btorch\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_F_CALL = re.compile(r"\bF\.[A-Za-z_][A-Za-z0-9_]*\s*\(")
_NN_USE = re.compile(r"\btorch\.nn\b")


def _strip_comments_and_strings(code: str) -> str:
    pattern = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#[^\n]*)')
    return pattern.sub("", code)


def detect_disallowed_torch(code: str) -> Optional[str]:
    """返回违规原因；合法则返回 None。"""
    sanitized = _strip_comments_and_strings(code)

    if _NN_USE.search(sanitized):
        return "torch.nn usage is not allowed in a kernel candidate"
    if _F_CALL.search(sanitized):
        return "torch.nn.functional (F.*) calls are not allowed"

    for match in _TORCH_CALL.finditer(sanitized):
        symbol = match.group(1)
        if symbol not in _ALLOWED_TORCH:
            return f"torch.{symbol}() computes the result; candidates must run their own kernel"

    return None


def validate_source(code: Optional[str], language: str = "triton") -> Optional[str]:
    """结构校验：语法、`run` 定义、triton 装饰器。返回错误原因或 None。"""
    if not code or not code.strip():
        return "empty candidate source"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return f"invalid Python syntax ({line}: {exc.msg})"

    has_run = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        for node in tree.body
    )
    if not has_run:
        return "missing top-level `run(*args)` definition"

    if language == "triton":
        has_jit = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    name = getattr(target, "attr", None) or getattr(target, "id", None)
                    if name == "jit":
                        has_jit = True
                        break
            if has_jit:
                break
        if not has_jit:
            return "no @triton.jit kernel found"

    return None


def check_candidate(code: Optional[str], language: str = "triton",
                    forbid_torch: bool = True) -> Optional[str]:
    """结构 + 反作弊的合并入口。"""
    err = validate_source(code, language)
    if err:
        return err
    if forbid_torch and language == "triton":
        return detect_disallowed_torch(code or "")
    return None
