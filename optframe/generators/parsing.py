"""从模型回复里抽代码（对齐 KernelAgent worker._extract_code_from_response 的规则）。

规则保持一致：优先取 ```python 块；没有语言标记则取任意 ``` 块；
多个块时优先包含 run/kernel 定义的那个；完全没有围栏时，若整体像代码则整体返回。

不直接 import KernelAgent，是为了避免把它的重依赖（jinja2/provider/DB）拖进本框架；
规则本身是等价的，且这里的契约是 `run(*args)`。
"""

from __future__ import annotations

import re
from typing import Optional

_FENCE_PY = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
_FENCE_ANY = re.compile(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
_CODE_HINTS = ("import ", "from ", "def ", "class ", "@", '"""', "'''")


def extract_code(text: str, prefer: str = "run") -> Optional[str]:
    if not text or not text.strip():
        return None

    blocks = _FENCE_PY.findall(text)
    if not blocks:
        blocks = _FENCE_ANY.findall(text)

    if blocks:
        if len(blocks) > 1:
            needle = re.compile(rf"\bdef\s+{re.escape(prefer)}\b")
            for block in blocks:
                if needle.search(block):
                    return block.strip()
            return max(blocks, key=len).strip()
        return blocks[0].strip()

    # 围栏未闭合（输出被 max_tokens 截断）：从第一个 ```python 截到末尾
    opened = re.search(r"```(?:python|py)\s*\n", text)
    if opened:
        return text[opened.end():].strip()

    lines = text.strip().splitlines()
    if any(line.strip().startswith(hint) for line in lines for hint in _CODE_HINTS):
        return text.strip()
    return None
