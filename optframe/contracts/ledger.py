"""JSONL 账本：每一个候选都必须留痕，且可复现（双档方案 §4.3）。

设计：
- 只追加，不改写；每行一个 JSON；
- 每行写完即 flush + fsync，进程崩溃也不丢已产生的判定；
- 记录 sha 而不是整份源码（源码另存 artifacts），避免账本膨胀。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: 允许的计时口径。见 contracts/README.md：不声明口径的加速比不可信。
#: cold = 每次计时前冲 L2（默认，最接近 serving）；warm = 权重常驻 L2；
#: profiler = 来自 torch profiler 的时间（只能用于占比，不能作为验收依据）。
CACHE_POLICIES = ("cold", "warm", "profiler")


class Ledger:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def append(
        self,
        *,
        stage: str,
        verdict: str,
        payload: Dict[str, Any],
        task: str = "",
        generator: str = "",
        candidate_sha: Optional[str] = None,
        cache_policy: str = "cold",
    ) -> Dict[str, Any]:
        if cache_policy not in CACHE_POLICIES:
            raise ValueError(
                f"cache_policy 必须是 {CACHE_POLICIES} 之一，收到 {cache_policy!r}"
            )
        record = {
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            "stage": stage,
            "task": task,
            "generator": generator,
            "candidate_sha": candidate_sha,
            "cache_policy": cache_policy,
            "verdict": verdict,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def read(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def iter_verdicts(self, verdict: str) -> Iterable[Dict[str, Any]]:
        return (rec for rec in self.read() if rec.get("verdict") == verdict)
