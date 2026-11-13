"""合并多次 microbench 日志，逐项取最大值作为可达峰值。

争用环境下单次测量会偏低；取多次的最大值更接近"空闲时能达到的峰值"，
对 roofline 判断是保守方向（峰值越高 → SOL 越低 → 越不会误判已到顶）。
"""
import json, re, sys
from pathlib import Path

rows = {}
for log in sys.argv[1:-1]:
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = re.match(r"\s*(matrix_\w+|vector_\w+|read_bw_gbps|write_bw_gbps)\s+([\d.]+)\s+(TFLOPS|GB/s)", line)
        if m:
            key, val = m.group(1), float(m.group(2))
            rows[key] = max(rows.get(key, 0.0), val)

out = Path(sys.argv[-1])
data = {
    "name": "Blackwell",
    "mem_bw_gbps": round(max(rows.pop("read_bw_gbps", 0.0), rows.pop("write_bw_gbps", 0.0)), 1),
    "memory_gb": 71.1,
    "max_achievable_tflops": {k: round(v, 1) for k, v in sorted(rows.items())},
    "source": "These are benchmark derived peak flops and bw (max over runs)",
}
out.write_text(json.dumps(data, indent=4))
print(json.dumps(data, indent=2))
