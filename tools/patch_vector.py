"""补测 vector TFLOPS 并合并进 arch.json（microbench 全量跑会被显存挤死）。"""
import json, sys
from pathlib import Path
from TraceLens.PerfModel.benchmarking.microbench import bench_vector_tflops

path = Path(sys.argv[1])
res = bench_vector_tflops(0)
data = json.loads(path.read_text())
data["max_achievable_tflops"].update(res)
path.write_text(json.dumps(data, indent=4))
print("merged:", res)
