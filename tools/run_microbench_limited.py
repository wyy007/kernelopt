"""Run TraceLens microbench with reduced shapes (GPU is shared with a live model).

Copied to /model/wyy/optframe/tools/run_microbench_limited.py on 10.51.198.60.
"""
import sys

from TraceLens.PerfModel.benchmarking import microbench as mb

# L20 total 44.4GB, ~10GB free -> keep per-case footprint under ~3GB.
mb.GEMM_SHAPES = [
    (8192, 8192, 8192),
    (16384, 8192, 7168),
    (16384, 1024, 8192),
]
mb.BW_SIZES = [256 * 1024 * 1024, 512 * 1024 * 1024]

out = sys.argv[1] if len(sys.argv) > 1 else "/model/wyy/optframe/arch/L20.json"
sys.argv = [
    "microbench",
    "--device", "0",
    "--allow-busy",
    "--skip-vector",
    "--output", out,
]
mb.main()
