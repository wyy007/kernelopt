"""CUDA 候选加载：把 kernel.cu 编译成可调用的 Python 模块。

ForgeKernel 产出的 kernel.cu 统一用
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
导出，和 Triton 侧的 `run(*args)` 契约一致 —— 这是两档能共用 evaluator 的前提。

编译走 torch.utils.cpp_extension，产物落在指定 build 目录，按源码 sha 命名以便复用缓存。
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Tuple

#: kernel.cu 首行的编译参数声明，ForgeKernel 用它固定 arch
_FLAG_LINE = re.compile(r"//\s*fk-cuda-cflags:\s*(.+)")
_ARCH_GENCODE = re.compile(r"^-gencode=|-^arch=")


def device_arch() -> Optional[str]:
    """返回当前卡的 sm_xxx（以实际设备为准，不信任务里写的）。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        return None
    return f"sm{major}{minor}"


def _cc(arch: str) -> str:
    """sm_120 -> 12.0；sm_89 -> 8.9。"""
    m = re.fullmatch(r"sm(\d{2,3})", arch or "")
    if not m:
        return "8.9"
    digits = m.group(1)
    if len(digits) == 2:
        return f"{digits[0]}.{digits[1]}"
    return f"{digits[:2]}.{digits[2]}"


_ARCH_CC = {f"sm{d}{m}": f"{d}.{m}" for d in range(7, 13) for m in range(0, 10)}


def parse_declared_flags(source: str) -> List[str]:
    """读取 `// fk-cuda-cflags: ...` 声明的编译参数。"""
    for line in source.splitlines()[:10]:
        match = _FLAG_LINE.search(line)
        if match:
            try:
                return shlex.split(match.group(1))
            except ValueError:
                return []
    return []


def _module_name(source: str, arch: str) -> str:
    # 名字里带 arch：torch 按 name+源码缓存，不重建；换卡必须换名字
    return f"fkcuda_{hashlib.sha1(source.encode()).hexdigest()[:10]}_{arch}"


def _cuda_lib_dir() -> Optional[str]:
    """定位 libcudart 所在目录（不同安装布局不一样）。"""
    candidates = [
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
        "/usr/local/cuda-12.8",
    ]
    for home in candidates:
        if not home:
            continue
        for sub in ("lib64", "lib", "targets/x86_64-linux/lib"):
            path = Path(home) / sub
            if (path / "libcudart.so").exists():
                return str(path)
    return None


def _find_cudart12() -> Optional[Path]:
    """找 torch 自带的 CUDA 12 runtime（与 torch 版本匹配）。"""
    import glob

    patterns = []
    for base in sys.path:
        patterns.append(f"{base}/nvidia/cuda_runtime/lib/libcudart.so.12")
    for pattern in patterns:
        hits = glob.glob(pattern)
        if hits:
            return Path(hits[0])
    return None


def find_torch_cudart() -> Optional[Path]:
    """对外暴露：找 torch 自带的 CUDA runtime（preflight 用来判断跨版本兼容性）。"""
    return _find_cudart12()


def _compat_cuda_home(build_root: Path) -> Optional[Path]:
    """构造一个 CUDA_HOME，使 nvcc 用 13、libcudart 用 12。

    本机只装了 CUDA 13 toolkit，但 torch 是 cu128。直接链 13 的 cudart 会让
    进程里同时存在两套 CUDA runtime，kernel 启动静默不执行（输出全 0）。
    所以做一个兼容根：bin/include 指向 13，lib64 的 libcudart 指向 torch 自带的 12。
    """
    real_home = Path("/usr/local/cuda")
    if not (real_home / "bin" / "nvcc").exists():
        return None
    cudart12 = _find_cudart12()
    if cudart12 is None:
        return None

    root = build_root / "cuda_compat"
    (root / "lib64").mkdir(parents=True, exist_ok=True)
    (root / "bin").mkdir(parents=True, exist_ok=True)

    link = root / "lib64" / "libcudart.so"
    if not link.exists():
        try:
            link.symlink_to(cudart12)
        except OSError:
            return None
    for name in ("nvcc",):
        target = real_home / "bin" / name
        lpath = root / "bin" / name
        if target.exists() and not lpath.exists():
            try:
                lpath.symlink_to(target)
            except OSError:
                return None
    inc = root / "include"
    if not inc.exists():
        try:
            inc.symlink_to(real_home / "include")
        except OSError:
            return None
    return root


def compile_cuda(
    source: str,
    arch: str,
    build_dir: str | Path,
    *,
    extra_flags: Optional[List[str]] = None,
    extra_include: Optional[List[str]] = None,
    timeout_note: str = "",
) -> Tuple[Optional[object], Optional[str]]:
    """编译并加载 CUDA 扩展，返回 (module, error)。"""
    build_dir = Path(build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    name = _module_name(source, arch)
    cu_path = build_dir / f"{name}.cu"
    if not cu_path.exists():
        cu_path.write_text(source, encoding="utf-8")

    flags = ["-O3"]
    # 任务/内核里声明的 -gencode/-arch 是生成时的卡；评测以当前设备为准，
    # 否则会出现 cubin 与设备不匹配（“no kernel image available”）。
    for flag in parse_declared_flags(source):
        if "-gencode" in flag or flag.startswith("-arch"):
            continue
        flags.append(flag)
    if extra_flags:
        flags.extend(extra_flags)
    cc = _cc(arch)
    flags.append(f"-gencode=arch=compute_{cc.replace('.', '')},code=sm_{cc.replace('.', '')}")

    ldflags: List[str] = []
    compat = _compat_cuda_home(build_dir)
    if compat is not None:
        os.environ["CUDA_HOME"] = str(compat)
        ldflags.append(f"-Wl,-rpath,{compat / 'lib64'}")
    lib_dir = _cuda_lib_dir()
    if lib_dir and compat is None:
        ldflags.append(f"-L{lib_dir}")
        # 运行期也要能找到 libcudart（否则 import 时 dlopen 失败）
        ldflags.append(f"-Wl,-rpath,{lib_dir}")
        os.environ.setdefault("CUDA_HOME", str(Path(lib_dir).parent))

    os.environ["TORCH_CUDA_ARCH_LIST"] = cc
    ext_dir = build_dir / "torch_ext"
    ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ext_dir))

    # torch 的 cpp_extension 需要 PATH 里有 ninja；直接跑 .venv/bin/python 时
    # 当前 shell 的 PATH 不一定包含 venv/bin，这里补上。
    venv_bin = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + path

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # noqa: BLE001
        return None, f"torch cpp_extension unavailable: {exc}"

    try:
        module = load(
            name=name,
            sources=[str(cu_path)],
            extra_cuda_cflags=flags,
            extra_cflags=["-O3"],
            extra_ldflags=ldflags,
            extra_include_paths=extra_include or [],
            build_directory=str(ext_dir),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - 编译失败要原样回传
        text = str(exc)
        return None, text[-3000:]

    return module, None


def validate_cuda_source(source: str) -> Optional[str]:
    """静态检查：必须是带 run 绑定的 torch 扩展。"""
    if not source or not source.strip():
        return "empty CUDA source"
    if "PYBIND11_MODULE" not in source:
        return "CUDA candidate must export a PYBIND11_MODULE"
    if not re.search(r'm\.def\s*\(\s*"run"', source):
        return 'CUDA candidate must bind m.def("run", ...)'
    return None
