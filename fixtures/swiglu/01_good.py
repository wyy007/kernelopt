import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(A, B, C, M, HALF_N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(HALF_N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    g_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    u_ptrs = B + rk[:, None] * stride_bk + (rn[None, :] + HALF_N) * stride_bn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_rem), other=0.0)
        g = tl.load(g_ptrs, mask=(rk[:, None] < k_rem) & (rn[None, :] < HALF_N), other=0.0)
        u = tl.load(u_ptrs, mask=(rk[:, None] < k_rem) & (rn[None, :] < HALF_N), other=0.0)
        acc_g += tl.dot(a, g)
        acc_u += tl.dot(a, u)
        a_ptrs += BLOCK_K * stride_ak
        g_ptrs += BLOCK_K * stride_bk
        u_ptrs += BLOCK_K * stride_bk

    out = acc_g * tl.sigmoid(acc_g) * acc_u
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < HALF_N))


def run(*args):
    a, b = args[0], args[1]
    M, K = a.shape
    HALF_N = b.shape[1] // 2
    c = torch.empty((M, HALF_N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 256, 64, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(HALF_N, BLOCK_N),)
    _swiglu_kernel[grid](
        a, b, c, M, HALF_N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=8, num_stages=3,
    )
    return c
