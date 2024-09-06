import triton
from triton import language as tl
import torch


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b

def previous_powe_of2(x):
    p = 1
    while p < x:
        p *= 2
    if p == x:
        return x
    else:
        return p // 2

@triton.jit
def softmax_kernel_online(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)


    m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

    # specialization does not improve performance inn this example, as tested
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        m_new = tl.maximum(m, inp)
        alpha = tl.exp(m - m_new)
        z = z * alpha + tl.exp(inp - m_new)
        m = m_new

    m_reduced = tl.max(m, 0)  # (TILE_K,)
    z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
    m = m_reduced

    # specialization does not improve performance inn this example, as tested
    previous_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, N, TILE_N):
        n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
        offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        o = tl.exp(inp - m[None, :]) / z[None, :]
        tl.store(output_ptr + offsets, o, mask=mask)


def softmax(x, TILE_K):
    inp = x.contiguous()
    M = 1
    N, K = inp.shape
    out = torch.empty_like(x)
    TILE_N = min(triton.cdiv(8192, TILE_K), previous_powe_of2(N))
    grid = (triton.cdiv(K, TILE_K), M, 1)
    # print(TILE_K, grid)
    softmax_kernel_online[grid](
        out,
        inp,
        M,
        N,
        K,
        TILE_N,
        TILE_K,
    )
    return out


import pytest
import flag_gems

@pytest.mark.parametrize("n", [10, 128])
@pytest.mark.parametrize("m", [512, 1024, 32 * 1024])
@pytest.mark.parametrize("TILE_K", [1, 2, 4])
def test_softmax(m, n, TILE_K):
    x = torch.randn((m, n), device="cuda")
    hyp = softmax(x, TILE_K)
    ref = torch.softmax(x, dim=0)
    torch.testing.assert_close(hyp, ref)

def benchmark_softmax(m, n, tile_k):
    x = torch.randn((m, n), device="cuda")
    t1 = triton.testing.do_bench(lambda: softmax(x, tile_k), return_mode="median")
    def throughput(t):
        return x.numel() * x.element_size() * 2 * 1e-9 / (t * 1e-3)
    return throughput(t1)

import pandas as pd
def run_benchmark():
    records = []
    for n in [128, 256, 512, 1024, 2048, 4096, 8192]:
        for m in [128 * 1024]:
            if m * n * 4 > 1024 * 1024 * 1024 * 4:
                continue
            for tile_k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                t1 = benchmark_softmax(m, n, tile_k)
                record = (m, n, tile_k, t1, triton.cdiv(n, tile_k))
                records.append(record)
    df = pd.DataFrame.from_records(records, columns=["reduce_size", "post_size", "tiel_k", "online", "grid"])
    print(df)
    df.to_excel("online_outer.xlsx")

def run_an_example(m, n, tile_k):
    x = torch.randn((m, n), device="cuda")
    y = softmax(x, tile_k)

if __name__ == "__main__":
    run_benchmark()
    # run_an_example(4096, 32 * 1024, 64)
    # run_an_example(4096, 32 * 1024, 2)
    # run_an_example(4096, 32 * 1024, 1024)
