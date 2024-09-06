import triton
from triton import language as tl
import torch

@triton.jit
def softmax_kernel(
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
    n_offsets = tl.arange(0, TILE_N)
    offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
    mask = (n_offsets[:, None] < N) & (k_offsets < K)
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.type.element_ty)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m[None, :])
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)


def softmax(x, TILE_K):
    inp = x.contiguous()
    M = 1
    N, K = inp.shape
    out = torch.empty_like(x)
    TILE_N = triton.next_power_of_2(N)
    # TILE_K = 1
    grid = (triton.cdiv(K, TILE_K), M, 1)
    softmax_kernel[grid](
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

@pytest.mark.parametrize("n", [10, 128])
@pytest.mark.parametrize("m", [512, 1024, 32 * 1024])
@pytest.mark.parametrize("TILE_K", [1, 2, 4])
def test_softmax(m, n, TILE_K):
    x = torch.randn((m, n), device="cuda")
    hyp = softmax(x, TILE_K)
    ref = torch.softmax(x, dim=0)
    torch.testing.assert_close(hyp, ref)

def benchmark_softmax(m, n):
    x = torch.randn((m, n), device="cuda")
    t1 = triton.testing.do_bench(lambda: softmax(x, 1), return_mode="median")
    t2 = triton.testing.do_bench(lambda: softmax(x, 2), return_mode="median")
    t3 = triton.testing.do_bench(lambda: softmax(x, 4), return_mode="median")
    t4 = triton.testing.do_bench(lambda: torch.softmax(x, dim=0), return_mode="median")
    def throughput(t):
        return x.numel() * x.element_size() * 2 * 1e-9 / (t * 1e-3)
    return throughput(t1), throughput(t2), throughput(t3), throughput(t4)

import pandas as pd
def run_benchmark():
    records = []
    for n in [10, 128, 1024, 4096]:
        for m in [512, 1024, 2048, 4096, 8192, 16 * 1024, 32* 1024, 64 * 1024, 128 * 1024]:
            if m * n * 4 > 1024 * 1024 * 1024 * 4:
                continue
            t1, t2, t3, t4 = benchmark_softmax(m, n)
            record = (m, n, t1, t2, t3, t4)
            records.append(record)
    df = pd.DataFrame.from_records(records, columns=["reduce_size", "post_size", "naive_outer_k1", "naive_outer_k2", "naive_outer_k4", "torch"])
    print(df)
    df.to_excel("naive_outer.xlsx")

def run_an_example(m, n, tile_k):
    x = torch.randn((m, n), device="cuda")
    y = softmax(x, tile_k)

if __name__ == "__main__":
    run_benchmark()
    # run_an_example(4096, 4096, )