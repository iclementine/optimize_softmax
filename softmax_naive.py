import triton
from triton import language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    offset = pid_m * N + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < N
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)


def softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = triton.next_power_of_2(N)
    grid = (M, 1, 1)
    softmax_kernel[grid](out, x, M, N, TILE_N)
    return out

import pytest

@pytest.mark.parametrize("n", [512, 1024, 32 * 1024, 128 * 1024])
@pytest.mark.parametrize("m", [10, 128, 1024])
def test_softmax(m, n):
    x = torch.randn((m, n), device="cuda")
    hyp = softmax(x)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(hyp, ref)

def benchmark_softmax(m, n):
    x = torch.randn((m, n), device="cuda")
    t1 = triton.testing.do_bench(lambda: softmax(x), return_mode="median")
    t2 = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), return_mode="median")
    def throughput(t):
        return x.numel() * x.element_size() * 2 * 1e-9 / (t * 1e-3)
    return throughput(t1), throughput(t2)

import pandas as pd
def run_benchmark():
    records = []
    for m in [10, 128, 1024, 4096]:
        for n in [512, 1024, 2048, 4096, 8192, 16 * 1024, 32* 1024, 64* 1024, 128 * 1024]:
            t1, t2 = benchmark_softmax(m, n)
            record = (m, n, t1, t2)
            records.append(record)
    df = pd.DataFrame.from_records(records, columns=["pre_size", "reduce_size", "naive", "torch"])
    print(df)
    df.to_excel("naive.xlsx")

def run_an_example(m, n):
    x = torch.randn((m, n), device="cuda")
    y = softmax(x)

if __name__ == "__main__":
    run_an_example(4096, 4 * 1024)




