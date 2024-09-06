import triton
from triton import language as tl
import torch

@triton.jit
def softmax_kernel_online_v1(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = tl.full((), value=-float("inf"), dtype=output_ptr.dtype.element_ty)
    z = tl.full((), value=0, dtype=output_ptr.dtype.element_ty)
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, tl.max(inp, 0))
        new_z = tl.exp(m - new_m) * z + tl.sum(tl.exp(inp - new_m), 0)
        m = new_m
        z = new_z

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)


def softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    # TODO tune
    TILE_N = min(4096, triton.next_power_of_2(N))
    grid = (M, 1, 1)
    softmax_kernel_online_v1[grid](out, x, M, N, TILE_N)
    return out

import pytest

@pytest.mark.parametrize("n", [512, 1024, 32 * 1024, 128 * 1024])
@pytest.mark.parametrize("m", [10, 128])
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
        for n in [512, 1024, 2048, 4096, 8192, 16 * 1024, 32* 1024, 64* 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]:
            if m * n * 4 > 1024 * 1024 * 1024 * 4:
                continue
            t1, t2 = benchmark_softmax(m, n)
            record = (m, n, t1, t2)
            records.append(record)
    df = pd.DataFrame.from_records(records, columns=["pre_size", "reduce_size", "online_v1", "torch"])
    print(df)
    df.to_excel("online_v1.xlsx")

if __name__ == "__main__":
    run_benchmark()