import math
import triton
from triton import language as tl
import torch

@triton.jit
def logsumexp_kernel(
    out_ptr,
    in_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    num_programs_n = tl.num_programs(0)
    pid_m = tl.program_id(1)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offset = pid_m * N + n_offsets
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(out_ptr.dtype.element_ty)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    logz = m + tl.log(z)

    output_ptrs = out_ptr + pid_m * num_programs_n + pid_n
    tl.store(output_ptrs, logz)

@triton.jit
def combine_logsumexp_kernel(out_ptr, inp_ptr, M, N, TILE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N
    logzs = tl.load(inp_ptr + pid_m * N + n_offsets, other=-float("inf"), mask=mask).to(out_ptr.dtype.element_ty)
    m = tl.max(logzs, 0)
    e = tl.exp(logzs - m)
    z = tl.sum(e, 0)
    logz = m + tl.log(z)
    tl.store(out_ptr + pid_m, logz)

@triton.jit
def softmax_kernel(out_ptr, in_ptr, logz_ptr, M, N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    offset = pid_m * N + n_offsets
    mask = n_offsets < N
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(out_ptr.dtype.element_ty)
    logz = tl.load(logz_ptr + pid_m).to(out_ptr.dtype.element_ty)
    out = tl.exp(inp - logz)
    tl.store(out_ptr + offset, out, mask=mask)



def softmax(x):
    M, N = x.shape

    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count

    TILE_N = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, TILE_N)
    logz = torch.empty((M, num_tiles_n), dtype=x.dtype, device=x.device)
    grid = (num_tiles_n, M, 1)
    logsumexp_kernel[grid](logz, x, M, N, TILE_N)

    combined_logz = torch.empty((M, ), dtype=x.dtype, device=x.device)
    TILE_N = triton.next_power_of_2(num_tiles_n)
    grid = (M, 1, 1)
    combine_logsumexp_kernel[grid](combined_logz, logz, M, num_tiles_n, TILE_N)

    out = torch.empty_like(x)
    TILE_N = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, TILE_N)
    grid = (num_tiles_n, M, 1)
    softmax_kernel[grid](out, x, combined_logz, M, N, TILE_N)
    return out

import pytest

@pytest.mark.parametrize("n", [128 * 1024, 1024 * 1024, 8192 * 1024])
@pytest.mark.parametrize("m", [10, ])
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
    for m in [1, 8, 16, 32]:
        for n in [16 * 1024, 32* 1024, 64* 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024, 2048 * 1024, 4096 * 1024, 8192 * 1024]:
            if m * n * 4 > 1024 * 1024 * 1024 * 4:
                continue
            t1, t2 = benchmark_softmax(m, n)
            record = (m, n, t1, t2)
            records.append(record)
    df = pd.DataFrame.from_records(records, columns=["pre_size", "reduce_size", "split", "torch"])
    print(df)
    df.to_excel("split.xlsx")

if __name__ == "__main__":
    run_benchmark()