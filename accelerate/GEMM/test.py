import torch

import triton
import triton.language as tl

import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[768, 1024, 1600, 2048, 4096, 8192],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['dense', 'sparse'],  # Possible values for `line_arg`.
        line_names=['Dense', 'Sparse'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    N = 64
    n = 2048
    x = torch.rand(N * n, 4 * size, device='cuda', dtype=torch.half).t().contiguous().t()
    W_sparse = to_sparse_semi_structured(torch.rand(size, 4 * size, device='cuda', dtype=torch.half))
    W_dense = W_sparse.to_dense()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'dense':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.mm(x, W_dense.t()), quantiles=quantiles)
    if provider == 'sparse':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.mm(x, W_sparse.t()), quantiles=quantiles)
    flops = lambda ms: 2 * size * (N * n) * (4 * size) * 1e-12 / (ms * 1e-3)
    return flops(ms), flops(max_ms), flops(min_ms)


benchmark.run(print_data=True, show_plots=False)
