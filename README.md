# gemm-cuda

GEMM (`c = A * B`) in CUDA. Benchmarks against cuBLAS with five kernals with two different optimization paths across FP32 with naive, tiled, register tiled, and float4 vectorized. Then WMMA **tensor-core** kernel in FP16. Every kernel is tested against cuBLAS result before it's timing is reported, that way a if there its a misleadingly fast number it will show as `fail`.

## Two ceilings

There isn't one "100%". There are two, because there are two sets of math units:

- **FP32 CUDA-core ceiling** = cuBLAS `Sgemm`. The four FP32 kernels chase this.
  Rough Turing-class shares: naive a few %, tiled ~10%, register-tiled ~50%,
  vectorized ~65–70%. Warptiling + double buffering would push toward ~85%.
- **FP16 tensor-core ceiling** = cuBLAS `GemmEx` on the tensor cores. A *much*
  higher GFLOP/s number, at FP16 precision. The WMMA kernel chases this one.
  It's a different kernel on different hardware, not a tweak to the FP32 path.

The `%cuBLAS` column is always relative to the cuBLAS baseline *of the same
tier*, so the FP32 kernels are measured against `Sgemm` and `wmma` against
`GemmEx`.

## Build & run

```bash
make ARCH=sm_75      # RTX 2060 (Turing)
make ARCH=sm_120     # RTX 5060 Ti (Blackwell)  -- needs CUDA 12.8+
# or just `make` to auto-detect the local GPU (-arch=native)

./gemm                  # default 1024 x 1024 x 1024
./gemm 2048             # square, M=N=K=2048
./gemm 1024 2048 512    # rectangular M N K
./gemm --sweep > results.csv   # 256..4096, CSV for plotting
```

Needs the CUDA toolkit (`nvcc`) and cuBLAS. The tensor-core kernel needs
**sm_70+** (both your cards qualify); it's skipped automatically on older GPUs.

## Plots

```bash
./gemm --sweep > results.csv

pip install matplotlib

python plot.py results.csv      # -> gflops_vs_size.png, pct_cublas.png
```

`plot.py` needs `matplotlib`. `gflops_vs_size.png` puts all kernels and both
cuBLAS baselines on one throughput axis (the tensor-core tier visibly towers
over the CUDA-core tier). `pct_cublas.png` is the FP32 kernels as a fraction of
FP32 cuBLAS.

## What it measures

- **Timing**: CUDA events, 5 warmup launches, mean over 20 timed launches.
- **Throughput**: `GFLOP/s = 2·M·N·K / time`.
- **Verification**: max relative error vs. the same-tier cuBLAS result; `PASS`
  if `< 1e-2` (FP32) / `< 2e-2` (FP16 inputs, looser by design).

Sample single-run output:

```
device: NVIDIA GeForce RTX 5060 Ti (sm_120)
problem: C[1024 * 1024] = A[1024 * 1024] * B[1024 * 1024] warmup=5 iters = 20

kernel                   time(ms)      GFLOP/S     %cuBlas   verify
naive                       1.425       1506.8       10.8%     pass
tiled                       1.063       2019.9       14.5%     pass
register-tiled              0.359       5989.5       42.9%     pass
vectorized                  0.206      10445.2       74.8%     pass
cuBLAS-fp16 (TC)            0.053      40456.3      100.0%      ref
wmma (TC)                   0.202      10651.9       26.3%     Pass
```

## Kernels

1. **naive** — one thread per output, `O(K)` global loads per element, no reuse.
2. **tiled** — `16×16` shared-memory tiles; coalesced loads, less global traffic.
3. **register-tiled** — `64×64` block, `BK=8`, each thread a `4×4` register
   micro-tile. Raises arithmetic intensity; closes most of the gap.
4. **vectorized** — `128×128` block, `8×8` micro-tile, `float4` global loads, A
   staged transposed in shared memory for contiguous inner-loop reads. Shape
   contract: dims multiple of the tile sizes (harness skips it otherwise).
5. **wmma (tensor core)** — FP16 in, FP32 accumulate, `16×16×16` MMA via the
   warp-level WMMA API. Each warp owns one tile and walks K loading fragments
   from global memory. This is the *naive* tensor-core kernel — correct and a
   large jump over FP32, but still global-memory-bound; the next step is staging
   tiles in shared memory exactly like kernels 2–4. Shape contract: dims
   multiple of 16, sm_70+.

## The row-major / column-major detail

cuBLAS is column-major; these kernels are row-major. To compute row-major
`C = A·B`, the cuBLAS call swaps the operands and the `m`/`n` extents, computing
`Cᵀ = Bᵀ·Aᵀ` in column-major terms. The buffer read back row-major is exactly
`C` — no transpose, no extra copy. Same trick for both `Sgemm` and `GemmEx`:

```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
```

## Profiling

`-lineinfo` is enabled, so Nsight Compute maps counters to source:

```bash
ncu --set full ./gemm 2048
```

Next steps: double-buffered / `cp.async` pipelines (Ampere+) to hide load
latency, warptiling for the FP32 path, and shared-memory staging + `mma.sync`
for the tensor-core path.
