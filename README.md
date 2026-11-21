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

# without make, the same build is:
nvcc -O3 -arch=native -lineinfo -o gemm main.cu -lcublas

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
cuBLAS baselines on one throughput axis (cuBLAS FP16 visibly towers over
everything else; the WMMA kernel doesn't yet). `pct_cublas.png` is the FP32 kernels as a fraction of
FP32 cuBLAS.

## What it measures

- **Timing**: CUDA events, 5 warmup launches, mean over 20 timed launches.
- **Throughput**: `GFLOP/s = 2·M·N·K / time`.
- **Verification**: max relative error vs. the same-tier cuBLAS result; `PASS`
  if `< 1e-2` (FP32) / `< 2e-2` (FP16 inputs, looser by design).

Sample single-run output at 4096³ (`./gemm 4096`):

```
device: NVIDIA GeForce RTX 5060 Ti (sm_120)
problem: C[4096 * 4096] = A[4096 * 4096] * B[4096 * 4096] warmup=5 iters = 20

kernel                   time(ms)      GFLOP/S     %cuBlas   verify
naive                     113.269       1213.4        8.5%     pass
tiled                      81.518       1686.0       11.8%     pass
register-tiled             30.768       4466.9       31.4%     pass
vectorized                 13.454      10215.2       71.7%     pass
cuBLAS-fp16 (TC)            3.443      39921.1      100.0%      ref
wmma (TC)                  13.767       9983.4       25.0%     Pass
```

Quote the 4096³ numbers, not 1024³. At 1024³ the 20 timed launches take about
a millisecond in total, so the result swings a lot between runs (cuBLAS FP16
read anywhere from 30 to 40 TFLOP/s in back-to-back runs). `runs.txt` has three
repeat runs each at 1024³, 2048³ and 4096³ on the RTX 5060 Ti, showing only the
`vectorized`, cuBLAS FP16 and `wmma` rows. At 4096³ they agree within ~1%.

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
   from global memory. This is the *naive* tensor-core kernel. It's correct
   and about 8× the naive FP32 kernel, but it only **ties** the optimized FP32
   `vectorized` kernel (both ~10.1 TFLOP/s at 4096³) and reaches ~25% of
   cuBLAS FP16. It has none of the data reuse that kernels 2–4 build up:
   - no shared-memory staging, so every `load_matrix_sync` goes to global
     memory and reuse depends on L1/L2 hits;
   - one `16×16` accumulator per warp, so each K step loads two fragments
     (1 KB) for a single 8 KFLOP MMA, about 8 FLOP/byte;
   - no `cp.async` or double buffering, so loads and MMAs don't overlap.

   It's limited by how fast fragments arrive, not by tensor-core math. The
   next steps are staging tiles in shared memory exactly like kernels 2–4,
   then giving each warp a grid of fragments so every A fragment it loads is
   reused across several B fragments. Shape contract: dims multiple of 16,
   sm_70+.

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
