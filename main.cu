#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "kernels.cuh"

#define CUDA_CHECK(x)   cuda_check((x),   __FILE__, __LINE__)
#define CUBLAS_CHECK(x) cublas_check((x), __FILE__, __LINE__)

void cuda_check(cudaError_t e, const char* file, int line){
    if(e != cudaSuccess){
        fprintf(stderr, "cuda error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        exit(1);
    }
}

void cublas_check(cublasStatus_t s, const char* file, int line){
    if(s != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cuBLAS error %s:%d: %s\n", file, line, cublasGetStatusString(s));
        exit(1);
    }
}

static double gflops(int M, int N, int K, double ms){
    return (2.0 * M * N * K) / (ms * 1e6);  // 2 * M * N * K flops / (ms * 1e6) = GFLOP/s
}

template <typename F>
static double time_kernel(F launch, int warmup, int iters){
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for(int i = 0; i < warmup; i++){
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < iters; i++){
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (double)ms / iters;
}

static double max_rel_error(const float* test, const float* ref, int n){
    double maxe = 0.0;
    for(int i = 0; i < n; i++){
        double d = fabs((double)test[i] - (double)ref[i]);
        double e = d / (fabs((double)ref[i]) + 1e-5);
        if(e > maxe){
            maxe = e;
        }
    }
    return maxe;
}

__global__ void f32_to_f16(half* out, const float* in, size_t n){
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = __float2half(in[i]);
    }
}

// runs every kernel for one problem size. csv=false -> pretty table;
// csv=true -> "size,kernel,gflops,pct_cublas,verify" rows for plotting.
static void benchmark(int M, int N, int K, int warmup, int iters, bool csv){
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *host_A = (float*)malloc(bytes_A);
    float *host_B = (float*)malloc(bytes_B);
    float *host_ref = (float*)malloc(bytes_C);
    float *host_C = (float*)malloc(bytes_C);

    srand(42); //random number generator seed

    for(size_t i = 0; i < (size_t)M * K; i++){
        host_A[i] = (float)rand() / RAND_MAX;
    }

    for(size_t i = 0; i < (size_t)K* N; i++){
        host_B[i] = (float)rand() / RAND_MAX;
    }

    float *device_A;
    float *device_B;
    float *device_C;

    CUDA_CHECK(cudaMalloc(&device_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&device_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&device_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(device_A, host_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_B, host_B, bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBlas is column major, row major is C = A*B while in column major C^T = B^T * A^
    // swap opperands + m/n extents, the buffer reads back row major as C 
    auto launch_cublas = [&](){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, device_B, N, device_A, K, &beta, device_C, N);
    };

    CUDA_CHECK(cudaMemset(device_C, 0, bytes_C));
    launch_cublas();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_ref, device_C, bytes_C, cudaMemcpyDeviceToHost));

    double ms_cublas = time_kernel(launch_cublas, warmup, iters);

    if(csv == false){\
        printf("%-20s %12s %12s %11s %8s\n", "kernel", "time(ms)", "GFLOP/S", "%cuBlas", "verify");
    }
    else{
        printf("%d,cuBlAS,%.1f,100.0,ref\n", N, gflops(M, N, K, ms_cublas));
    }
    
    auto run = [&](const char* name, auto launch){
        CUDA_CHECK(cudaMemset(device_C, 0, bytes_C));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_C, device_C, bytes_C, cudaMemcpyDeviceToHost));
        double error = max_rel_error(host_C, host_ref, M * N);
        double ms = time_kernel(launch, warmup, iters);
        const char* v;
        if(error < 1e-2){
            v = "pass";
        }
        else{
            v = "fail";
        }

        if(csv == false){
            printf("%-20s %12.3f %12.1f %10.1f%% %8s\n", name, ms, gflops(M, N, K, ms), 100.0 * ms_cublas / ms, v);
        }
        else{
            printf("%d,%s,%.1f,%.1f,%s\n", N, name, gflops(M, N, K, ms), 100.0 * ms_cublas / ms, v);
        }
    };
    dim3 block2d(16, 16);
    dim3 grid2d((N + 15) / 16, (M + 15) / 16);
    run("naive", [&](){matmul_naive<<<grid2d, block2d>>>(device_C, device_A, device_B, M, N, K);});
    run("tiled", [&](){matmul_tiled<<<grid2d, block2d>>>(device_C, device_A, device_B, M, N, K);});

    dim3 gridRT((N + BN - 1) / BN, (M + BM - 1)/ BM);
    int blockRT = (BM / TM) * (BN / TN);
    run("register-tiled", [&](){matmul_register_tiled<<<gridRT, blockRT>>>(device_C, device_A, device_B, M, N, K);});

    // Vectorized kernel requires tile-aligned shapes (its float4 fast path has
    // no boundary handling). Skip it otherwise.

    if(M % VBM == 0 && N % VBN == 0 && K % VBK == 0){
        dim3 gridV(N / VBN, M / VBM);
        int blockV = (VBM / VTM) * (VBN / VTN);
        run("vectorized", [&](){matmul_vectorized<<<gridV, blockV>>>(device_C, device_A, device_B, M, N, K);});
    }
    else if(csv == false){
        printf("%-20s %12s %12s %11s %8s\n", "vectorized", "-", "-", "-", "skip");
    }

    // FP16 tensor-core section:
    // Requires sm_70+ and tile-aligned (mult. of 16) shapes. Inputs are cast to
    // FP16; both cuBLAS and the WMMA kernel accumulate in FP32. The baseline
    // here is cublasGemmEx on the tensor cores, so %cuBLAS in these rows is
    // relative to the *FP16* baseline, not the FP32 one above.

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    bool tc_ok = prop.major >= 7 && M % 16 == 0 && N % 16 == 0 && K % 16 == 0;

    if(tc_ok == true){
        half *device_A_half;
        half *device_B_half;
        CUDA_CHECK(cudaMalloc(&device_A_half, (size_t)M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&device_B_half, (size_t)K * N * sizeof(half)));
        int threads = 256;
        f32_to_f16<<<((size_t)M * K + threads - 1) / threads, threads>>>(device_A_half, device_A, (size_t)M * K);
        f32_to_f16<<<((size_t)K * N + threads - 1) / threads, threads>>>(device_B_half, device_B, (size_t)K * N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto launch_tc = [&](){cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                                            &alpha, device_B_half, CUDA_R_16F, N, device_A_half,
                                            CUDA_R_16F, K, &beta, device_C, CUDA_R_32F, N,
                                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        CUDA_CHECK(cudaMemset(device_C, 0, bytes_C));
        launch_tc();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_ref, device_C, bytes_C, cudaMemcpyDeviceToHost));
        double ms_tc = time_kernel(launch_tc, warmup, iters);
        if(csv == false){
            printf("%-20s %12.3f %12.1f %10.1f%% %8s\n", "cuBLAS-fp16 (TC)", ms_tc, gflops(M, N, K, ms_tc), 100.0, "ref");
        }
        else{
            printf("%d,cuBLAS-fp16,%.1f,100.0,ref\n", N, gflops(M, N, K, ms_tc));
        }
        dim3 blockW(128, 4);
        dim3 gridW((N + 63) / 64, (M + 63)/ 64);
        auto launchW = [&](){matmul_wmma<<<gridW, blockW>>>(device_C, device_A_half, device_B_half, M, N, K);};
        CUDA_CHECK(cudaMemset(device_C, 0, bytes_C));
        launchW();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_C, device_C, bytes_C, cudaMemcpyDeviceToHost));
        double errW = max_rel_error(host_C, host_ref, M * N);
        double msW = time_kernel(launchW, warmup, iters);
        const char* vW;
        if(errW < 2e-2){
            vW = "Pass";
        }
        else{
            vW = "Fail";
        }
        if(csv == false){
            printf("%-20s %12.3f %12.1f %10.1f%% %8s\n", "wmma (TC)", msW, gflops(M, N, K, msW), 100.0 * ms_tc / msW, vW);
        }
        else{
            printf("%d,wmma,%.1f,%.1f,%s\n", N, gflops(M, N, K, msW), 100.0 * ms_tc / msW, vW);
        }
        cudaFree(device_A_half);
        cudaFree(device_B_half);
    }
    else if (csv == false){
        printf("%-20s %12s %12s %11s %8s\n", "wmma (tc)", "-", "-", "-", "skip");
    }
    cublasDestroy(handle);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);
    free(host_ref);
}

int main(int argc, char** argv){
    const int warmup = 5;
    const int iters = 20;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    bool sweep = (argc == 2 && strcmp(argv[1], "--sweep") ==0 );

    if(sweep == true){
        printf("size,kernel,gflops,pct_cublas,verify\n");
        const int sizes[] = {256, 512, 1024, 4096};
        for(size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++){
            int s = sizes[i];
            benchmark(s, s, s, warmup, iters, true);
        }
        return 0;
    }
    int M = 1024;
    int N = 1024;
    int K = 1024;
    if(argc == 2){
        M = N = K = atoi(argv[1]);
    }
    else if(argc == 4){
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("problem: C[%d * %d] = A[%d * %d] * B[%d * %d] warmup=%d iters = %d\n\n", M, N, M, K, K, N, warmup, iters);
    benchmark(M, N, K, warmup, iters, false);
    return 0;

}