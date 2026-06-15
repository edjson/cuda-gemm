#pragma once
#include <cuda_runtime.h>
#include <mma.h>

// Matricies are row major C[M * N] = A[M * K] * B[K * N]

// Kernel 1 naive
// One thread per output element, each thread streams a full row of A
// and a full column of B form global memory

__global__ void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < N && row < M){
        float sum = 0.0f;
        for(int i = 0; i < K; i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N +col] = sum;
    }
}

// Kernel 2 shared memory tiled
// stage a tile size * tile size of block a and b in shared memory per k step
// and resue it across the block. Cuts the global traffic by tilse size and 
// makes the loads coalesced

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K){
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    float sum = 0.0f;

    for(int i = 0; i < (K + TILE_SIZE -1) / TILE_SIZE; i++){
        if(row < M && i * TILE_SIZE + tx < K){
            a_tile[ty][tx] = A[row * K + i * TILE_SIZE + tx];
        }
        else{
            a_tile[ty][tx] = 0.0f;
        }

        if(i * TILE_SIZE + ty < K && col < N){
            b_tile[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
        }
        else{
            b_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        for(int j = 0; j < TILE_SIZE; j++){
            sum += a_tile[ty][j] * b_tile[j][tx];
        }
        __syncthreads();
    }
    if(col < N && row < M){
        C[row * N + col] = sum;
    }
}

// Kernel 3 2d register block tiling
// Each thread computes a TM * TN micro tile of c, keeping the partial sums in registers
// Per k step a BM * BK slab of A and a BK * BN slab of B are staged  in shared memory
// then the ntter product is done form registers.

#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

__global__ void matmul_register_tiled(float* C, const float* A, const float* B, int M, int N, int K){
    const int blockCol = blockIdx.x;
    const int blockRow = blockIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int threadsPerRow = BN / TN;
    const int threadCol = threadIdx.x % threadsPerRow;
    const int threadRow = threadIdx.x / threadsPerRow;
    const int numThreads = (BM / TM) * (BN / TN);

    float acc[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    const int numTiles = (K + BK - 1) / BK;
    for(int i = 0; i < numTiles; i++){
        const int koffset = i * BK;

        for(int j = threadIdx.x; j < BM * BK; j += numThreads){
            int r = j / BK;
            int c = j % BK;
            int gRow = blockRow * BM + r;
            int gCol = koffset + c;

            if(gRow < M &&  gCol < K){
                As[r][c] = A[gRow * K + gCol];
            }
            else{
                As[r][c] = 0.0f;
            }
        }

        for(int j = threadIdx.x; j < BK * BN; j += numThreads){
            int r = j / BN;
            int c = j % BN;
            int gCol = blockCol * BN + c;
            int gRow = koffset + r;
            
            if(gCol < N && gRow < K){
                Bs[r][c] = B[gRow * N + gCol];
            }
            else{
                Bs[r][c] = 0.0f;
            }
        }
        __syncthreads();

        for(int k = 0; k < BK; k++){
            for(int ti = 0; ti < TM; ti++){
                regA[ti] = As[threadRow * TM + ti][k];
            }

            for(int ti = 0; ti < TN; ti++){
                regB[ti] = Bs[k][threadCol * TN + ti];
            }

            for(int ti = 0; ti < TM; ti++){
                for(int tj = 0; tj < TN; tj++){
                    acc[ti][tj] += regA[ti] * regB[tj];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM; i++){
        for(int j = 0; j < TN; j++){
            int gCol = blockCol * BN + threadCol * TN + j; 
            int gRow = blockRow * BM + threadRow * TM + i;
            if(gCol < N && gRow < M){
                C[gRow * N + gCol] = acc[i][j];
            }
            
        }
    }

}

// Kernel 4 vectorized 128x128 block tile, 8x8 register micro-tile, float4 loads
// Two further wins over kernel 3:
//   (a) 128-bit (float4) global loads -> fewer, fatter memory transactions.
//   (b) A is staged TRANSPOSED in shared memory, so the inner loop reads a
//       column of the A-tile as a contiguous row (As[k][...]).
// Each thread computes an 8x8 = 64-element micro-tile in registers.
// Launch: blockDim = (VBM/VTM)*(VBN/VTN) = 256 threads (1D);
//         gridDim  = ( N/VBN, M/VBM ).

#define VBM 128
#define VBN 128
#define VBK 8
#define VTM 8
#define VTN 8

__global__ void matmul_vectorized(float* C, const float* A, const float* B, int M, int N, int K){
    const int blockCol = blockIdx.x;
    const int blockRow = blockIdx.y;

    __shared__ float As[VBK][VBM];
    __shared__ float Bs[VBK][VBN];

    const int threadCol = threadIdx.x % (VBN / VTN);
    const int threadRow = threadIdx.x / (VBN / VTN);
    // decompose the linear thread id into float4 load coordinates
    const int innerRowA = threadIdx.x / (VBK / 4);
    const int innerColA = threadIdx.x % (VBK / 4);
    const int innerRowB = threadIdx.x / (VBN / 4);
    const int innerColB = threadIdx.x % (VBN / 4);

    float acc[VTM][VTN] = {0.0f};
    float regM[VTM];
    float regN[VTN];

    for(int koffset = 0; koffset < K; koffset += VBK){
        float4 a4 = *reinterpret_cast<const float4*>(
            &A[(blockRow * VBM + innerRowA) * K + koffset + innerColA * 4]);
        As[innerColA * 4 + 0][innerRowA] = a4.x;
        As[innerColA * 4 + 1][innerRowA] = a4.y;
        As[innerColA * 4 + 2][innerRowA] = a4.z;
        As[innerColA * 4 + 3][innerRowA] = a4.w;

        *reinterpret_cast<float4*>(&Bs[innerRowB][innerColB * 4]) = 
            *reinterpret_cast<const float4*>(
                &B[(koffset + innerRowB) * N + blockCol * VBN + innerColB * 4]);
        
        __syncthreads();

        for(int k = 0; k < VBK; k++){
            for(int i = 0; i < VTM; i ++){
                regM[i] = As[k][threadRow * VTM + i];
            }

            for(int i = 0; i < VTN; i++){
                regN[i] = Bs[k][threadCol * VTN + i];
            }

            for(int i = 0; i < VTM; i++){
                for(int j = 0; j < VTN; j++){
                    acc[i][j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < VTM; i++){
        for(int j = 0; j < VTN; j++){
            const int gRow = blockRow * VBM + threadRow * VTM + i;
            const int gCol = blockCol * VBN + threadCol * VTN + j;
            if(gRow < M && gCol < N){
                C[gRow * N + gCol] = acc[i][j];
            }
        }
    }

}

// Kernel 5: WMMA tensor-core GEMM (FP16 in, FP32 accumulate)
// issues 16x16x16 matrix-multiply-accumulate instructions
// on the tensor cores via the warp-level WMMA API, the whole warp (32 lanes)
// Requires sm_70+ (Turing sm_75 and Blackwell sm_120 both qualify)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_wmma(float* C, const half* A, const half* B, int M, int N, int K){
    using namespace nvcuda;

    int warpM = blockIdx.y * blockDim.y + threadIdx.y;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    const int bCol = warpN * WMMA_N;
    const int aRow = warpM * WMMA_M;
    
    if(aRow < M && bCol < N){
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);
        for(int i = 0; i < K; i += WMMA_K){
                wmma::load_matrix_sync(a_frag, A + aRow * K + i, K);
                wmma::load_matrix_sync(b_frag, B + i * N + bCol, N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(C + aRow * N + bCol, acc_frag, N, wmma::mem_row_major);
    }
}
