#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
// Kernel function to add the elements of two arrays

__global__
void mat_mult(int n, float *A, float *B, float *C)
{ 

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int sum = 0;
  for (int k = 0; k < n; k++){
    sum += A[idy*n + k]*B[k*n + idx];
  }
  C[idy*n +idx] = sum;
}

int main(void)
{
  int N = 1<<6;
  int threads_per_block = 1<<5;
  int blocks = N/threads_per_block;
  float *A, *B, *C;
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, N*N*sizeof(float));
  cudaMallocManaged(&B, N*N*sizeof(float));
  cudaMallocManaged(&C, N*N*sizeof(float));
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    for ( int j = 0; i < N; i++){
      if(i == j){
        A[i*N +j] = i;
        B[i*N +j] = i;
      } else {
      	A[i*N +j] = 0;
	B[i*N +j] = 0;
      }
    }
  }

  // Run kernel on 1M elements on the GPU
  dim3 threadsPerBlock(threads_per_block,threads_per_block);
  dim3 numBlocks(blocks, blocks);
  mat_mult<<<numBlocks, threadsPerBlock>>>(N, A, B, C);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      std::cout << C[i*N +j] << ", " ;
    }
    std::cout << std::endl;
  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return 0;
}
