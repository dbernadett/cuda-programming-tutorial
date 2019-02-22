#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
// Kernel function to add the elements of two arrays
__global__
void add_vec(float *A, float *B, float *C)
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  C[idx] = A[idx] + B[idx];
}

int main(void)
{
  int N = 1<<7;
  int threads_per_block = 1<<5;
  
  //Each thread should add one element
  int blocks = N/threads_per_block;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  float *A, *B, *C;
  cudaMallocManaged(&A, N*sizeof(float));
  cudaMallocManaged(&B, N*sizeof(float));
  cudaMallocManaged(&C, N*sizeof(float));
  
  // initialize A and B arrays on the host
  for (int i = 0; i < N; i++) {
    A[i] = 2;
    B[i] = 7;
  }

  // Run kernel on the GPU with 1-D blocks and threads.
  add_vec<<<blocks, threads_per_block>>>(A, B, C);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 9.0f) 
  int Failure = 0;
  for (int i = 0; i < N; i++){
    if(C[i] != 9.0){
      Failure = 1;
      break;
    }
  }

  //Print test results.
  if(Failure){
    printf("Result Incorrect, big sad!\n");
  } else {
    printf("Result Correct!\n");
  }
  
  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return 0;
}
