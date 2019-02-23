#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

// Kernel function to perform C = A*B
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
  //N is the size of the matrix
  int N = 1<<8;
  
  //block_size is the width and height of cuda block
  //total threads per block = block_size*block_size
  int block_size = 1<<5;

  //grid_size is the width and height of a cuda grid
  //total blocks per grid = grid_size*grid_size
  int grid_size = N/block_size;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  float *A, *B, *C;
  cudaMallocManaged(&A, N*N*sizeof(float));
  cudaMallocManaged(&B, N*N*sizeof(float));
  cudaMallocManaged(&C, N*N*sizeof(float));

  // initialize A and B arrays on the host
  for (int i = 0; i < N; i++) {
    for ( int j = 0; j < N; j++){
      if(i == j){
        A[i*N +j] = i;
        B[i*N +j] = i;
      } else {
      	A[i*N +j] = 0;
	B[i*N +j] = 0;
      }
    }
  }

  // Run kernel with 2-D grid and 2-D blocks.
  dim3 block_dim(block_size, block_size);
  dim3 grid_dim(grid_size, grid_size);
  mat_mult<<<grid_dim, block_dim>>>(N, A, B, C);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (diagonal should be squares)
  int failure = 0;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if(i == j && C[i*N + j] != (float)i*i){
        failure = 1;
      } else if (i !=j && C[i*N + j] != 0.0){
        failure = 1;
      }
    }
  }

  //Helpful for printing out a matrix/debugging :^)
  /*for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      printf("%f,",A[i*N +j]);
    }
    printf("\n");
  }*/

  if(failure){
    printf("There was a failure, big sad!\n");
  } else {
    printf("Tests Pass!\n");
  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return 0;
}
