#include <stdint.h> // Required for uint32_t
#include <stdio.h>
#include <stdlib.h> // For rand()
#include <sys/sysinfo.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

struct timespec begin, end;
double elapsed;
#define MAT_N 100
#define MAT_M 1000
float *mat_A;//[MAT_M*MAT_N];
float *mat_B;//[MAT_N*MAT_M];
float *mat_C;
float *mat_C_result_copy;
float *mat_A_gpu;
float *mat_B_gpu;
float *mat_C_gpu;//[MAT_M*MAT_M]; // results in MxM matrix

/*
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
*/

__host__ __device__ inline void matMul_base(int i, float* A, float* B, float* C){
  float current;
  for (int j = 0; j < MAT_M; j++) {
    current = 0;
    for (int k = 0; k < MAT_N; k++) {
      current += A[i*MAT_M + k] * B[k*MAT_N + j];
    }
    C[i*MAT_M + j] = current;
  }
}

void matMul_CPU(){
  for (int i = 0; i < MAT_M; i++) {
    matMul_base(i, mat_A, mat_B, mat_C);
  }
}

__global__ void matMul_GPU(float* A, float* B, float* C){
  int global_thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  matMul_base(global_thread_id, A,B,C);
}

int main(int argc, char *argv[]) {
  mat_A = (float*)malloc(MAT_M*MAT_N*sizeof(float));
  mat_B = (float*)malloc(MAT_M*MAT_N*sizeof(float));
  mat_C = (float*)malloc(MAT_M*MAT_M*sizeof(float));
  mat_C_result_copy = (float*)malloc(MAT_M*MAT_M*sizeof(float));
  cudaMallocManaged(&mat_A_gpu, MAT_M*MAT_N*sizeof(float));
  cudaMallocManaged(&mat_B_gpu, MAT_N*MAT_M*sizeof(float));
  cudaMallocManaged(&mat_C_gpu, MAT_M*MAT_M*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC, &begin);

  printf("This system has %d processors configured and "
         "%d processors available.\n",
         get_nprocs_conf(), get_nprocs());

  printf("initializing\n");

  for (int i = 0; i < MAT_M; i++) {
    for (int j = 0; j < MAT_N; j++) {
      mat_A[i*MAT_M + j] = i+j;
      mat_B[j*MAT_N + i] = i+j;
    }
  }


  cudaMemcpy(mat_A_gpu,mat_A, MAT_M*MAT_N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mat_B_gpu,mat_B, MAT_M*MAT_N*sizeof(float), cudaMemcpyHostToDevice);

  printf("done initializing\n");

  matMul_CPU();
  clock_gettime(CLOCK_MONOTONIC, &end);

  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("took %f s\n", elapsed);


  ////
  ////
  ////
  ////
  ////

  clock_gettime(CLOCK_MONOTONIC, &begin);
  int block_dim = 32;
  int num_blocks = (MAT_M + block_dim - 1)/block_dim;
  printf("num blocks is: %d\n", num_blocks);
  matMul_GPU<<<num_blocks,block_dim>>>(mat_A_gpu, mat_B_gpu, mat_C_gpu);

  cudaError_t err = cudaSuccess;
  if (cudaGetLastError() != cudaSuccess){
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaMemcpy(mat_C_result_copy,mat_C_gpu, MAT_M*MAT_M*sizeof(float), cudaMemcpyDeviceToHost);

  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("took %f s\n", elapsed);

  for (int i = 0; i < MAT_M * MAT_M; i++){
     if (fabs(mat_C_result_copy[i] - mat_C[i]) > 1e-5){
       printf("no honey\n");
     }
  }
  printf("yes honey\n");

  return 0;
}
