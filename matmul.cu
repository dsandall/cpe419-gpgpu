#include <stdint.h> // Required for uint32_t
#include <stdio.h>
#include <stdlib.h> // For rand()
#include <sys/sysinfo.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

struct timespec begin, end;
double elapsed;
#define MAT_N 800
#define MAT_M 800
float *mat_A;//[MAT_M*MAT_N];
float *mat_B;//[MAT_N*MAT_M];
float *mat_C_cpu;//[MAT_M*MAT_M]; // results in MxM matrix
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
    matMul_base(i, mat_A, mat_B, mat_C_cpu);
  }
}

__global__ void matMul_GPU(float* mat_A, float* mat_B, float* mat_C_gpu){
  int global_thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  matMul_base(global_thread_id, mat_A, mat_B, mat_C_gpu);
}

int main(int argc, char *argv[]) {
  cudaMallocManaged(&mat_A, MAT_M*MAT_N*sizeof(float));
  cudaMallocManaged(&mat_B, MAT_N*MAT_M*sizeof(float));
  cudaMallocManaged(&mat_C_cpu, MAT_M*MAT_M*sizeof(float));
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
  int num_blocks = (MAT_M + 32 - 1)/32;
  printf("num blocks is: %d\n", num_blocks);
  matMul_GPU<<<num_blocks,32>>>(mat_A, mat_B, mat_C_gpu);

  cudaError_t err = cudaSuccess;
  if (cudaGetLastError() != cudaSuccess){
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &end);

  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("took %f s\n", elapsed);

  for (int i = 0; i < MAT_M * MAT_M; i++){
     if (fabs(mat_C_gpu[i] - mat_C_cpu[i]) > 1e-5){
       printf("no honey\n");
     }
  }
  printf("yes honey\n");

  return 0;
}
