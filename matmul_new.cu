#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

struct timespec begin, end;
double elapsed;

#define MAT_N 4
#define MAT_M 4

float *mat_A;
float *mat_B;
float *mat_C_cpu;
float *mat_C_gpu;

// GPU Kernel for Matrix Multiplication
__global__ void matMul_GPU(float* mat_A, float* mat_B, float* mat_C_gpu) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MAT_M && col < MAT_M) {
        float current = 0;
        for (int k = 0; k < MAT_N; k++) {
            current += mat_A[row * MAT_N + k] * mat_B[k * MAT_M + col];
        }
        mat_C_gpu[row * MAT_M + col] = current;
    }
}

// CPU Matrix Multiplication
void matMul_CPU() {
    float current;
    for (int i = 0; i < MAT_M; i++) {
        for (int j = 0; j < MAT_M; j++) {
            current = 0;
            for (int k = 0; k < MAT_N; k++) {
                current += mat_A[i * MAT_N + k] * mat_B[k * MAT_M + j];
            }
            mat_C_cpu[i * MAT_M + j] = current;
        }
    }
}

int main(int argc, char *argv[]) {
    // Allocate memory on the device
    cudaMallocManaged(&mat_A, MAT_M * MAT_N * sizeof(float));
    cudaMallocManaged(&mat_B, MAT_N * MAT_M * sizeof(float));
    cudaMallocManaged(&mat_C_cpu, MAT_M * MAT_M * sizeof(float));
    cudaMallocManaged(&mat_C_gpu, MAT_M * MAT_M * sizeof(float));

    clock_gettime(CLOCK_MONOTONIC, &begin);

    printf("This system has %d processors configured and %d processors available.\n",
           get_nprocs_conf(), get_nprocs());

    printf("Initializing matrices...\n");

    // Initialize matrices (you can choose to initialize with specific values)
    for (int i = 0; i < MAT_M; i++) {
        for (int j = 0; j < MAT_N; j++) {
            mat_A[i * MAT_N + j] = 1.0f;  // Just using 1 for simplicity
            mat_B[j * MAT_M + i] = 1.0f;  // Transpose to make multiplication possible
        }
    }

    printf("Matrix initialization done.\n");

    // Perform matrix multiplication on CPU
    matMul_CPU();
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("CPU multiplication took %f seconds.\n", elapsed);

    printf("Starting GPU computation...\n");

    // Perform matrix multiplication on GPU
    clock_gettime(CLOCK_MONOTONIC, &begin);
    
    dim3 threadsPerBlock(16, 16); // A 2D block size
    dim3 numBlocks((MAT_M + 15) / 16, (MAT_M + 15) / 16); // 2D grid size

    matMul_GPU<<<numBlocks, threadsPerBlock>>>(mat_A, mat_B, mat_C_gpu);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("GPU multiplication took %f seconds.\n", elapsed);

    // Compare GPU and CPU results
    for (int i = 0; i < MAT_M * MAT_M; i++) {
        if (fabs(mat_C_gpu[i] - mat_C_cpu[i]) > 1e-5) {
            printf("Mismatch found at index %d\n", i);
            break;
        }
    }
    printf("Comparison complete.\n");

    // Free allocated memory
    cudaFree(mat_A);
    cudaFree(mat_B);
    cudaFree(mat_C_cpu);
    cudaFree(mat_C_gpu);

    return 0;
}

