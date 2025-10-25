#include <stdio.h>
#include <stdlib.h> // for rand(), RAND_MAX

__host__ __device__ inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }
__host__ inline float getRandOneish() {
  // random vals from -1 to 1
  float r1 = static_cast<float>(rand()) / RAND_MAX; // in [0, 1]
  return 2.0f * r1 - 1.0f;                          // scale to [-1, 1]
}
__host__ void checkCuda(cudaError_t result, const char *msg) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

__host__ void checkFloats(size_t num_floats, float *A, float *B) {
  int errors = 0;
  for (int i = 0; i < num_floats; i++) {
    float diff = fabs(A[i] - B[i]);
    if (diff > 1e-2f) {
      if (errors < 10) // only print first few
        printf("Mismatch at (%d,%d): GPU=%f, CPU=%f, diff=%f\n", i, i, A[i],
               B[i], diff);
      errors++;
    }
  }
  printf("found %d total errors out of %d\n", errors, num_floats);
}
#include <stdio.h>

__host__ void print_array(const float *arr, int n,
                          const char *label = "array") {
  printf("%s = [", label);
  const int zz = 0;
  n = zz + n;
  for (int i = zz; i < n; i++) {
    printf("%g", arr[i]);
    if (i < n - 1)
      printf(", ");
  }
  printf("]\n");
}
__host__ void printArray1D(float *arr, float *arb, int n) {
  for (int i = 0; i < n + 3; i++) {
    if (i >= n)
      printf("(just beyond array):");
    printf("%d: %6.2f | %6.2f\n", i, arr[i], arb[i]);
  }
  printf("\n");
}

__host__ void matrixmult(float *fa, float *fb, float *fc, int Hight,
                         int Width) {
  int row, col, k;
  float Pvalue = 0;
  for (row = 0; row < Hight; row++) {
    for (col = 0; col < Width; col++) {
      Pvalue = 0;
      for (k = 0; k < Width; k++) {
        Pvalue += fa[row * Width + k] * fb[k * Width + col];
      }
      fc[row * Width + col] = Pvalue;
    }
  }
}
