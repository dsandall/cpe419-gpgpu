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
    if (diff > 1e-3f) {
      if (errors < 10) // only print first few
        printf("Mismatch at (%d,%d): GPU=%f, CPU=%f, diff=%f\n", i, i, A[i],
               B[i], diff);
      errors++;
    }
  }
  printf("found %d total errors out of %d\n", errors, num_floats);
}

__host__ void printArray1D(float *arr, float *arb, int n) {
  for (int i = 0; i < n; i++)
    printf("%d: %6.2f | %6.2f\n", i, arr[i], arb[i]);
  printf("\n");
}
