#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000

int main() {
  float *arr;
  if (!(arr = malloc(sizeof(float) * N)))
    return 1;

// Initialize the array with some values
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    arr[i] = (float)(i % 100); // simple pattern
  }

  // ---- Compute mean ----
  double sum = 0.0;
  /* reduction over sum prevents collisions between threads when writing to
   * "sum" var */
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < N; i++) {
    sum += arr[i];
  }

  double mean = sum / N;

  // ---- Compute variance ----
  double var_sum = 0.0;
#pragma omp parallel for reduction(+ : var_sum)
  for (int i = 0; i < N; i++) {
    double diff = arr[i] - mean;
    var_sum += diff * diff;
  }

  double stddev = sqrt(var_sum / N);

  printf("Mean = %f\n", mean);
  printf("Standard deviation = %f\n", stddev);

  free(arr);
  return 0;
}
