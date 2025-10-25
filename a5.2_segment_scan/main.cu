#include "../helpful.cuh"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_NUMS 400
#define COMPARE_A
#define COMPARE_B

__global__ void blelloch_scan(float *data, int n) {
  __shared__ float
      temp[NUM_NUMS]; // in practice, allocated as extern __shared__
  int tid = threadIdx.x;

  // Load input into shared memory
  temp[2 * tid] = data[2 * tid];
  temp[2 * tid + 1] = data[2 * tid + 1];
  __syncthreads();

  // --- Upsweep (reduce) ---
  for (int stride = 1; stride < n; stride *= 2) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n)
      temp[idx] += temp[idx - stride];
    __syncthreads();
  }

  // Clear the last element (root)
  if (tid == 0)
    temp[n - 1] = 0;
  __syncthreads();

  // --- Downsweep ---
  for (int stride = n / 2; stride >= 1; stride /= 2) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) {
      float t = temp[idx - stride];
      temp[idx - stride] = temp[idx];
      temp[idx] += t;
    }
    __syncthreads();
  }

  // Write result back to global memory
  data[2 * tid] = temp[2 * tid];
  data[2 * tid + 1] = temp[2 * tid + 1];
}

int main() {
  // assumptions:

  // malloc host memory

  float *h_arr;
  float *h_res;
  float *h_result;
  size_t size_arr = sizeof(float) * NUM_NUMS;
  h_arr = (float *)malloc(size_arr);
  h_res = (float *)malloc(size_arr);
  h_result = (float *)malloc(size_arr);
  // Initialize image with random integer values
  srand(time(NULL));
  for (int i = 0; i < NUM_NUMS; i++)
    h_arr[i] = ((float)(rand() % 128)) - 64.0; // convert to float automatically

  // init timing stuff
  cudaEvent_t startEvent, stopEvent;
  float ms;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // setup device memory
  float *d_arr;
  checkCuda(cudaMalloc((void **)&d_arr, size_arr), "Alloc device array");
  checkCuda(cudaMemcpy(d_arr, h_arr, size_arr, cudaMemcpyHostToDevice),
            "Memcpy arr");

#ifdef COMPARE_A
  printf("starting timer ...\n");
  cudaEventRecord(startEvent);
  /*
   checkCuda(cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice), "Memcpy F");
   checkCuda(cudaMemcpy(d_H, h_H, size_H, cudaMemcpyHostToDevice), "Memcpy H");
   */

  dim3 dimBlock(256);
  dim3 dimGrid(1);
  printf("dimblock is %d x %d\n", dimBlock.x, dimBlock.y);
  printf("dimgrid  is %d\n", dimGrid.x);

  printf("starting timer ...\n");
  // puts result in the first index of input
  blelloch_scan<<<dimGrid, dimBlock>>>(d_arr, NUM_NUMS);

  printf("starting timer ...\n");
  cudaDeviceSynchronize();
  printf("starting timer ...\n");

  checkCuda(cudaMemcpy(h_result, d_arr, size_arr, cudaMemcpyDeviceToHost),
            "Memcpy  back to the host");

  cudaDeviceSynchronize();
  printf("starting timer ...\n");

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, parallel): %f\n", ms);

  // calculate sequential
  cudaEventRecord(startEvent);
  h_res[0] = 0.0;

  for (int i = 1; i < NUM_NUMS; i++) {
    h_res[i] = h_arr[i - 1] + h_res[i - 1];
  }
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, sequential): %f\n", ms);

  // compare to device
#ifdef DBG
  print_array(h_arr, NUM_NUMS, "statrt");
  print_array(h_res, NUM_NUMS, "ii");
  print_array(h_result, NUM_NUMS);
#endif

  bool f = false;
  for (int i = 0; i < NUM_NUMS; i++) {
    if (h_res[i] != h_result[i]) {
      f = true;
      break;
    }
  }

  if (f) {
    printf("fire and brimstone\n", ms);
  } else {
    printf("yahoo!\n", ms);
  }
#endif

  // Cleanup
  // free(h_G_comp);
}
