#include "../helpful.cuh"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_NUMS 1000000
#define COMPARE_A
#define COMPARE_B

__global__ void kernel(float *d_arr, int num_remaining, int step) {
  const int nums_per_thread = 2;
  // kernel<<<dimGrid, dimBlock>>>(d_arr, num_remaining, step);
  const int stream_offset = 0;
  int ti = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;

  if (ti == 0) {
    if (num_remaining % 2) {
      d_arr[0] = d_arr[0];
#ifdef DBG
      printf("%d <- %d \n", 0, 0);
#endif
      return;
    }
  }
  if (ti < (num_remaining + (nums_per_thread - 1)) / nums_per_thread) {
    d_arr[ti] = d_arr[ti] + d_arr[ti + num_remaining / 2];
#ifdef DBG
    printf("%d <- %d + %d\n", ti, ti, ti + num_remaining / 2);
#endif
    return;

  } else {
    // nothin

    return;
  }
}

int main() {

  // malloc host memory
  printf("NUM_NUMS = %d\n", NUM_NUMS);
#ifdef spread
  printf("spread mode\n");
#endif

  float *h_arr;
  float h_result;
  size_t size_arr = sizeof(float) * NUM_NUMS;
  h_arr = (float *)malloc(size_arr);
  // Initialize image with random integer values
  srand(time(NULL));
  for (int i = 0; i < NUM_NUMS; i++)
    h_arr[i] =
        ((float)(rand() % 256) - 128.0); // convert to float automatically
  // h_arr[i] = 1.0;

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

  int num_remaining = NUM_NUMS;
  int step = 1;
  int launches = 0;

  while (num_remaining > 1) {
    launches++;

    // setup device threads and blocks
    dim3 dimBlock(128);
    dim3 dimGrid(((128 * 2) + num_remaining - 1) / (128 * 2));

#ifdef DBG
    printf("kernel %d... (%d remain)\n", launches, num_remaining);
    printf("dimblock is %d x %d\n", dimBlock.x, dimBlock.y);
    printf("dimgrid  is %d\n", dimGrid.x);
#endif

    // puts result in the first index of input
    kernel<<<dimGrid, dimBlock>>>(d_arr, num_remaining, step);

    cudaDeviceSynchronize();

    if (num_remaining % 2 == 0) {
      num_remaining -= num_remaining / 2;
    } else {
      num_remaining -= (num_remaining - 1) / 2;
    }

    // step *= 2;
  }

  checkCuda(cudaMemcpy(&h_result, d_arr, sizeof(float), cudaMemcpyDeviceToHost),
            "Memcpy  back to the host");

  cudaDeviceSynchronize();

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, parallel): %f\n", ms);

  // compare to reality
  long accum;

  cudaEventRecord(startEvent);
  for (int i = 0; i < NUM_NUMS; i++) {
    accum += (long)h_arr[i];
  }
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, sequential): %f\n", ms);
  if (accum != (long)h_result) {
    printf("fire and brimstone\n", ms);
  } else {
    printf("yahoo!\n", ms);
  }
#endif

#ifdef COMPARE_B
#endif

  // Cleanup
  // free(h_G_comp);
}
