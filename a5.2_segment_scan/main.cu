#include "../helpful.cuh"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUM_NUMS
#define NUM_NUMS 1025
#endif
#define TEST

void blelloch_scan(float *d_data, int N);
__global__ void block_scan(float *data, float *block_sums, int n);
__global__ void add_offsets(float *data, float *block_sums, int n);

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

#ifdef TIME
  // init timing stuff
  cudaEvent_t startEvent, stopEvent;
  float ms;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
#endif

  // setup device memory
  float *d_arr;
  checkCuda(cudaMalloc((void **)&d_arr, size_arr), "Alloc device array");
  checkCuda(cudaMemcpy(d_arr, h_arr, size_arr, cudaMemcpyHostToDevice),
            "Memcpy arr");

  /*
   checkCuda(cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice), "Memcpy F");
   checkCuda(cudaMemcpy(d_H, h_H, size_H, cudaMemcpyHostToDevice), "Memcpy H");
   */

#ifdef TIME
  printf("starting timer ...\n");
  cudaEventRecord(startEvent);
#endif
  // puts result in the first index of input
  blelloch_scan(d_arr, NUM_NUMS);

  cudaDeviceSynchronize();

  checkCuda(cudaMemcpy(h_result, d_arr, size_arr, cudaMemcpyDeviceToHost),
            "Memcpy  back to the host");

#ifdef TEST

#ifdef TIME
  cudaDeviceSynchronize();
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, parallel): %f\n", ms);

  cudaEventRecord(startEvent);
#endif

  // calculate sequential
  h_res[0] = 0.0;
  for (int i = 1; i < NUM_NUMS; i++) {
    h_res[i] = h_arr[i - 1] + h_res[i - 1];
  }

#ifdef TIME
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, sequential): %f\n", ms);
#endif

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
      printf("fire and brimstone\n");
      break;
    }
  }
  if (!f) {
    printf("yahoo!\n");
  }
#endif
}

const int BLOCK_SIZE = 512; // must match kernel’s design
const int ELEMS_PER_BLOCK = BLOCK_SIZE * 2;

void blelloch_scan(float *d_data, int N) {
  int numBlocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

  // Allocate memory for per-block sums
  float *d_block_sums = nullptr;
  cudaMalloc(&d_block_sums, numBlocks * sizeof(float));

  printf("dimblock is %d \n", BLOCK_SIZE);
  printf("dimgrid  is %d\n", numBlocks);
  // --- Phase 1: local scans ---
  block_scan<<<numBlocks, BLOCK_SIZE, ELEMS_PER_BLOCK * sizeof(float)>>>(
      d_data, d_block_sums, N);

  // --- Phase 2: recursively scan block sums ---
  if (numBlocks > 1)
    blelloch_scan(d_block_sums, numBlocks); // recursion on smaller array

  // --- Phase 3: add scanned block offsets back ---
  add_offsets<<<numBlocks, BLOCK_SIZE>>>(d_data, d_block_sums, N);

  cudaFree(d_block_sums);
}

//////
//////
//////
//////
//////
//////
//////

__global__ void block_scan(float *data, float *block_sums, int n) {
  __shared__ float temp[BLOCK_SIZE * 2];
  int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  int tid = threadIdx.x;

  int start = blockIdx.x * ELEMS_PER_BLOCK;
  int end = min(start + ELEMS_PER_BLOCK, n);

  int idx0 = start + 2 * tid;
  int idx1 = start + 2 * tid + 1;

  temp[2 * tid] = (idx0 < n) ? data[idx0] : 0.0f;
  temp[2 * tid + 1] = (idx1 < n) ? data[idx1] : 0.0f;

  __syncthreads();
  // run your existing Blelloch scan on temp[]
  for (int stride = 1; stride < blockDim.x * 2; stride <<= 1) {
    __syncthreads();
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < blockDim.x * 2)
      temp[idx] += temp[idx - stride];
  }

  if (tid == 0)
    temp[blockDim.x * 2 - 1] = 0;

  for (int stride = blockDim.x; stride > 0; stride >>= 1) {
    __syncthreads();
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < blockDim.x * 2) {
      float t = temp[idx - stride];
      temp[idx - stride] = temp[idx];
      temp[idx] += t;
    }
  }
  __syncthreads();

  // write back
  if (2 * gid < n)
    data[2 * gid] = temp[2 * tid];
  if (2 * gid + 1 < n)
    data[2 * gid + 1] = temp[2 * tid + 1];

  // store block total
  if (block_sums && tid == 0) {
    int last = end - start - 1;
    int second_last = max(0, last - 1);
    block_sums[blockIdx.x] = temp[last] + temp[second_last];
  }
}
__global__ void add_offsets(float *data, float *block_sums, int n) {
  if (blockIdx.x == 0)
    return; // first block has no offset

  int start = blockIdx.x * ELEMS_PER_BLOCK;
  int end = min(start + ELEMS_PER_BLOCK, n);

  int idx0 = start + 2 * threadIdx.x;
  int idx1 = start + 2 * threadIdx.x + 1;

  float offset = block_sums[blockIdx.x - 1];

  if (idx0 < n)
    data[idx0] += offset;
  if (idx1 < n)
    data[idx1] += offset;
}
