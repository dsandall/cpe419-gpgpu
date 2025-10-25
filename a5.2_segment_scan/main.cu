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
  checkCuda(cudaMalloc((void **)&d_arr, size_arr * 2),
            "Alloc device array"); // TODO: quick and dirty fix *2
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
  print_array(h_arr, NUM_NUMS + 1, "statrt");
  print_array(h_res, NUM_NUMS + 1, "ii");
  print_array(h_result, NUM_NUMS + 1);
#endif

  bool f = false;
  int count = 0;
  for (int i = 0; i < NUM_NUMS; i++) {
    if (h_res[i] != h_result[i]) {
      f = true;
      count++;
    }
  }
  if (!f) {
    printf("yahoo!\n");
  } else {
    printf("fire and brimstone\n");
    printf("failed %d\n", count);
  }
#endif
}
const int BLOCK_SIZE = 1024;
const int ELEMS_PER_BLOCK = BLOCK_SIZE * 2;

// -----------------------------------------------------
// Block-level Blelloch exclusive scan
// -----------------------------------------------------
__global__ void block_scan(float *data, float *block_sums, int n) {
  __shared__ float temp[ELEMS_PER_BLOCK]; // size = ELEMS_PER_BLOCK
  int tid = threadIdx.x;
  if (tid == 0)
    printf("blok scan\n");

  const int start = blockIdx.x * ELEMS_PER_BLOCK;

  // Load elements (zero-pad if past end)
  for (int i = 0; i < 2; i++) {
    const int id = start + 2 * tid + i;
    temp[2 * tid + i] = (id < n) ? data[id] : 0.0f;
  }

  // --- upsweep ---
  for (int stride = 1; stride < ELEMS_PER_BLOCK; stride <<= 1) {
    __syncthreads();
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < ELEMS_PER_BLOCK)
      temp[idx] += temp[idx - stride];
  }

  // clear last element (exclusive scan)
  if (tid == 0 && blockIdx.x == 0)
    temp[ELEMS_PER_BLOCK - 1] = 0.0f;

  // --- downsweep ---
  for (int stride = ELEMS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < ELEMS_PER_BLOCK) {
      float t = temp[idx - stride];
      temp[idx - stride] = temp[idx];
      temp[idx] += t;
    }
  }
  __syncthreads();

  // Write back to global memory
  //
  for (int i = 0; i < 2; i++) {
    const int id = start + 2 * tid + i;
    if (id < n)
      data[id] = temp[2 * tid + i];
  }

  const int idx0 = start + 2 * tid;
  const int idx1 = start + 2 * tid + 1;

  // if there is another , each block sets up carry
  // Write total block sums
  if (block_sums && tid == 0) {
    const int last_valid = min(ELEMS_PER_BLOCK, n - start) - 1;

    const float iv = (idx1 < n) ? data[idx1] : ((idx0 < n) ? data[idx0] : 0.0f);
    const float val = temp[last_valid] + iv;

    block_sums[blockIdx.x] = val;
    printf("last_valid %d\n", last_valid);
    printf("block_sums[%d] = %f;\n", blockIdx.x, val);
  }
}

// -----------------------------------------------------
// Add scanned block offsets to each block’s data
// -----------------------------------------------------
__global__ void add_offsets(float *data, const float *block_offsets, int n) {
  int block = blockIdx.x;
  int tid = threadIdx.x;

  int start = block * ELEMS_PER_BLOCK;
  int gid0 = start + 2 * tid;
  int gid1 = gid0 + 1;

  float offset = (block > 0) ? block_offsets[block - 1] : 0.0f;

  if (gid0 < n)
    data[gid0] += offset;
  if (gid1 < n)
    data[gid1] += offset;
}

// -----------------------------------------------------
// Recursive Blelloch scan driver
// -----------------------------------------------------
void blelloch_scan(float *d_data, int N) {
  const int numBlocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
#ifdef DBG
  printf("blelloch_scan called w %p ,, %d \n", d_data, N);
  printf("numblocks %d \n", numBlocks);
#endif

  float *d_block_sums = nullptr;
  if (numBlocks > 1) {
    cudaMalloc(&d_block_sums, (numBlocks + 1) * sizeof(float));
    cudaMemset(&d_block_sums, 0, (numBlocks + 1) * sizeof(float));
  }
  // Phase 1: scan each block locally
  block_scan<<<numBlocks, BLOCK_SIZE, ELEMS_PER_BLOCK * sizeof(float)>>>(
      d_data, d_block_sums, N);

  // Phase 2: recursively scan block sums
  if (numBlocks > 1) {
    cudaDeviceSynchronize();
    blelloch_scan(d_block_sums, numBlocks);
  }

  // Phase 3: add scanned block offsets back
  add_offsets<<<numBlocks, BLOCK_SIZE>>>(d_data, (const float *)d_block_sums,
                                         N);
  cudaDeviceSynchronize();

  cudaFree(d_block_sums);
}
