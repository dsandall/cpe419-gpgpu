#include "../helpful.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define N 1024ULL     // Matrix dimension (N x N)
#define NUM_STREAMS 4 // Number of CUDA streams
// split the 1024x1024 matrix into 4 streams (4 equal rows)
// each of these rows are 256x1024, split into

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

__global__ void matMulTiled(const float *A, const float *B, float *C, int n) {
  // Calculates a single element's result, by cooperating within a thread block
  // to collect all necessary data, across tiles

  // Shared memory (shared per block)
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  // use thread index to calculate global position
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Cvalue = 0.0f;
  // each element in matrix C requires data collection along the entire matrix
  // by row and col for each tile in a row/column of the entire NxN matrix,
  for (int t = 0; t < (n / TILE_WIDTH); ++t) {
    // cache the values of that tile element:
    //  calculate the address,
    //  put A and B in cache (if in bounds)
    //  (we are essentially walking row and column (A and B) at once)
    bool AinBounds = (Row < n && (t * TILE_WIDTH + tx) < n);
    As[ty][tx] = AinBounds ? A[Row * n + t * TILE_WIDTH + tx] : 0.0f;

    bool BinBounds = ((t * TILE_WIDTH + ty) < n && Col < n);
    Bs[ty][tx] = BinBounds ? B[(t * TILE_WIDTH + ty) * n + Col] : 0.0f;

    // wait for all threads in this block to finish populating the shared memory
    __syncthreads();

    // finally, multiply and accumulate
    for (int i = 0; i < TILE_WIDTH; ++i)
      Cvalue += As[ty][i] * Bs[i][tx];

    // don't start next iteration before all threads have retrieved from shared
    // mem
    __syncthreads();
  }

  // finally, copy Cval to the relevant location in device memory so it can be
  // accessed
  if (Row < n && Col < n)
    C[Row * n + Col] = Cvalue;
}

int main() {
  size_t size = N * N * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  float *h_C_seq = (float *)malloc(size);

  // init A and B to random vals from -1 to 1
  srand(0); // seed the random generator once
  for (int i = 0; i < N * N; i++) {
    h_A[i] = getRandOneish();
    h_B[i] = getRandOneish();
  }

  // init timing stuff
  cudaEvent_t startEvent, stopEvent;
  float ms;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // prepare streams and device memory
  float *d_A[NUM_STREAMS], *d_B, *d_C[NUM_STREAMS];
  cudaStream_t streams[NUM_STREAMS];

  checkCuda(cudaMalloc((void **)&d_B, size), "Alloc B");

  for (int i = 0; i < NUM_STREAMS; i++) {
    checkCuda(cudaMalloc((void **)&d_A[i], size / NUM_STREAMS), "Alloc A part");
    checkCuda(cudaMalloc((void **)&d_C[i], size / NUM_STREAMS), "Alloc C part");
    checkCuda(cudaStreamCreate(&streams[i]), "Stream create");
  }

  checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy B");

  int rowsPerStream = N / NUM_STREAMS; // 256x1024 for each stream
  dim3 dimBlock(TILE_WIDTH,
                TILE_WIDTH); // 16x16= spawn 256 threads for each block
  dim3 dimGrid(
      ceilDiv(N, TILE_WIDTH),
      ceilDiv(rowsPerStream,
              TILE_WIDTH)); // 64 x 16 blocks per stream
                            // NxN = 256 x (64x16) x 4 streams = 1024x1024

  // run kernel(s) and record time
  cudaEventRecord(startEvent, 0);
  for (int i = 0; i < NUM_STREAMS; i++) {
    int offset = i * rowsPerStream * N;
    checkCuda(cudaMemcpyAsync(d_A[i], h_A + offset, size / NUM_STREAMS,
                              cudaMemcpyHostToDevice, streams[i]),
              "MemcpyAsync A");

    matMulTiled<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A[i], d_B, d_C[i], N);

    checkCuda(cudaMemcpyAsync(h_C + offset, d_C[i], size / NUM_STREAMS,
                              cudaMemcpyDeviceToHost, streams[i]),
              "MemcpyAsync C");
  }

  // Synchronize all streams
  for (int i = 0; i < NUM_STREAMS; i++)
    cudaStreamSynchronize(streams[i]);

  // report parallelized execution time
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, parallelized): %f\n", ms);

  // perform sequentially
  cudaEventRecord(startEvent);
  matrixmult(h_A, h_B, h_C_seq, N, N);
  cudaEventRecord(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, sequential): %f\n", ms);

  // Verify correctness
  checkFloats(N * N, h_C, h_C_seq);

  // Cleanup
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaFree(d_A[i]);
    cudaFree(d_C[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(d_B);
  free(h_A);
  free(h_B);
  free(h_C);
}
