#include "../helpful.cuh"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define KERNEL_N 9
#define IMAGE_N 33ULL
#define OUTPUT_N (IMAGE_N - KERNEL_N + 1)

#define COMPARE_IM2COL
#define COMPARE_SIMPLE_CONV
#define COMPARE_IM2COL_SEQ
#define COMPARE_SEQ

// tile size 16 makes our threads per block 256
#define TILE_SZ 16

__global__ void simpleConv(float *img, float *kernel, float *imgf, int Nx,
                           int Ny, int kernel_size, int center) {
  // given indexes for the output matrix,
  int output_i = blockIdx.x * TILE_SZ + threadIdx.x;
  int output_j = blockIdx.y * TILE_SZ + threadIdx.y;
  // just shift indexes over to find associated input indexes
  int image_i = output_i + center;
  int image_j = output_j + center;

  bool valid = image_i < (Ny - center) && image_j < Nx - center;
  if (!valid)
    return; // diverged
  // printf("(%d,%d) -> (%d,%d)\n", output_i, output_j, image_i, image_j);

  // MAC each surrounding pixel in the (square) kernel zone

  float sum = 0;
  // TODO: shared mem
  for (int ki = 0; ki < kernel_size; ki++) {
    for (int kj = 0; kj < kernel_size; kj++) {
      int ii = image_j + kj - center;
      int jj = image_i + ki - center;
      sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
    }
  }

  // and store the result in device memory
  imgf[output_i * OUTPUT_N + output_j] = sum;
}
__host__ void doConv(float *img, float *kernel, float *imgf, int Nx, int Ny,
                     int kernel_size, int center) {
  // for each pixel in the image (excluding borders)
  for (int i = center; i < (Ny - center); i++) {
    for (int j = center; j < (Nx - center); j++) {
      float sum = 0;
      // MAC each surrounding pixel in the (square) kernel zone
      for (int ki = 0; ki < kernel_size; ki++) {
        for (int kj = 0; kj < kernel_size; kj++) {
          int ii = j + kj - center;
          int jj = i + ki - center;
          sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
        }
      }
      // and store the result
      // imgf[i * Nx + j] = sum;
      int out_i = i - center;
      int out_j = j - center;
      imgf[out_i * OUTPUT_N + out_j] = sum;
    }
  }
}

__global__ void im2col_par(float *input, float *kernel, float *out, int H,
                           int W, int KH, int KW) {

  const int out_h = H - KH + 1;
  const int out_w = out_h;
  const int col_h = KERNEL_N * KERNEL_N;
  // for (int i = 0; i < out_h; ++i) {
  //  for (int j = 0; j < out_w; ++j) {
  // given indexes for the output matrix,

  const int patch_n = (KERNEL_N + TILE_SZ - 1);
  const int patch_alt = (KERNEL_N + IMAGE_N - 1);
  __shared__ float patch[patch_n][patch_n];
  int ltpat_n = (patch_alt < patch_n) ? patch_alt : patch_n;
  float colloc[col_h];

  // populate shared mem with image data
  int idx = (threadIdx.x + threadIdx.y * TILE_SZ);
  while (idx < (ltpat_n * ltpat_n)) {
    int pi = idx % ltpat_n;
    int pj = idx / ltpat_n;
    int ii = pi + blockIdx.x * TILE_SZ;
    int ij = pj + blockIdx.y * TILE_SZ;
    patch[pi][pj] = input[ii + ij * W];
    // printf("ltpat_n = %d\n", ltpat_n);
    // printf("%d(%d,%d) : %d,%d - %d,%d\n", idx, blockIdx.x, blockIdx.y, ii,
    // ij,
    //        pi, pj);

    idx += TILE_SZ * TILE_SZ;
  };

  const int i = blockIdx.x * ltpat_n + threadIdx.x;
  const int j = blockIdx.y * ltpat_n + threadIdx.y;
  bool valid = i < OUTPUT_N && j < OUTPUT_N;
  if (!valid)
    return; // diverge all threads that dont have an output px

  __syncthreads();
  // if (idx_start == 0) {
  //   int n = patch_n * patch_n;
  //   for (int i = 0; i < n + 3; i++) {
  //     if (i >= n)
  //       printf("(just beyond array):");
  //     // printf("%d: %6.2f | %6.2f\n", i, patch[i], input[i]);
  //   }
  //   printf("\n");
  // }

  // build col
  idx = 0;
  // idx = idx_start * col_h;
  for (int ki = 0; ki < KH; ++ki) {
    for (int kj = 0; kj < KW; ++kj) {
      // colloc[idx++] = patch[((threadIdx.x + kj) * patch_n) + threadIdx.y +
      // ki];
      int ix = (blockIdx.x * ltpat_n + threadIdx.x + ki);
      int iy = (blockIdx.y * ltpat_n + threadIdx.y + kj);
      float iv = input[ix * W + iy];

      int pi = threadIdx.x + ki;
      int pj = threadIdx.y + kj;
      float pv = patch[pj][pi];
      colloc[idx++] = pv;

      // if (ix != pi || iy != pj) {
      //   printf("coord not match:%d,%d    %d,%d", ix, iy, pi, pj);
      // }
      if (pv != iv) {
        printf("bad: pi,pj %d,%d = %d not %d\n", pi, pj, iv, pv);
        printf("bad2: ix iy %d,%d\n", ix, iy);
      } else {
        // printf("good\n");
      }
    }
  }

  // do matmul
  float sum = 0.0f;
  for (int k = 0; k < col_h; ++k) {
    // sum += kernel[k] * colloc[k + idx_start * col_h];
    sum += kernel[k] * colloc[k];
  }

  int p = i * out_w + j; // note: == idx start, but with block offset
  out[p] = sum;
}

__host__ void im2col_mm_seq(float *input, float *kernel, float *out, int H,
                            int W, int KH, int KW) {
  int out_h = H - KH + 1;
  int out_w = W - KW + 1;
  int patch = 0;

  int col_h = KH * KW;
  int col_w = out_h * out_w;
  float *col = (float *)malloc(sizeof(float) * col_h * col_w);

  for (int i = 0; i < out_h; ++i) {
    for (int j = 0; j < out_w; ++j) {

      int idx = patch * (KH * KW);
      for (int ki = 0; ki < KH; ++ki) {
        for (int kj = 0; kj < KW; ++kj) {
          col[idx++] = input[(i + ki) * W + (j + kj)];
        }
      }
      patch++;
    }
  }

  // matmul: out = Wᵀ * col
  // (1 x KH*KW) * (KH*KW x out_h*out_w)
  for (int p = 0; p < col_w; ++p) {
    float sum = 0.0f;
    for (int k = 0; k < col_h; ++k) {
      sum += kernel[k] * col[k + p * col_h];
    }
    out[p] = sum;
  }

  free(col);
}

int main() {
  // malloc host memory
  const size_t size_F = IMAGE_N * IMAGE_N * sizeof(float);
  const size_t size_H = KERNEL_N * KERNEL_N * sizeof(float);
  const size_t size_G = OUTPUT_N * OUTPUT_N * sizeof(float);

  float *h_F = (float *)malloc(size_F);
  float *h_H = (float *)malloc(size_H);
  float *h_G_known_good = (float *)malloc(size_G);
  h_G_known_good[0] = (float)0xDEADBEEF;
  float *h_G_comp = (float *)malloc(size_G);

  // Initialize image with random integer values [0, 255]
  srand(time(NULL));
  for (int i = 0; i < IMAGE_N * IMAGE_N; i++)
    h_F[i] = (float)(rand() % 256); // convert to float automatically

  // Initialize kernel with small random floats [-1, 1]
  for (int i = 0; i < KERNEL_N * KERNEL_N; i++)
    h_H[i] = getRandOneish();

  // init timing stuff
  cudaEvent_t startEvent, stopEvent;
  float ms;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // setup device memory
  float *d_F, *d_H, *d_G;
  checkCuda(cudaMalloc((void **)&d_F, size_F), "Alloc F");
  checkCuda(cudaMalloc((void **)&d_H, size_H), "Alloc H");
  checkCuda(cudaMalloc((void **)&d_G, size_G), "Alloc G");

  // setup device threads and blocks
  dim3 dimBlock(TILE_SZ, TILE_SZ); // constant tile size
  dim3 dimGrid(ceilDiv(OUTPUT_N, TILE_SZ),
               ceilDiv(OUTPUT_N, TILE_SZ)); // varied based on image size

  printf("dimblock is %d x %d\n", TILE_SZ, TILE_SZ);
  printf("dimgrid  is %d\n", dimGrid.x);

#ifdef COMPARE_SIMPLE_CONV
  cudaEventRecord(startEvent);

  // TODO: make async later
  checkCuda(cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice), "Memcpy F");
  checkCuda(cudaMemcpy(d_H, h_H, size_H, cudaMemcpyHostToDevice), "Memcpy H");
  // checkCuda(cudaMemcpy(d_G, h_G, size_G, cudaMemcpyHostToDevice), "Memcpy
  // G");
  simpleConv<<<dimGrid, dimBlock>>>(d_F, d_H, d_G, IMAGE_N, IMAGE_N, KERNEL_N,
                                    KERNEL_N / 2);

  checkCuda(cudaMemcpy(h_G_known_good, d_G, size_G, cudaMemcpyDeviceToHost),
            "Memcpy G back to the host");

  cudaDeviceSynchronize();

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, parallel): %f\n", ms);
#endif

#ifdef COMPARE_IM2COL
  cudaEventRecord(startEvent);

  // TODO: make async later
  checkCuda(cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice), "Memcpy F");
  checkCuda(cudaMemcpy(d_H, h_H, size_H, cudaMemcpyHostToDevice), "Memcpy H");
  const int out_h = IMAGE_N - KERNEL_N + 1;
  const int out_w = out_h;
  const int col_h = KERNEL_N * KERNEL_N;
  const int col_w = out_h * out_w;
  const size_t size_col = sizeof(float) * col_h * col_w;
  float *d_col;
  checkCuda(cudaMalloc((void **)&d_col, size_col), "Alloc col");
  im2col_par<<<dimGrid, dimBlock>>>(d_F, d_H, d_G, IMAGE_N, IMAGE_N, KERNEL_N,
                                    KERNEL_N);

  // cudaDeviceSynchronize();
  // mm_par<<<dimGrid, dimBlock>>>(d_col, d_H, d_G, IMAGE_N, IMAGE_N, KERNEL_N,
  //                               KERNEL_N);

  checkCuda(cudaMemcpy(h_G_comp, d_G, size_G, cudaMemcpyDeviceToHost),
            "Memcpy G back to the host");

  cudaDeviceSynchronize();

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, parallel im2col): %f\n", ms);

  // verify
  checkFloats(OUTPUT_N * OUTPUT_N, h_G_known_good, h_G_comp);

#endif

#ifdef COMPARE_IM2COL_SEQ
  float *h_Fcolumn = (float *)malloc(size_F);

  cudaEventRecord(startEvent);
  im2col_mm_seq(h_F, h_H, h_G_comp, IMAGE_N, IMAGE_N, KERNEL_N, KERNEL_N);
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, im2col sequential): %f\n", ms);

  // verify
  checkFloats(OUTPUT_N * OUTPUT_N, h_G_known_good, h_G_comp);
#endif

  // do sequential
#ifdef COMPARE_SEQ
  cudaEventRecord(startEvent);
  doConv(h_F, h_H, h_G_comp, IMAGE_N, IMAGE_N, KERNEL_N, KERNEL_N / 2);
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, sequential): %f\n", ms);

  // verify
  checkFloats(OUTPUT_N * OUTPUT_N, h_G_known_good, h_G_comp);
#endif

  // Cleanup
  free(h_F);
  free(h_G_known_good);
  free(h_H);
  free(h_G_comp);
}
