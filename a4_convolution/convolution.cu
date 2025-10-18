
#include "../helpful.cuh"
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define KERNEL_N 77
#define IMAGE_N 611ULL
#define OUTPUT_N (IMAGE_N - KERNEL_N + 1)

#define COMPARE_IM2COL
#define COMPARE_SIMPLE_CONV
#define COMPARE_IM2COL_SEQ
#define COMPARE_SEQ

// tile size 16 makes our threads per block 256
#define TILE_SZ 16

const int output_elements =
    IMAGE_N - (KERNEL_N / 2) +
    1; // this should be the number of vars in the output

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
  const int out_w = W - KW + 1;

  // const int col_h = KH * KW;
  const int col_h = KERNEL_N * KERNEL_N;
  const int col_w = out_h * out_w;
  // float *col = (float *)malloc(sizeof(float) * col_h * col_w);
  float col[col_h];

  // for (int i = 0; i < out_h; ++i) {
  //  for (int j = 0; j < out_w; ++j) {
  // given indexes for the output matrix,
  int i_block = blockIdx.x * TILE_SZ;
  int j_block = blockIdx.y * TILE_SZ;
  int i = i_block + threadIdx.x;
  int j = j_block + threadIdx.y;
  // just shift indexes over to find associated input indexes
  int image_i = i + KH / 2;
  int image_j = j + KH / 2;

  bool valid = image_i < (W - KH / 2) && image_j < H - KH / 2;
  if (!valid)
    return; // diverged
            //

  // UNFINISHED:
  //__shared__ float patch[(KERNEL_N + 17)*(KERNEL_N + 17)];
  // const int tid = threadIdx.x + 16 * threadIdx.y;
  // const int block_offset = i_block + j_block * IMAGE_N;

  // patch[threadIdx.x + ((KERNEL_N+17) * threadIdx.y)] = input[tid +
  // block_offset]; tid += 256; if (tid < (KERNEL_N + 17)*(KERNEL_N + 17)){
  //   patch[threadIdx.x + ((KERNEL_N+17) * threadIdx.y)] = input[tid +
  //   block_offset];
  // }
  //

  // build col
  // int idx = (i * out_w + j) * (KH * KW);
  int idx = 0;
  for (int ki = 0; ki < KH; ++ki) {
    for (int kj = 0; kj < KW; ++kj) {
      col[idx++] = input[(i + ki) * W + (j + kj)];
    }
  }

  // do matmul
  float sum = 0.0f;
  int p = i * out_w + j;
  for (int k = 0; k < col_h; ++k) {
    // sum += kernel[k] * col[k + p * col_h];
    sum += kernel[k] * col[k];
  }
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
  im2col_par<<<dimGrid, dimBlock>>>(d_F, d_H, d_G, IMAGE_N, IMAGE_N, KERNEL_N,
                                    KERNEL_N);

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
