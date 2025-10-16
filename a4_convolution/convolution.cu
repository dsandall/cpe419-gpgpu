#include "../helpful.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#define KERNEL_N 3
#define IMAGE_N 11ULL
#define OUTPUT_N (IMAGE_N - KERNEL_N + 1)

// tile size 16 makes our threads per block 256
#define TILE_SZ 16

__global__ void simpleConv(float *img, float *kernel, float *imgf, int Nx,
                           int Ny, int kernel_size, int center) {
  // given indexes for the output matrix,
  int output_i = blockIdx.x * TILE_SZ + threadIdx.x;
  int output_j = blockIdx.y * TILE_SZ + threadIdx.y;
  // just shift indexes over to find associated input indexes
  int image_i = output_i + 1;
  int image_j = output_j + 1;

  float sum = 0;
  // MAC each surrounding pixel in the (square) kernel zone
  for (int ki = 0; ki < kernel_size; ki++) {
    for (int kj = 0; kj < kernel_size; kj++) {
      int ii = image_j + kj - center;
      int jj = image_i + ki - center;
      sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
    }
  }
  // and store the result in device memory
  imgf[image_i * Nx + image_j] = sum;
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
      imgf[i * Nx + j] = sum;
    }
  }
}

int main() {
  // malloc host memory
  size_t size_F = IMAGE_N * IMAGE_N * sizeof(float);
  size_t size_H = KERNEL_N * KERNEL_N * sizeof(float);
  size_t size_G = OUTPUT_N * OUTPUT_N * sizeof(float);
  float *h_F = (float *)malloc(size_F);
  float *h_H = (float *)malloc(size_H);
  float *h_G = (float *)malloc(size_G);
  float *h_G_sequential = (float *)malloc(size_G);

  // Initialize image with random integer values [0, 255]
  srand(0);
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

  // setup device
  float *d_F, *d_H, *d_G;
  checkCuda(cudaMalloc((void **)&d_F, size_F), "Alloc F");
  checkCuda(cudaMalloc((void **)&d_H, size_H), "Alloc H");
  checkCuda(cudaMalloc((void **)&d_G, size_G), "Alloc G");

  // TODO: make async later
  checkCuda(cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice), "Memcpy F");
  checkCuda(cudaMemcpy(d_H, h_H, size_H, cudaMemcpyHostToDevice), "Memcpy H");
  checkCuda(cudaMemcpy(d_G, h_G, size_G, cudaMemcpyHostToDevice), "Memcpy G");

  const int output_elements =
      IMAGE_N - (KERNEL_N / 2) +
      1; // this should be the number of vars in the output

  dim3 dimBlock(TILE_SZ, TILE_SZ); // constant tile size
  dim3 dimGrid(ceilDiv(OUTPUT_N, TILE_SZ),
               ceilDiv(OUTPUT_N, TILE_SZ)); // varied based on image size

  printf("dimblock is %d x %d\n", TILE_SZ, TILE_SZ);
  printf("dimgrid  is %d\n", dimGrid.x);

  // do parallel
  simpleConv<<<dimGrid, dimBlock>>>(d_F, d_H, d_G, IMAGE_N, IMAGE_N, KERNEL_N,
                                    KERNEL_N / 2);
  checkCuda(cudaMemcpy(h_G, d_G, size_G, cudaMemcpyDeviceToHost),
            "Memcpy G back to the host");

  // do sequential
  cudaEventRecord(startEvent);
  doConv(h_F, h_H, h_G_sequential, IMAGE_N, IMAGE_N, KERNEL_N, KERNEL_N / 2);
  cudaEventRecord(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("time elapsed(ms, sequential): %f\n", ms);

  // verify
  checkFloats(OUTPUT_N * OUTPUT_N, h_G, h_G_sequential);

  // Cleanup
  free(h_F);
  free(h_G);
  free(h_H);
  free(h_G_sequential);
}
