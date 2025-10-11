#include "boiler.c"
#include "timer.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void forceOneBody(Body *bodies, float dt, int n, int stream_offset) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;

  float Fx = 0.0f;
  float Fy = 0.0f;
  float Fz = 0.0f;

  for (int j = 0; j < n; j++) {
    float dx = bodies[j].x - bodies[i].x;
    float dy = bodies[j].y - bodies[i].y;
    float dz = bodies[j].z - bodies[i].z;
    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    float invDist = 1.0f / sqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    Fx += dx * invDist3;
    Fy += dy * invDist3;
    Fz += dz * invDist3;
  }

  bodies[i].vx += dt * Fx;
  bodies[i].vy += dt * Fy;
  bodies[i].vz += dt * Fz;
}

__global__ void updateBodyPos(Body *p, float dt, int stream_offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;
  p[i].x += p[i].vx * dt;
  p[i].y += p[i].vy * dt;
  p[i].z += p[i].vz * dt;
}

int main(const int argc, const char **argv) {
  printf("hello cuda\n");

  int n;
  if (argc > 1)
    n = atoi(argv[1]);
  else
    n = 30000;

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  // init on host
  float *buf;
  cudaHostAlloc(&buf, n * sizeof(Body), cudaHostAllocDefault);
  randomizeBodies(buf, 6 * n); // Init pos / vel data

  // on device
  Body *p;
  cudaMalloc(&p, n * sizeof(Body));

  cudaMemcpy(&p, buf, n * sizeof(Body), cudaMemcpyHostToDevice);
  // cudaMemcpy(buf, &p, n * sizeof(Body), cudaMemcpyDeviceToHost);
  //
  const int num_blocks = 28;
  float ms; // elapsed time in milliseconds

  // create events and streams
  const int nStreams = 4;
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventCreate(&dummyEvent);
  for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);

  // run iterations
  double totalTime = 0.0;
  for (int iter = 1; iter <= nIters; iter++) {

    const int streamSize = n / nStreams;
    const int streamBytes = streamSize * sizeof(Body);
    const int gridSize = 28;
    const int threads = streamSize / gridSize;

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < nStreams; ++i) {
      // NOTE:streams are kind of pointless unless you are doing host-device
      // xfers interleaved with kernel execution .here is an example of the
      // syntax WITH mem transfers
      //
      /*cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes,
                                cudaMemcpyHostToDevice, stream[i]);
       kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a,
       offset);
      cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes,
                                cudaMemcpyDeviceToHost, stream[i]);
                                */

      forceOneBody<<<gridSize, threads, 0, stream[i]>>>(p, dt, n,
                                                        i * streamSize);
    }
    for (int i = 0; i < nStreams; ++i) {
      cudaStreamSynchronize(stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
      updateBodyPos<<<gridSize, threads, 0, stream[i]>>>(p, dt, i * streamSize);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    if (iter > 1) { // First iter is warm up
      totalTime += ms;
    }
  }
  double avgTime = totalTime / (double)(nIters - 1);

  printf("n = %d, n*n/avgTime = %0.3f\n", n, n * n / avgTime);
  printf("totalTime = %0.3f, avgTime = %0.3f\n", totalTime, avgTime);
  /*printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per "
         "second.\n",
         nIters, rate);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies,
         1e-9 * nBodies * nBodies / avgTime);*/

  cudaFree(buf);
}
