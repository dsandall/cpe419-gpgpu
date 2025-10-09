#include "boiler.c"
#include "timer.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void forceOneBody(Body *bodies, float dt, int n) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void updateBodyPos(Body *p, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  p[i].x += p[i].vx * dt;
  p[i].y += p[i].vy * dt;
  p[i].z += p[i].vz * dt;
}

int main(const int argc, const char **argv) {

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

  // run iterations
  double totalTime = 0.0;
  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    forceOneBody<<<28, n / (28 * 1024)>>>(p, dt, n);
    //  synch all threads nd blockshere
    updateBodyPos<<<28, n / (28 * 1024)>>>(p, dt);

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed;
    }
    // printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters - 1);

  printf("n = %d, n*n/avgTime = %0.3f\n", n, 1e-9 * n * n / avgTime);
  /*printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per "
         "second.\n",
         nIters, rate);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies,
         1e-9 * nBodies * nBodies / avgTime);*/

  cudaFree(buf);
}
