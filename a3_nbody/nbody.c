#include "timer.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f

typedef struct {
  float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *data, int n) {
  srand(67);
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

// for each body (in parallel):
//  for all bodies: calculate force between them
//    d_xyz = current dist between
//    invDist = 1/sqrt(d_xyz**2)
//    force_xyz += d_xyz * invdist**3
//   set new position based on force between them
//
//

__global__ void forceOneBody(Body *bodies, float dt, int n) {
  float Fx = 0.0f;
  float Fy = 0.0f;
  float Fz = 0.0f;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

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

  // TODO: synch threads to keep data safe?
  // or double buffer?
  // would like to know what fast

  bodies[i].vx += dt * Fx;
  bodies[i].vy += dt * Fy;
  bodies[i].vz += dt * Fz;
}

void bodyForce(Body *p, float dt, int n) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

int main(const int argc, const char **argv) {

  int nBodies = 30000;
  if (argc > 1)
    nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf = (float *)malloc(bytes);
  Body *p = (Body *)buf;

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces

    // for all bodies, update position
    for (int i = 0; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx * dt;
      p[i].y += p[i].vy * dt;
      p[i].z += p[i].vz * dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed;
    }
#ifndef SHMOO
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
#endif
  }
  double avgTime = totalTime / (double)(nIters - 1);

#ifdef SHMOO
  printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#else
  printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per "
         "second.\n",
         nIters, rate);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies,
         1e-9 * nBodies * nBodies / avgTime);
#endif

  free(buf);
}
