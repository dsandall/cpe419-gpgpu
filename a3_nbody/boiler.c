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
