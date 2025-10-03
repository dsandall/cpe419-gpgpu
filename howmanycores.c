#include <stdint.h> // Required for uint32_t
#include <stdio.h>
#include <stdlib.h> // For rand()
#include <sys/sysinfo.h>
#include <time.h>

struct timespec begin, end;
double elapsed;
#define MAT_N 500
#define MAT_M 1000
float mat_A[MAT_M][MAT_N];
float mat_B[MAT_N][MAT_M];
float mat_C[MAT_M][MAT_M]; // results in MxM matrix

int main(int argc, char *argv[]) {
  clock_gettime(CLOCK_MONOTONIC, &begin);

  printf("This system has %d processors configured and "
         "%d processors available.\n",
         get_nprocs_conf(), get_nprocs());

  printf("initializing\n");

  for (int i = 0; i < MAT_M; i++) {
    for (int j = 0; j < MAT_N; j++) {
      mat_A[i][j] = rand();
      mat_B[j][i] = rand();
    }
  }

  printf("done\n");

  float current;
  for (int i = 0; i < MAT_M; i++) {
    for (int j = 0; j < MAT_M; j++) {
      current = 0;
      for (int k = 0; k < MAT_N; k++) {
        current += mat_A[i][k] * mat_B[k][j];
      }
      mat_C[i][j] = current;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

  // spawn threads to do work here
  printf("took %f s\n", elapsed);
  return 0;
}
