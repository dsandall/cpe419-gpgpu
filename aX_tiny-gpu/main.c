#define N 20
#define M 30
#define K 40
#define DATA_TYPE unsigned _BitInt(8)

void simple(DATA_TYPE A[N], DATA_TYPE B[N], DATA_TYPE C[N]) {
  int i;
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}

void matmul(DATA_TYPE A[N][K], DATA_TYPE B[K][M], DATA_TYPE C[N][M]) {
  int i, j, k;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
