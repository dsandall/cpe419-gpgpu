#include <cstdlib>
#include <math.h>
#include <stdio.h>

const float lr = 0.2f;
const int ITERATIONS = 2010;
const int NUM_HIDDEN = 2;
#define TRAIN

// ---------------------- Activation ----------------------
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ float sigmoid_prime(float y) {
  return y * (1.0f - y); // assumes y = sigmoid(x)
}

// ---------------------- Forward --------------------------
__global__ void forward_layer(const float *A_in, const float *W, const float *B,
                              float *A_out, int in_dim, int out_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_dim)
    return;

  // mac, bias, and activation func
  float sum = 0.0f;
  for (int j = 0; j < in_dim; j++) {
    sum += W[i * in_dim + j] * A_in[j];
  }
  A_out[i] = sigmoid(sum + B[i]);

#ifdef DBG
  if (i == NUM_HIDDEN) {
    printf("Predicted output: %.4f\n", A_out[NUM_HIDDEN]);
  }
#endif
}

// ---------------------- Backward -------------------------
__global__ void backward_layer(const float *dA_out, const float *W,
                               const float *A_in, const float *A_out,
                               float *dA_in, float *dW, float *dB, int in_dim,
                               int out_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_dim)
    return;

  // local gradient
  float dZ = dA_out[i] * sigmoid_prime(A_out[i]);
  dB[i] = dZ;

  for (int j = 0; j < in_dim; j++) {
    dW[i * in_dim + j] = dZ * A_in[j];
    atomicAdd(&dA_in[j], W[i * in_dim + j] * dZ);
  }
}

// ---------------------- Update weights -------------------
__global__ void update_params(float *W, float *B, const float *dW,
                              const float *dB, float lr, int W_size,
                              int B_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < W_size)
    W[i] -= lr * dW[i];
  if (i < B_size)
    B[i] -= lr * dB[i];
}

// ---------------------- Host code ------------------------
int main() {
  // Dimensions
  const int input_dim = 5;
  const int hidden_dim = 1024;
  const int output_dim = 1;

  // Host data
  float h_input[input_dim] = {0.5f, 0.2f, 0.1f, 0.1f, 0.1f};
  float h_target[output_dim] = {1.0f};

  // allocate host weights
  // init host weights
  float *h_B[NUM_HIDDEN];
  float *h_W[NUM_HIDDEN];

  // Device allocations
  // init device inputs
  // init weights and biases on device
  float *d_A[NUM_HIDDEN + 1];
  float *d_W[NUM_HIDDEN];
  float *d_B[NUM_HIDDEN];

  for (int i = 0; i < NUM_HIDDEN; i++) {
    int i_dim = hidden_dim;
    int o_dim = hidden_dim;
    if (i == NUM_HIDDEN - 1) {
      o_dim = output_dim;
      cudaMalloc(&d_A[NUM_HIDDEN], output_dim * sizeof(float));
    }
    if (i == 0)
      i_dim = input_dim;

    // host memory and initialization
    h_B[i] = (float *)std::malloc(o_dim * sizeof(float));
    h_W[i] = (float *)std::malloc(o_dim * i_dim * sizeof(float));
    for (int z = 0; z < o_dim; z++) {
      h_B[i][z] = ((float)(rand() % 256) / 64.0) - 2.0;
      printf("%f\n", h_B[i][z]);
    }
    for (int z = 0; z < i_dim * o_dim; z++) {
      h_W[i][z] = ((float)(rand() % 256) / 64.0) - 2.0;
      printf("%f\n", h_W[i][z]);
    }

    // device memory and copying
    cudaMalloc(&d_A[i], i_dim * sizeof(float));
    cudaMalloc(&d_W[i], o_dim * i_dim * sizeof(float));
    cudaMalloc(&d_B[i], o_dim * sizeof(float));

    cudaMemcpy(d_B[i], h_B[i], o_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W[i], h_W[i], i_dim * o_dim * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  // also copy the testing input
  cudaMemcpy(d_A[0], h_input, sizeof(h_input), cudaMemcpyHostToDevice);

#ifdef TRAIN
  // delta (for training passes)
  float *d_dA[NUM_HIDDEN];
  float *d_dW[NUM_HIDDEN];
  float *d_dB[NUM_HIDDEN];
  for (int i = 0; i < NUM_HIDDEN; i++) {
    int i_dim = hidden_dim;
    int o_dim = hidden_dim;
    if (i == NUM_HIDDEN - 1)
      o_dim = output_dim;
    if (i == 0)
      i_dim = input_dim;

    cudaMalloc(&d_dA[i], o_dim * sizeof(float));
    cudaMalloc(&d_dW[i], o_dim * i_dim * sizeof(float));
    cudaMalloc(&d_dB[i], o_dim * sizeof(float));
  }

  for (int i = 0; i < ITERATIONS; i++) {
#endif

    // -------- Forward pass --------
    // 1 thread per neuron in that layer
    // A_out = A*W + B;
    for (int i = 0; i < NUM_HIDDEN; i++) {
      int i_dim = hidden_dim;
      int o_dim = hidden_dim;
      if (i == NUM_HIDDEN - 1)
        o_dim = output_dim;
      if (i == 0)
        i_dim = input_dim;

      forward_layer<<<1, o_dim>>>(d_A[i], d_W[i], d_B[i], d_A[i + 1], i_dim,
                                  o_dim);
    }

#ifdef TRAIN
    // -------- Error = Actual out - Expected out --------
    float h_output[output_dim];
    cudaMemcpy(h_output, d_A[NUM_HIDDEN], sizeof(h_output),
               cudaMemcpyDeviceToHost);
    float h_dA2[output_dim];
    for (int i = 0; i < output_dim; i++)
      h_dA2[i] = (h_output[i] - h_target[i]); // dL/dA = (pred - target)

    printf("error is %f\n", h_dA2[0]);
    cudaMemcpy(d_dA[1], h_dA2, sizeof(h_dA2), cudaMemcpyHostToDevice);
    cudaMemset(d_dA[0], 0, hidden_dim * sizeof(float));

    // -------- Backward pass --------
    for (int i = NUM_HIDDEN - 1; i >= 0; i--) {
      float *dA;
      int i_dim = hidden_dim;
      int o_dim = hidden_dim;
      if (i == NUM_HIDDEN - 1) {
        dA = d_dA[i - 1];
        o_dim = output_dim;
      }
      if (i == 0) {
        dA = d_A[0];
        i_dim = input_dim;
      }

      backward_layer<<<1, o_dim>>>(d_dA[i], d_W[i], d_A[i], d_A[i + 1], dA,
                                   d_dW[i], d_dB[i], i_dim, o_dim);
    }

    // -------- Update weights --------
    for (int i = NUM_HIDDEN - 1; i >= 0; i--) {
      int i_dim = hidden_dim;
      int o_dim = hidden_dim;
      if (i == NUM_HIDDEN - 1) {
        o_dim = output_dim;
      }
      if (i == 0) {
        i_dim = input_dim;
      }

      update_params<<<1, o_dim * i_dim>>>(d_W[i], d_B[i], d_dW[i], d_dB[i], lr,
                                          o_dim * i_dim, o_dim);
    }
  }
#endif

  // Cleanup
  // cudaFree(d_A[0]);
  // cudaFree(d_A[1]);
  // cudaFree(d_A[2]);

#ifdef TRAIN
////  cudaFree(d_dA2);
// cudaFree(d_dA1);
// cudaFree(d_dW1);
// cudaFree(d_dB1);
// cudaFree(d_dW2);
// cudaFree(d_dB2);
#endif
}
