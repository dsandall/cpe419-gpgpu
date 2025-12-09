#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

// Device kernel for inclusive prefix sum
void prefix_sum_kernel(queue &q, std::vector<float> &weights,
                       float *prefix_dev) {
  size_t n = weights.size();

  // Copy weights to device buffer
  buffer<float, 1> w_buf(weights.data(), range<1>(n));
  buffer<float, 1> prefix_buf(prefix_dev, range<1>(n));

  q.submit([&](handler &h) {
     auto w = w_buf.get_access<access::mode::read>(h);
     auto p = prefix_buf.get_access<access::mode::write>(h);

     h.parallel_for(range<1>(n), [=](id<1> i) {
       // naive inclusive scan (O(N^2)) for simplicity
       float sum = 0.0f;
       for (size_t j = 0; j <= i; j++)
         sum += w[j];
       p[i] = sum;
     });
   }).wait();
}

// Fractional knapsack
float fractional_knapsack_sycl(const std::vector<float> &values,
                               const std::vector<float> &weights,
                               float capacity) {
  size_t n = values.size();

  // --- Host: sort indices by density descending ---
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; i++)
    idx[i] = i;

  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    return (values[a] / weights[a]) > (values[b] / weights[b]);
  });

  // --- Build sorted weights ---
  std::vector<float> sorted_w(n);
  std::vector<float> sorted_v(n);
  for (size_t i = 0; i < n; i++) {
    sorted_w[i] = weights[idx[i]];
    sorted_v[i] = values[idx[i]];
  }

  queue q;

  // --- Device: prefix sum of weights ---
  std::vector<float> prefix(n);
  prefix_sum_kernel(q, sorted_w, prefix.data());

  // --- Host: greedy selection with fractional item ---
  float total_value = 0.0f;
  float total_weight = 0.0f;

  for (size_t i = 0; i < n; i++) {
    if (prefix[i] <= capacity) {
      total_value += sorted_v[i];
      total_weight += sorted_w[i];
    } else {
      float remaining = capacity - total_weight;
      if (remaining > 0)
        total_value += sorted_v[i] * (remaining / sorted_w[i]);
      break;
    }
  }

  return total_value;
}
void gen_runner(float (*funcPtr)(const std::vector<float> &values,
                                 const std::vector<float> &weights,
                                 float capacity),
                float n, float cap) {

  std::srand(42); // reproducible random numbers

  // --- 500-element test case ---
  std::vector<float> values1(n), weights1(n);
  for (size_t i = 0; i < n; i++) {
    values1[i] = static_cast<float>(std::rand() % 100 + 1);
    weights1[i] = static_cast<float>(std::rand() % 50 + 1);
  }

  auto start = std::chrono::high_resolution_clock::now();
  float result1 = funcPtr(values1, weights1, cap);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << n << "-element test max value: " << result1
            << " (time: " << elapsed.count() << " s)" << std::endl;
}

void simple_runner(float (*funcPtr)(const std::vector<float> &values,
                                    const std::vector<float> &weights,
                                    float capacity)) {

  // --- hardcoded small case ---
  std::vector<float> values = {60, 100, 120};
  std::vector<float> weights = {10, 20, 30};
  float capacity = 50;

  auto start = std::chrono::high_resolution_clock::now();
  float result = funcPtr(values, weights, capacity);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Small test max value: " << result
            << " (time: " << elapsed.count() << " s)" << std::endl;
}

int main() {

  float (*funcPtr)(const std::vector<float> &values,
                   const std::vector<float> &weights, float capacity);
  funcPtr = fractional_knapsack_sycl;

  simple_runner(funcPtr);
  gen_runner(funcPtr, 500, 1000);
  gen_runner(funcPtr, 5000000, 60000);

  return 0;
}
