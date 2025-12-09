#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <vector>

// Functor to compute value density
struct density_functor {
  __host__ __device__ float operator()(const thrust::tuple<float, float> &t) {
    return thrust::get<0>(t) / thrust::get<1>(t);
  }
};

float fractional_knapsack_cuda(const std::vector<float> &values,
                               const std::vector<float> &weights,
                               float capacity) {
  size_t n = values.size();

  // --- Copy data to device ---
  thrust::device_vector<float> d_values(values.begin(), values.end());
  thrust::device_vector<float> d_weights(weights.begin(), weights.end());

  // --- Compute value densities ---
  thrust::device_vector<float> d_density(n);
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                        d_values.begin(), d_weights.begin())),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(d_values.end(), d_weights.end())),
                    d_density.begin(), density_functor());

  // --- Sort by density descending ---
  thrust::device_vector<int> d_indices(n);
  thrust::sequence(d_indices.begin(), d_indices.end()); // 0..n-1

  // Sort indices by density
  thrust::sort_by_key(d_density.begin(), d_density.end(), d_indices.begin(),
                      thrust::greater<float>());

  // --- Reorder values and weights according to sorted indices ---
  thrust::device_vector<float> d_sorted_values(n);
  thrust::device_vector<float> d_sorted_weights(n);

  thrust::gather(d_indices.begin(), d_indices.end(), d_values.begin(),
                 d_sorted_values.begin());
  thrust::gather(d_indices.begin(), d_indices.end(), d_weights.begin(),
                 d_sorted_weights.begin());

  // --- Prefix sum of weights ---
  thrust::device_vector<float> d_prefix_weights(n);
  thrust::inclusive_scan(d_sorted_weights.begin(), d_sorted_weights.end(),
                         d_prefix_weights.begin());

  // --- Copy prefix sums back to host ---
  thrust::host_vector<float> h_prefix_weights = d_prefix_weights;
  thrust::host_vector<float> h_sorted_values = d_sorted_values;
  thrust::host_vector<float> h_sorted_weights = d_sorted_weights;

  // --- Greedy selection (host-side, full + fractional) ---
  float total_value = 0.0f;
  float total_weight = 0.0f;

  for (size_t i = 0; i < n; i++) {
    if (h_prefix_weights[i] <= capacity) {
      total_value += h_sorted_values[i];
      total_weight += h_sorted_weights[i];
    } else {
      float remaining = capacity - total_weight;
      if (remaining > 0)
        total_value += h_sorted_values[i] * (remaining / h_sorted_weights[i]);
      break;
    }
  }

  return total_value;
}
int main() {

  float (*funcPtr)(const std::vector<float> &values,
                   const std::vector<float> &weights, float capacity);
  funcPtr = fractional_knapsack_cuda;

  simple_runner(funcPtr);
  gen_runner(funcPtr, 500, 1000);
  gen_runner(funcPtr, 5000000, 600000);

  return 0;
}
