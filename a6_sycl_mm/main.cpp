#include "hipSYCL/sycl/info/device.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr size_t Width = 1024;
constexpr size_t Height = Width;

#include <CL/sycl.hpp>
#include <iostream>

void query_mem(queue &q) {
  sycl::device dev = q.get_device();

  auto global_mem = dev.get_info<sycl::info::device::global_mem_size>();
  auto local_mem = dev.get_info<sycl::info::device::local_mem_size>();
  auto max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();

  std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
  std::cout << "Global memory: " << global_mem / (1024 * 1024) << " MB\n";
  std::cout << "Local memory: " << local_mem / 1024 << " KB\n";
  std::cout << "Max alloc size: " << max_alloc / (1024 * 1024) << " MB\n";
}

size_t get_safe_device_mem(const sycl::device &dev,
                           float safety_factor = 0.8f) {
  size_t total_mem = dev.get_info<sycl::info::device::local_mem_size>();
  return static_cast<size_t>(total_mem * safety_factor);
}

void compute_tile_dims(size_t safe_mem_bytes, size_t Width, size_t &tile_h,
                       size_t &tile_w) {
  size_t max_tile_area =
      safe_mem_bytes / (3 * sizeof(float)); // 3 buffers: A_tile, B_tile, C_tile

  // Simple approach: square tiles
  size_t tile_dim = static_cast<size_t>(std::sqrt(max_tile_area));
  tile_h = std::min(tile_dim, Width);
  tile_w = tile_h;
}

void mm(float *fa, float *fb, float *fc) {
  int row, col, k;
  float Pvalue = 0;
  for (row = 0; row < Height; row++) {
    for (col = 0; col < Width; col++) {
      Pvalue = 0;
      for (k = 0; k < Width; k++) {
        Pvalue += fa[row * Width + k] * fb[k * Width + col];
      }
      fc[row * Width + col] = Pvalue;
    }
  }
}

void mm_sycl(queue &q, float *fa, float *fb, float *fc) {
  // Wrap host arrays in buffers to pass to the device
  buffer<float, 1> buf_a(fa, range<1>(Width * Height));
  buffer<float, 1> buf_b(fb, range<1>(Width * Height));
  buffer<float, 1> buf_c(fc, range<1>(Width * Height));

  // run lambda fn as kernel in device q
  q.submit([&](handler &h) {
    // Accessors for the passed memory
    auto a = buf_a.get_access<access::mode::read>(h);
    auto b = buf_b.get_access<access::mode::read>(h);
    auto c = buf_c.get_access<access::mode::write>(h);

    // Parallelize outer loop (row)
    h.parallel_for(range<1>(Height), [=](id<1> row_id) {
      int row = row_id[0];
      for (int col = 0; col < Width; col++) {
        float Pvalue = 0;
        for (int k = 0; k < Width; k++) {
          Pvalue += a[row * Width + k] * b[k * Width + col];
        }
        c[row * Width + col] = Pvalue;
      }
    });
    // buffer destructors handle host copy-back
  });
}

// note the lack of the buffer code block structure
void mm_sycl_usm(queue &q, float *fa, float *fb, float *fc, int NW, int NH) {
  // Allocate device memory for matrices
  float *d_a = malloc_device<float>(NW * NH, q);
  float *d_b = malloc_device<float>(NW * NH, q);
  float *d_c = malloc_device<float>(NW * NH, q);

  // Copy host data to device
  q.memcpy(d_a, fa, sizeof(float) * NW * NH).wait();
  q.memcpy(d_b, fb, sizeof(float) * NW * NH).wait();

  // Launch kernel: parallelize outer loop (row)
  q.parallel_for(range<1>(NH), [=](id<1> row_id) {
     int row = row_id[0];
     for (int col = 0; col < NW; col++) {
       float Pvalue = 0;
       for (int k = 0; k < NW; k++) {
         Pvalue += d_a[row * NW + k] * d_b[k * NW + col];
       }
       d_c[row * NW + col] = Pvalue;
     }
   }).wait(); // wait for completion

  // Copy results back to host
  q.memcpy(fc, d_c, sizeof(float) * NW * NH).wait();

  // Free device memory
  free(d_a, q);
  free(d_b, q);
  free(d_c, q);
}

void mm_sycl_tiled(queue &q, float *A, float *B, float *C, int N) {
  int max_tile = 512; // or based on available memory

  if (N <= max_tile) {
    mm_sycl_usm(q, A, B, C, N, N);
    return;
  }

  int half = N / 2;
  // assume row-major layout

  // pointers to sub-blocks of A, B, and C
  float *A11 = A;
  float *A12 = A + half;
  float *A21 = A + half * N;
  float *A22 = A + half * N + half;

  float *B11 = B;
  float *B12 = B + half;
  float *B21 = B + half * N;
  float *B22 = B + half * N + half;

  float *C11 = C;
  float *C12 = C + half;
  float *C21 = C + half * N;
  float *C22 = C + half * N + half;

  // Recursively compute:
  // C11 = A11*B11 + A12*B21
  mm_sycl_tiled(q, A11, B11, C11, half);
  mm_sycl_tiled(q, A12, B21, C11, half);

  // C12 = A11*B12 + A12*B22
  mm_sycl_tiled(q, A11, B12, C12, half);
  mm_sycl_tiled(q, A12, B22, C12, half);

  // C21 = A21*B11 + A22*B21
  mm_sycl_tiled(q, A21, B11, C21, half);
  mm_sycl_tiled(q, A22, B21, C21, half);

  // C22 = A21*B12 + A22*B22
  mm_sycl_tiled(q, A21, B12, C22, half);
  mm_sycl_tiled(q, A22, B22, C22, half);
}

inline auto time() { return std::chrono::high_resolution_clock::now(); }

int main() {
  queue q;
  query_mem(q);

  // Randomize input
  std::vector<float> A(Width * Height), B(Width * Height);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &x : A)
    x = dist(gen);
  for (auto &x : B)
    x = dist(gen);

  std::vector<float> C_cpu(Width * Height);
  // std::vector<float> C_sycl(Width * Height);
  // std::vector<float> C_usm(Width * Height);
  std::vector<float> C_tiled(Width * Height);

  // tiled
  auto t7 = time();
  mm_sycl_tiled(q, A.data(), B.data(), C_tiled.data(), Height);
  q.wait();
  auto t8 = time();
  std::cout << "SYCL tiled time: "
            << std::chrono::duration<double, std::milli>(t8 - t7).count()
            << " ms\n";

  // if (Width < 4000) {
  //   // SYCL buffer
  //   auto t3 = time();
  //   mm_sycl(q, A.data(), B.data(), C_sycl.data());
  //   q.wait();
  //   auto t4 = time();
  //   std::cout << "SYCL buffer mm time: "
  //             << std::chrono::duration<double, std::milli>(t4 - t3).count()
  //             << " ms\n";

  //   // SYCL USM
  //   auto t5 = time();
  //   mm_sycl_usm(q, A.data(), B.data(), C_usm.data());
  //   q.wait();
  //   auto t6 = time();
  //   std::cout << "SYCL USM mm time: "
  //             << std::chrono::duration<double, std::milli>(t6 - t5).count()
  //             << " ms\n";
  // }

  // CPU
  auto t1 = time();
  mm(A.data(), B.data(), C_cpu.data());
  auto t2 = time();
  std::cout << "CPU mm time: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << " ms\n";

  // Verify results
  auto check = [&](const std::vector<float> &v, const char *name) {
    for (size_t i = 0; i < Width * Height; i++) {
      if (std::fabs(C_cpu[i] - v[i]) > 1e-4f) {
        std::cout << name << " mismatch at index " << i << ": CPU=" << C_cpu[i]
                  << ", " << name << "=" << v[i] << "\n";
        return false;
      }
    }
    return true;
  };

  // std::cout << "SYCL buffer matches CPU? "
  //           << (check(C_sycl, "SYCL buffer") ? "YES" : "NO") << "\n";
  // std::cout << "SYCL USM matches CPU? "
  //           << (check(C_usm, "SYCL USM") ? "YES" : "NO") << "\n";
  std::cout << "tiled matches CPU? "
            << (check(C_tiled, "SYCL tiled") ? "YES" : "NO") << "\n";
  return 0;
}
