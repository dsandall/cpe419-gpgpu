#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/info/device.hpp"
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int TILE_SIZE = 16; // tile height/width
constexpr size_t Width = 1024;
constexpr size_t Height = Width;

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
void mm_sycl_usm(queue &q, float *fa, float *fb, float *fc, int W, int H) {
  // Allocate device memory for matrices
  float *d_a = malloc_device<float>(W * H, q);
  float *d_b = malloc_device<float>(W * H, q);
  float *d_c = malloc_device<float>(W * H, q);

  // Copy host data to device
  q.memcpy(d_a, fa, sizeof(float) * W * H).wait();
  q.memcpy(d_b, fb, sizeof(float) * W * H).wait();

  // Launch kernel: parallelize outer loop (row)
  q.parallel_for(range<2>(H, W), [=](id<2> row_id) {
     int row = row_id[0];
     int col = row_id[1];
     float Pvalue = 0;
     for (int k = 0; k < W; k++) {
       Pvalue += d_a[row * W + k] * d_b[k * W + col];
     }
     d_c[row * W + col] = Pvalue;
   }).wait(); // wait for completion

  // Copy results back to host
  q.memcpy(fc, d_c, sizeof(float) * W * H).wait();

  // Free device memory
  free(d_a, q);
  free(d_b, q);
  free(d_c, q);
}

void mm_sycl_tiled(sycl::queue &q, float *a, float *b, float *fc, int N) {
  constexpr int TS = TILE_SIZE;

  q.submit([&](sycl::handler &h) {
    // allocate shared mem for tiles
    sycl::local_accessor<float, 2> tileA({TS, TS}, h); // shared mem
    sycl::local_accessor<float, 2> tileB({TS, TS}, h); // shared mem

    // a thread for every output
    h.parallel_for(sycl::nd_range<2>{{(size_t)N, (size_t)N}, {TS, TS}},
                   [=](sycl::nd_item<2> wg) {
                     int lr = wg.get_local_id(0);
                     int lc = wg.get_local_id(1);

                     const int gr = wg.get_global_id(0); // global row
                     const int gc = wg.get_global_id(1); // global col

                     float accum = 0.0f;
                     bool output = gr < N && gc < N;

                     // for each tile along the matrix
                     for (int k0 = 0; k0 < N; k0 += TS) {
                       // load
                       tileA[lr][lc] = a[gr * N + (k0 + lc)];
                       tileB[lr][lc] = b[(k0 + lr) * N + gc];
                       wg.barrier(sycl::access::fence_space::local_space);

                       // iterate across row/col of the tile and accum
                       if (output) {
                         for (int k = 0; k < TS; k++) {
                           if (k0 + k < N) {
                             // float va = a[gr * N + (k0 + k)];
                             // float vb = b[(k0 + k) * N + gc];
                             accum += tileA[lr][k] * tileB[k][lc];
                           }
                         }
                       }
                     }

                     // if the output is needed,
                     if (output)
                       fc[gr * N + gc] = accum;
                   });
  });

  // synch
  q.wait();
}

inline auto time() { return std::chrono::high_resolution_clock::now(); }

int main() {
  queue q;

  // Randomize input
  float *A = malloc_host<float>(Width * Height, q);
  float *B = malloc_host<float>(Width * Height, q);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < Width * Height; i++) {
    A[i] = dist(gen);
    B[i] = dist(gen);
  }

  std::vector<float> C_cpu(Width * Height);
  float *C_tiled = malloc_host<float>(Width * Height, q);
  float *C_sycl = malloc_host<float>(Width * Height, q);
  float *C_usm = malloc_host<float>(Width * Height, q);
  memset(C_tiled, 0, Width * Height * sizeof(float));

  // tiled
  auto t7 = time();
  mm_sycl_tiled(q, A, B, C_tiled, Height);
  q.wait();
  auto t8 = time();
  std::cout << "SYCL tiled time: "
            << std::chrono::duration<double, std::milli>(t8 - t7).count()
            << " ms\n";

  if (Width < 1000) {
    // SYCL buffer
    auto t3 = time();
    mm_sycl(q, A, B, C_sycl);
    q.wait();
    auto t4 = time();
    std::cout << "SYCL buffer mm time: "
              << std::chrono::duration<double, std::milli>(t4 - t3).count()
              << " ms\n";

    // SYCL USM
    auto t5 = time();
    mm_sycl_usm(q, A, B, C_usm, Width, Height);
    q.wait();
    auto t6 = time();
    std::cout << "SYCL USM mm time: "
              << std::chrono::duration<double, std::milli>(t6 - t5).count()
              << " ms\n";
  }

  // CPU
  auto t1 = time();
  mm(A, B, C_cpu.data());
  auto t2 = time();
  std::cout << "CPU mm time: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << " ms\n";

  auto check_ptr = [&](const float *v, const char *name) {
    int f = 0;
    for (size_t i = 0; i < Width * Height; i++) {
      if (std::fabs(C_cpu[i] - v[i]) > 1e-4f) {
        // std::cout << name << " mismatch at index " << i << ": CPU=" <<
        // C_cpu[i]
        //           << ", " << name << "=" << v[i] << "\n";
        f++;
      }
    }
    printf("%d/%zu failuers\n", f, Width * Height);

    return true;
  };

#ifdef DEBUG
  for (int i = 0; i < Width * Height; i++) {
    std::cout << "CPU " << C_cpu[i] << " " << C_tiled[i] << "\n";
  }
#endif // DEBUG

  std::cout << "tiled matches CPU? "
            << (check_ptr(C_tiled, "SYCL tiled") ? "YES" : "NO") << "\n";
  return 0;
}
