#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/info/device.hpp"
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

inline auto time() { return std::chrono::high_resolution_clock::now(); }
constexpr int TILE_SIZE = 16; // tile height/width
constexpr size_t Width = 10000;
constexpr size_t Height = Width;
constexpr size_t W = Width;
constexpr size_t H = Width;
constexpr size_t KW = 13;
constexpr size_t KH = KW;
constexpr int outH = H - KH + 1;
constexpr int outW = W - KW + 1;

void conv2d_sycl_split(sycl::queue &q, float *a, float *b, float *fc) {
  constexpr int TS = TILE_SIZE;
  const int tileA_h = TS + KH - 1;
  const int tileA_w = TS + KW - 1;

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 2> tileA(
        {static_cast<size_t>(tileA_h), static_cast<size_t>(tileA_w)}, h);
    sycl::local_accessor<float, 2> tileB(
        {static_cast<size_t>(KH), static_cast<size_t>(KW)}, h);

    h.parallel_for(sycl::nd_range<2>{{(size_t)outH, (size_t)outW}, {TS, TS}},
                   [=](sycl::nd_item<2> it) {
                     int lr = it.get_local_id(0);
                     int lc = it.get_local_id(1);

                     int gr = it.get_global_id(0);
                     int gc = it.get_global_id(1);

                     // Ignore padded threads outside real output dims

                     /// load Tile B (kernel size < thread count)
                     if (lr < KH && lc < KW)
                       tileB[lr][lc] = b[lr * KW + lc];

                     /// load Tile A (size > thread count)

                     // TODO:flatten?
                     for (int rr = lr; rr < tileA_h; rr += TS) {
                       for (int cc = lc; cc < tileA_w; cc += TS) {
                         // pos of top left in WG + thread's offset
                         int in_r = it.get_group(0) * TS + rr;
                         int in_c = it.get_group(1) * TS + cc;

                         if (in_r < H && in_c < W)
                           tileA[rr][cc] = a[in_r * W + in_c];
                         else
                           tileA[rr][cc] =
                               0.0f; // zero padding for out-of-bounds
                       }
                     }

                     it.barrier(sycl::access::fence_space::local_space);

                     // Ignore padded threads outside real output dims
                     if (!(gr >= outH || gc >= outW)) {

                       // ---- Compute convolution from shared memory ----

                       float accum = 0.0f;
                       for (int kr = 0; kr < KH; kr++) {
                         for (int kc = 0; kc < KW; kc++) {
                           float val = tileA[lr + kr][lc + kc];
                           float k = tileB[kr][kc];
                           accum += val * k;
                         }
                       }

                       // ---- Write output ----
                       fc[gr * outW + gc] = accum;
                     }
                   });
  });

  q.wait();
}
void conv2d_sycl_tiled(sycl::queue &q, float *a, float *b, float *fc) {
  constexpr int TS = TILE_SIZE;
  const int tileA_h = TS + KH - 1;
  const int tileA_w = TS + KW - 1;

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 2> tileA(
        {static_cast<size_t>(tileA_h), static_cast<size_t>(tileA_w)}, h);
    sycl::local_accessor<float, 2> tileB(
        {static_cast<size_t>(KH), static_cast<size_t>(KW)}, h);

    h.parallel_for(sycl::nd_range<2>{{(size_t)outH, (size_t)outW}, {TS, TS}},
                   [=](sycl::nd_item<2> it) {
                     int lr = it.get_local_id(0);
                     int lc = it.get_local_id(1);

                     int gr = it.get_global_id(0);
                     int gc = it.get_global_id(1);

                     // Ignore padded threads outside real output dims

                     /// load Tile B (kernel size < thread count)
                     if (lr < KH && lc < KW)
                       tileB[lr][lc] = b[lr * KW + lc];

                     /// load Tile A (size > thread count)

                     // TODO:flatten?
                     for (int rr = lr; rr < tileA_h; rr += TS) {
                       for (int cc = lc; cc < tileA_w; cc += TS) {
                         // pos of top left in WG + thread's offset
                         int in_r = it.get_group(0) * TS + rr;
                         int in_c = it.get_group(1) * TS + cc;

                         if (in_r < H && in_c < W)
                           tileA[rr][cc] = a[in_r * W + in_c];
                         else
                           tileA[rr][cc] =
                               0.0f; // zero padding for out-of-bounds
                       }
                     }

                     it.barrier(sycl::access::fence_space::local_space);

                     // Ignore padded threads outside real output dims
                     float accum = 0.0f;
                     if (!(gr >= outH || gc >= outW)) {

                       // ---- Compute convolution from shared memory ----

                       for (int kr = 0; kr < KH; kr++) {
                         for (int kc = 0; kc < KW; kc++) {
                           float val = tileA[lr + kr][lc + kc];
                           float k = tileB[kr][kc];
                           accum += val * k;
                         }
                       }

                       // ---- Write output ----
                     }

                     it.barrier(sycl::access::fence_space::local_space);

                     if (!(gr >= outH || gc >= outW)) {
                       fc[gr * outW + gc] = accum;
                     }
                   });
  });

  q.wait();
}
void conv2d_seq_baseline(float *a, float *b, float *fc) {
  for (int r = 0; r < outH; r++) {
    for (int c = 0; c < outW; c++) {

      float accum = 0.0f;

      for (int kr = 0; kr < KH; kr++) {
        for (int kc = 0; kc < KW; kc++) {

          float val = a[(r + kr) * W + (c + kc)];
          float k = b[kr * KW + kc];

          accum += val * k;
        }
      }

      fc[r * outW + c] = accum;
    }
  }
}

int main() {
  queue q;

  // Randomize input
  float *A = malloc_host<float>(Width * Height, q);
  float *B = malloc_host<float>(KW * KH, q);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < Width * Height; i++) {
    A[i] = dist(gen);
  }
  for (size_t i = 0; i < KW * KH; i++) {
    B[i] = dist(gen);
  }

  float *C_tiled = malloc_host<float>(outH * outW, q);
  float *C_seq = malloc_host<float>(outH * outW, q);
  memset(C_tiled, 0, outH * outW * sizeof(float));
  memset(C_seq, 0, outH * outW * sizeof(float));

  // tiled
  auto t7 = time();
  conv2d_sycl_split(q, A, B, C_tiled);
  q.wait();
  auto t8 = time();
  std::cout << "SYCL tiled time: "
            << std::chrono::duration<double, std::milli>(t8 - t7).count()
            << " ms\n";

  // CPU
  auto t1 = time();
  conv2d_seq_baseline(A, B, C_seq);
  auto t2 = time();
  std::cout << "CPU conv time: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << " ms\n";

  auto check_ptr = [&](const float *v, const char *name) {
    int f = 0;
    for (size_t i = 0; i < outH * outW; i++) {
      if (std::fabs(C_seq[i] - v[i]) > 1e-4f) {
        // std::cout << name << " mismatch at index " << i << ": CPU=" <<
        // C_cpu[i]
        //           << ", " << name << "=" << v[i] << "\n";
        f++;
      }
    }
    printf("%d/%d failuers\n", f, outH * outW);

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
