// bandwidth_probe.cu - measure achievable HBM bandwidth on a full GPU or a MIG slice.
// Why: LLM decode reads every weight once per generated token, so tokens/s per sequence is capped by
// memory bandwidth. This probe turns that into a number you can put next to your ITL measurements.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -o bandwidth_probe bandwidth_probe.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

// Copy: 1 read + 1 write per element (classic STREAM-style).
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

// Read-only stream: models decode, which streams weights/KV and writes almost nothing.
__global__ void read_kernel(const float4* __restrict__ src, float* __restrict__ sink, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float acc = 0.f;
  for (; i < n; i += stride) {
    const float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == -1.2345f) sink[0] = acc;  // practically never true; stops the compiler removing the loads
}

__global__ void init_kernel(float4* p, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) p[i] = make_float4(1.f, 2.f, 3.f, 4.f);
}

static float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 30;
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));

  // Two buffers of ~35% of free memory each (capped at 8 GiB): far larger than L2, so we measure HBM.
  size_t bytes = std::min<size_t>(free_b * 35 / 100, static_cast<size_t>(8) << 30);
  bytes -= bytes % (sizeof(float4) * 1024);
  const size_t n = bytes / sizeof(float4);

  float4 *a = nullptr, *b = nullptr;
  float* sink = nullptr;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMalloc(&sink, sizeof(float)));

  const int threads = 256;
  const int blocks = prop.multiProcessorCount * 16;
  init_kernel<<<blocks, threads>>>(a, n);
  init_kernel<<<blocks, threads>>>(b, n);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  std::vector<float> copy_ms, read_ms;
  for (int it = 0; it < iters + 3; ++it) {  // first 3 iterations are warm-up
    float ms = 0.f;
    CK(cudaEventRecord(t0));
    copy_kernel<<<blocks, threads>>>(a, b, n);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    if (it >= 3) copy_ms.push_back(ms);

    CK(cudaEventRecord(t0));
    read_kernel<<<blocks, threads>>>(a, sink, n);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    if (it >= 3) read_ms.push_back(ms);
  }
  CK(cudaGetLastError());

  const double copy_gbps = (2.0 * bytes) / (median(copy_ms) * 1e-3) / 1e9;
  const double read_gbps = (1.0 * bytes) / (median(read_ms) * 1e-3) / 1e9;
  const double llama8b_bf16_bytes = 16.06e9;  // Llama-3.1-8B weights in BF16
  std::printf("{\"device\":\"%s\",\"sm_count\":%d,\"total_mem_gib\":%.1f,\"buffer_gib\":%.2f,"
              "\"copy_gbps\":%.0f,\"read_gbps\":%.0f,\"iters\":%d}\n",
              prop.name, prop.multiProcessorCount, total_b / 1073741824.0, bytes / 1073741824.0,
              copy_gbps, read_gbps, iters);
  std::printf("Decode floor for Llama-3.1-8B BF16 at batch 1: >= %.1f ms/token (~%.0f tok/s per sequence)\n",
              llama8b_bf16_bytes / (read_gbps * 1e9) * 1e3, read_gbps * 1e9 / llama8b_bf16_bytes);
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(sink));
  return 0;
}
