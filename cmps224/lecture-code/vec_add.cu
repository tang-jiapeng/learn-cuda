#include "time.h"
#include <cstdlib>

__host__ __device__ float add(float a, float b ) {
  return a + b;
}

void vecadd_cpu(float *x, float *y, float *z, int N) {
  Timer timer;
  startTime(&timer);
  for (int i = 0; i < N; ++i) {
    z[i] = add(x[i], y[i]);
  }
  stopTime(&timer);
  printElapsedTime(timer, "CPU vecadd function time", GREEN);
}

__global__ void vecadd_kernel(float *x, float *y, float *z, int N) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < N) {
    z[i] = add(x[i], y[i]);
  }
}

void vecadd_gpu(float *x, float *y, float *z, int N) {
  // Allocate GPU memory
  float *x_d, *y_d, *z_d;
  cudaMalloc((void **)&x_d, N * sizeof(float));
  cudaMalloc((void **)&y_d, N * sizeof(float));
  cudaMalloc((void **)&z_d, N * sizeof(float));

  // Copy to the GPU
  cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Call a GPU kernel function (lanuch a grid of threads)
  Timer timer;
  startTime(&timer);
  const unsigned int numThreadsPerBlock = 512;
  const unsigned int numBlock = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
  vecadd_kernel<<<numBlock, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
  stopTime(&timer);
  printElapsedTime(timer, "GPU vecadd function time", GREEN);
  
  // Copy from the GPU
  cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Deallocate GPU memory
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
}

int main(int argc, char **argv) {
  cudaDeviceSynchronize();

  Timer timer, timer1;

  int N = (argc > 1) ? (atoi(argv[1])) : (1 << 25);
  float *x = (float *)malloc(N * sizeof(float));
  float *y = (float *)malloc(N * sizeof(float));
  float *z = (float *)malloc(N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    x[i] = rand();
    y[i] = rand();
  }

  startTime(&timer);
  vecadd_cpu(x, y, z, N);
  stopTime(&timer);
  printElapsedTime(timer, "CPU time", CYAN);

  startTime(&timer1);
  vecadd_gpu(x, y, z, N);
  stopTime(&timer1);
  printElapsedTime(timer1, "GPU time", CYAN);

  free(x);
  free(y);
  free(z);
}