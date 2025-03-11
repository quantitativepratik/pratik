#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#define IDX(i, j, n) ((i) * (n) + (j))

// Custom atomicAdd for double
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// Kernel for stencil calculation
__global__ void stencilUpdate(double* A, double* A_old, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1) {
        double neighbors[4] = {
            A_old[IDX(i + 1, j + 1, n)],
            A_old[IDX(i + 1, j - 1, n)],
            A_old[IDX(i - 1, j + 1, n)],
            A_old[IDX(i - 1, j - 1, n)]
        };

        // Find the second smallest value in the neighbors
        double min1 = fmin(neighbors[0], neighbors[1]);
        double min2 = fmax(neighbors[0], neighbors[1]);

        for (int k = 2; k < 4; ++k) {
            if (neighbors[k] < min1) {
                min2 = min1;
                min1 = neighbors[k];
            } else if (neighbors[k] < min2) {
                min2 = neighbors[k];
            }
        }

        // Update the current element
        A[IDX(i, j, n)] = A_old[IDX(i, j, n)] + min2;
    }
}

// Kernel for computing verification values
__global__ void computeVerification(const double* A, int n, double* sum, double* value) {
    __shared__ double blockSum[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;

    blockSum[threadId] = (idx < n * n) ? A[idx] : 0.0;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            blockSum[threadId] += blockSum[threadId + stride];
        }
        __syncthreads();
    }

    // Write the block's sum to global memory
    if (threadId == 0) {
        atomicAddDouble(sum, blockSum[0]); // Use custom atomic add
    }

    // Fetch A(37, 47) (only once)
    if (idx == IDX(37, 47, n)) {
        *value = A[idx];
    }
}

// Host function
int main() {
    int n = 2000; // You can modify this for different runs (e.g., 500, 1000, 2000)
    int t = 10;

    size_t size = n * n * sizeof(double);

    // Allocate memory on CPU
    double* h_A = new double[n * n];
    double h_sum = 0.0, h_value = 0.0;

    // Initialize A on the CPU
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_A[IDX(i, j, n)] = pow(1 + cos(2 * i) + sin(j), 2);
        }
    }

    // Allocate memory on GPU
    double *d_A, *d_A_old, *d_sum, *d_value;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_A_old, size);
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_value, sizeof(double));

    // Copy initial matrix to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start CUDA timer
    cudaEventRecord(start);

    // Perform t iterations
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    for (int iter = 0; iter < t; ++iter) {
        cudaMemcpy(d_A_old, d_A, size, cudaMemcpyDeviceToDevice);
        stencilUpdate<<<gridDim, blockDim>>>(d_A, d_A_old, n);
        cudaDeviceSynchronize();
    }

    // Reset sum and compute verification values
    cudaMemset(d_sum, 0, sizeof(double));
    computeVerification<<<(n * n + 255) / 256, 256>>>(d_A, n, d_sum, d_value);

    // Stop CUDA timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy results back to CPU
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_value, d_value, sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print results
    std::cout << "Elapsed time: " << elapsedTime / 1000.0f << " seconds\n";
    std::cout << "Sum of all elements: " << h_sum << "\n";
    std::cout << "A(37, 47): " << h_value << "\n";

    // Cleanup
    delete[] h_A;
    cudaFree(d_A);
    cudaFree(d_A_old);
    cudaFree(d_sum);
    cudaFree(d_value);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

