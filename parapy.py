import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
from threading import Thread
from statsmodels.tsa.stattools import adfuller

# Parameters
mu = 0.05           # Drift
kappa = 2.0         # Volatility mean-reversion speed
theta = 0.04        # Long-term variance
xi = 0.1            # Volatility of volatility
rho = -0.5          # Correlation between W1 and W2
S0 = 100.0          # Initial price
v0 = 0.04           # Initial variance
lambda_low = 1.0    # Jump intensity in calm regime (jumps/year)
lambda_high = 10.0  # Jump intensity in turbulent regime (jumps/year)
alpha = 1.0         # Transition rate: calm -> turbulent (per year)
beta = 1.0          # Transition rate: turbulent -> calm (per year)
mu_J = 0.0          # Mean of log jump size
sigma_J = 0.1       # Std dev of log jump size

# Time parameters
num_years = 2
minutes_per_year = 365 * 24 * 60
num_steps = num_years * minutes_per_year  # Total steps at 1-minute granularity
Delta_t = 1.0 / minutes_per_year          # Time step in years
num_paths = 600                        # Total number of paths
num_gpus = 3                             # Number of GPUs

# Custom CUDA kernel for Milstein update
kernel_code = """
__global__ void milstein_update(float* S, float* v, float mu, float kappa, float theta, float xi, float rho, float Delta_t, float* dW1, float* dW2, int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        float v_t = max(v[idx], 0.0f);
        float sqrt_v = sqrt(v_t);
        float v_term = kappa * (theta - v_t) * Delta_t + xi * sqrt_v * dW2[idx];
        float v_milstein = (xi * xi / 4.0f) * (dW2[idx] * dW2[idx] - Delta_t);
        v[idx] = max(v_t + v_term + v_milstein, 0.0f);
        
        float S_t = S[idx];
        float S_term = mu * S_t * Delta_t + sqrt_v * S_t * dW1[idx];
        float S_milstein = 0.5f * v_t * S_t * (dW1[idx] * dW1[idx] - Delta_t);
        S[idx] = S_t + S_term + S_milstein;
    }
}
"""
milstein_kernel = cp.RawKernel(kernel_code, 'milstein_update')

# Asynchronous save function
def save_async(data, filename, dataset_name, dtype, compression="gzip", compression_opts=9):
    def save():
        with h5py.File(filename, 'a') as f:
            f.create_dataset(dataset_name, data=data, dtype=dtype, compression=compression, compression_opts=compression_opts)
    Thread(target=save).start()

# Simulate a chunk of paths on a specific GPU
def simulate_chunk(gpu_id, num_paths_chunk, num_steps, Delta_t, mu, kappa, theta, xi, rho, S0, v0, 
                   lambda_low, lambda_high, alpha, beta, mu_J, sigma_J, results):
    with cp.cuda.Device(gpu_id):
        # Initialize arrays on GPU
        S = cp.full((num_paths_chunk, num_steps + 1), S0, dtype=cp.float32)
        v = cp.full((num_paths_chunk, num_steps + 1), v0, dtype=cp.float32)
        regime = cp.zeros((num_paths_chunk, num_steps + 1), dtype=cp.int8)  # 0: calm, 1: turbulent
        
        # Precompute transition probabilities
        transition_probs = cp.array([[1 - alpha * Delta_t, alpha * Delta_t], 
                                     [beta * Delta_t, 1 - beta * Delta_t]], dtype=cp.float32)
        transition_probs[:, 1] = 1 - transition_probs[:, 0]  # Ensure sum to 1
        
        # Lists to store jump data
        all_jump_times = []
        all_jump_sizes = []
        path_indices = []
        
        # Precompute constants
        sqrt_dt = cp.sqrt(Delta_t)
        xi2_4 = (xi**2) / 4
        
        for t in range(num_steps):
            # Generate correlated Brownian increments
            Z1 = cp.random.normal(0, 1, num_paths_chunk)
            Z2 = cp.random.normal(0, 1, num_paths_chunk)
            dW1 = Z1 * sqrt_dt
            dW2 = rho * dW1 + cp.sqrt(1 - rho**2) * Z2 * sqrt_dt
            
            # Milstein update using custom kernel
            threads_per_block = 256
            blocks = (num_paths_chunk + threads_per_block - 1) // threads_per_block
            milstein_kernel((blocks,), (threads_per_block,), 
                           (S[:, t], v[:, t], mu, kappa, theta, xi, rho, Delta_t, dW1, dW2, num_paths_chunk))
            
            # Regime-dependent jump intensity
            lambda_t = cp.where(regime[:, t] == 0, lambda_low, lambda_high)
            
            # Vectorized jump handling
            N_Delta_t = cp.random.poisson(lambda_t * Delta_t, num_paths_chunk)
            has_jumps = N_Delta_t > 0
            num_jumps = cp.sum(N_Delta_t)
            if num_jumps > 0:
                jump_sizes = cp.exp(mu_J + sigma_J * cp.random.normal(size=num_jumps))
                cumsum_N = cp.cumsum(N_Delta_t[has_jumps])
                split_points = cp.concatenate([cp.array([0]), cumsum_N])
                jump_factors = cp.ones(len(split_points) - 1, dtype=cp.float32)
                for i in range(len(split_points) - 1):
                    start, end = split_points[i], split_points[i + 1]
                    if end > start:
                        jump_factors[i] = cp.prod(jump_sizes[start:end])
                S[has_jumps, t+1] = S[has_jumps, t+1] * jump_factors
                # Store jump data
                all_jump_times.append(t * Delta_t * cp.ones(num_jumps))
                all_jump_sizes.append(jump_sizes)
                path_indices.append(cp.repeat(cp.arange(num_paths_chunk)[has_jumps], N_Delta_t[has_jumps]))
            else:
                S[:, t+1] = S[:, t]
            
            # Update regimes
            rand = cp.random.random(num_paths_chunk)
            regime[:, t+1] = cp.where(rand < transition_probs[regime[:, t], 1], 1 - regime[:, t], regime[:, t])
        
        # Concatenate jump data
        all_jump_times = cp.concatenate(all_jump_times) if all_jump_times else cp.empty(0)
        all_jump_sizes = cp.concatenate(all_jump_sizes) if all_jump_sizes else cp.empty(0)
        path_indices = cp.concatenate(path_indices) if path_indices else cp.empty(0)
        
        # Transfer to CPU
        results.append((S.get(), v.get(), regime.get(), all_jump_times.get(), all_jump_sizes.get(), path_indices.get()))

# Run simulation across multiple GPUs
def run_simulation():
    chunks = [num_paths // num_gpus] * num_gpus
    processes = []
    manager = mp.Manager()
    results = manager.list()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=simulate_chunk, args=(gpu_id, chunks[gpu_id], num_steps, Delta_t, mu, kappa, theta, xi, rho, S0, v0, 
                                                    lambda_low, lambda_high, alpha, beta, mu_J, sigma_J, results))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    # Combine results from all GPUs
    S_all = np.concatenate([res[0] for res in results], axis=0)
    v_all = np.concatenate([res[1] for res in results], axis=0)
    regime_all = np.concatenate([res[2] for res in results], axis=0)
    jump_times_all = np.concatenate([res[3] for res in results])
    jump_sizes_all = np.concatenate([res[4] for res in results])
    path_indices_all = np.concatenate([res[5] + (gpu_id * chunks[gpu_id]) for gpu_id, res in enumerate(results)])
    
    # Validate volatility stationarity
    p_value = adfuller(v_all.flatten())[1]
    print(f"Volatility Stationarity: p={p_value:.3f}")
    
    return S_all, v_all, regime_all, jump_times_all, jump_sizes_all, path_indices_all

# Main execution
if __name__ == '__main__':
    print("Starting simulation...")
    start_time = time()
    S, v, regime, jump_times, jump_sizes, path_indices = run_simulation()
    print(f"Simulation completed in {time() - start_time:.2f} seconds")
    
    # Save data asynchronously
    save_async(S, 'financial_data.h5', 'S', dtype='f2')
    save_async(v, 'financial_data.h5', 'v', dtype='f2')
    save_async(regime, 'financial_data.h5', 'regime', dtype='i1')
    save_async(jump_times, 'financial_data.h5', 'jump_times', dtype='f4')
    save_async(jump_sizes, 'financial_data.h5', 'jump_sizes', dtype='f4')
    save_async(path_indices, 'financial_data.h5', 'path_indices', dtype='i4')
    
    # Plot sample path (path 0)
    time_axis = np.arange(num_steps + 1) * Delta_t
    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, S[0], label='Price')
    plt.title('Asset Price (Path 0)')
    plt.ylabel('S_t')
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, v[0], label='Volatility', color='orange')
    plt.title('Volatility (Path 0)')
    plt.ylabel('v_t')
    plt.subplot(3, 1, 3)
    plt.step(time_axis, regime[0], label='Regime', color='green')
    plt.title('Regime (Path 0)')
    plt.xlabel('Time (years)')
    plt.ylabel('Regime (0: Calm, 1: Turbulent)')
    plt.tight_layout()
    plt.show()