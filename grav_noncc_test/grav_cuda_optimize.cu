#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


#include "Body.h"
/**
 * 
 * This file is to generate a output file for the solar system positions after calcualtion and iteration for certain time
 * 
 */

constexpr float G = 5.67e-11f;       
constexpr float POSITION_CORRECTION = 1e-9f;   

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t __err = (expr);                                                  \
        if (__err != cudaSuccess) {                                                  \
            std::cerr << "CUDA error (" << #expr << "): "                           \
                      << cudaGetErrorString(__err) << '\n';                         \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

// kernels
__global__ void update_velocities_tiled(
    int n,
    const float* __restrict__ m,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    float dt) {
    extern __shared__ float4 tile[];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const bool active = i < n;

    float my_x = 0.0f;
    float my_y = 0.0f;
    float my_z = 0.0f;
    if (active) {
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
    }

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        const int j = tile_start + tid;
        if (j < n) {
            tile[tid] = make_float4(x[j], y[j], z[j], G * m[j]);
        } else {
            tile[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();
        const int tile_count = min(blockDim.x, n - tile_start);

        if (active) {
            #pragma unroll 4
            for (int k = 0; k < tile_count; ++k) {
                if (tile_start + k == i) continue;

                const float4 body = tile[k];
                const float dx = body.x - my_x;
                const float dy = body.y - my_y;
                const float dz = body.z - my_z;
                const float dist_sq = dx * dx + dy * dy + dz * dz + POSITION_CORRECTION;
                const float inv_r = rsqrtf(dist_sq);
                const float factor = body.w * inv_r * inv_r * inv_r;

                ax = fmaf(factor, dx, ax);
                ay = fmaf(factor, dy, ay);
                az = fmaf(factor, dz, az);
            }
        }

        __syncthreads();
    }

    if (active) {
        vx[i] += ax * dt;
        vy[i] += ay * dt;
        vz[i] += az * dt;
    }
}

__global__ void update_positions(int n, float* x, float* y, float* z,
                                 const float* vx, const float* vy, const float* vz, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

//main function
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " [dt] [END] [print_interval]\n";
        return 1;
    }
    constexpr float YEAR = 365.25f * 24.0f * 60.0f * 60.0f ; // matches gravsim0.cpp
    //constexpr float MONTH = 30.0f * 24.0f * 60.0f * 60.0f;

    const float dt = argc > 1 ? static_cast<float>(atof(argv[1])) : 10.0f; // timestep in seconds
    const float END = argc > 2 ? static_cast<float>(atof(argv[2])) * YEAR : YEAR;
    const float print_interval = argc > 3 ? static_cast<float>(atof(argv[3])) : 100.0f; // seconds between GPU snapshots


 std::string input_filename = "planet_bodies_100k.dat";

    if (!std::isfinite(dt) || dt <= 0.0f ||
        !std::isfinite(END) || END < 0.0f ||
        !std::isfinite(print_interval) || print_interval <= 0.0f) {
        std::cerr << "Error: dt and print_interval must be positive and finite; END must be nonnegative and finite.\n";
        return 1;
    }

    // read from input
    std::ifstream infile(input_filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file '" << input_filename << "'\n";
        return 1;
    }

    int n = 0;
    std::string line;
    
    // Host Vectors (Structure of Arrays)
    std::vector<std::string> h_names;
    std::vector<float> h_m, h_x, h_y, h_z, h_vx, h_vy, h_vz;
    
    std::cout << "Reading from " << input_filename << "...\n";

    while (infile.peek() != EOF) {
        // Skip 't=...' headers if present (e.g., from a restart file)
        if (infile.peek() == 't') {
            std::getline(infile, line);
            continue;
        }

        std::string name;
        float mass, x, y, z, vx, vy, vz;
        // name, mass, x, y, z, vx, vy, vz
        if (infile >> name >> mass >> x >> y >> z >> vx >> vy >> vz) {
            h_names.push_back(name);
            h_m.push_back(mass);
            h_x.push_back(x);
            h_y.push_back(y);
            h_z.push_back(z);
            h_vx.push_back(vx);
            h_vy.push_back(vy);
            h_vz.push_back(vz);
        } else {
            // Skip malformed lines or whitespace
            infile.clear(); 
            std::getline(infile, line);
        }
    }
    infile.close();

    n = h_names.size();
    if (n == 0) {
        std::cerr << "Error: No bodies found in file.\n";
        return 1;
    }
    std::cout << "Loaded " << n << " bodies.\n";

    // allocate memories on device
    float *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    // Target 75 decimal GB across all ten device buffers. Store whole frames
    // in the history buffers and wrap when the simulation exceeds capacity.
    constexpr size_t target_device_bytes = 75'000'000'000ULL;
    const size_t state_bytes = 7 * bytes;
    if (state_bytes >= target_device_bytes ||
        (target_device_bytes - state_bytes) / (3 * bytes) == 0) {
        std::cerr << "Error: Body state and one history frame exceed the 75 GB budget.\n";
        return 1;
    }
    const size_t total_frames = (target_device_bytes - state_bytes) / (3 * bytes);
    const size_t history_bytes = total_frames * bytes;
    const size_t allocated_bytes = state_bytes + 3 * history_bytes;
    size_t free_device_bytes = 0;
    size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_device_bytes, &total_device_bytes));
    if (allocated_bytes > free_device_bytes) {
        std::cerr << "Error: 75 GB target requires " << allocated_bytes
                  << " bytes, but only " << free_device_bytes << " GPU bytes are free.\n";
        return 1;
    }
    std::cout << "Allocating " << allocated_bytes / 1e9
              << " GB of GPU buffers (" << total_frames << " history frames).\n";
    
    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));

    CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice));

    // Allocate bounded position history on the GPU.
    std::cerr << "dt=" << dt << "\tEND=" << (END / YEAR) << " years\tprint="
              << print_interval << '\n';
          
    float *d_hist_x, *d_hist_y, *d_hist_z;
    CUDA_CHECK(cudaMalloc(&d_hist_x, history_bytes));
    CUDA_CHECK(cudaMalloc(&d_hist_y, history_bytes));
    CUDA_CHECK(cudaMalloc(&d_hist_z, history_bytes));
    CUDA_CHECK(cudaMemGetInfo(&free_device_bytes, &total_device_bytes));
    std::cout << "GPU memory: "
              << (total_device_bytes - free_device_bytes) / (1024.0 * 1024.0)
              << " MB used, " << free_device_bytes / (1024.0 * 1024.0)
              << " MB free of " << total_device_bytes / (1024.0 * 1024.0)
              << " MB\n";
    std::cout << "Done.\n";

    // Setting up kernal parameters
    int threads = 256;
    int blocks = (n+threads-1)/threads;
    size_t frame_count = 0;
    float t = 0.0f;
    std::cerr<< "Using " << blocks << " blocks of " << threads << " threads\n";    
    /* Output file writing is disabled; retained for later use.
    std::string output_filename = "output_cuda_optimized_100k.txt";
    std::ofstream outfile(output_filename);
    outfile << std::scientific << std::setprecision(5);
    */

    cudaEvent_t gpu_start, gpu_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    CUDA_CHECK(cudaEventRecord(gpu_start));

    while (t < END) {
        // Copy current d_x to d_hist_x
        size_t offset = (frame_count % total_frames) * static_cast<size_t>(n);
        
        
        CUDA_CHECK(cudaMemcpy(d_hist_x + offset, d_x, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_hist_y + offset, d_y, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_hist_z + offset, d_z, bytes, cudaMemcpyDeviceToDevice));
        
        frame_count++; // Older snapshots are overwritten once history is full.

        //update velocity and position, step foward based on time interval
        float next_print = t + print_interval;
        while (t < next_print && t < END) {
            float step_dt = std::min(dt, END - t);
            
            update_velocities_tiled<<<blocks, threads, threads * sizeof(float4)>>>(
                n, d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            update_positions<<<blocks, threads>>>(n, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            
            t += step_dt;
        }
    }

    // last frame needed
    size_t offset = (frame_count % total_frames) * static_cast<size_t>(n);
    CUDA_CHECK(cudaMemcpy(d_hist_x + offset, d_x, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_hist_y + offset, d_y, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_hist_z + offset, d_z, bytes, cudaMemcpyDeviceToDevice));
    frame_count++;

    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed_ms, gpu_start, gpu_stop));

    /* CPU transfer and TXT output are disabled; retained for later use.
    // transfer all data back to CPU
    std::cout << "Simulation complete. Transferring " << frame_count << " frames to CPU...\n";
    
    std::vector<float> h_hist_x(total_frames * n);
    std::vector<float> h_hist_y(total_frames * n);
    std::vector<float> h_hist_z(total_frames * n);

    CUDA_CHECK(cudaMemcpy(h_hist_x.data(), d_hist_x, frame_count * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_y.data(), d_hist_y, frame_count * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_z.data(), d_hist_z, frame_count * n * sizeof(float), cudaMemcpyDeviceToHost));

    // write to a file
    std::cout << "Writing to disk...\n";
    std::cerr << "Writing output to " << output_filename << '\n';
    std::cerr << "Total bodies framed: " << h_names.size() << '\n';
    float write_t = 0.0f;
    
    for (int f = 0; f < frame_count; f++) {
        outfile << "t=:" << write_t << "\n";
        //don't need vx,vy,vz for output,only the relative positions and names
        for (int i = 0; i < n; i++) {
            size_t idx = f * n + i;
            outfile << h_names[i] << " "
                    << h_hist_x[idx] << " " 
                    << h_hist_y[idx] << " " 
                    << h_hist_z[idx] << "\n";
            
        }
        write_t += print_interval;
    }

    outfile.close();
    */

    std::cout << "GPU elapsed (kernels + device-to-device snapshots): "
              << gpu_elapsed_ms << " ms\n";

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));

    // free memories
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_hist_x)); CUDA_CHECK(cudaFree(d_hist_y)); CUDA_CHECK(cudaFree(d_hist_z));

    return 0;
}
//optimized version
//100 planet in 1 year 10 1 100: Execution time : 36136 ms ~= 36.136 seconds ~= 0.6 minutes
