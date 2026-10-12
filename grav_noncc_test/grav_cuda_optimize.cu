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
 * Transfer-heavy version: the host owns the simulation state. Every time step
 * uploads the full state to the GPU, runs a trivial O(N) kernel (each body only
 * feels the central star, no body-body forces), and downloads the state again,
 * so host<->device transfers dominate the runtime instead of GPU math.
 * The 75 GB device history ring buffer is still filled every print_interval.
 */

constexpr float GM_SUN = 1.32712440018e20f;    // central star at the origin; the input orbits assume one
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
// Trivial O(N) step: each body only feels the central star, so the GPU does a
// handful of flops per body. Kick then drift, same order as the old
// update_velocities + update_positions pair.
__global__ void step_central_star(
    int n,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ z,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float px = x[i];
    const float py = y[i];
    const float pz = z[i];
    const float dist_sq = px * px + py * py + pz * pz + POSITION_CORRECTION;
    const float inv_r = rsqrtf(dist_sq);
    const float factor = -GM_SUN * inv_r * inv_r * inv_r;

    const float new_vx = fmaf(factor * px, dt, vx[i]);
    const float new_vy = fmaf(factor * py, dt, vy[i]);
    const float new_vz = fmaf(factor * pz, dt, vz[i]);
    vx[i] = new_vx;
    vy[i] = new_vy;
    vz[i] = new_vz;

    x[i] = fmaf(new_vx, dt, px);
    y[i] = fmaf(new_vy, dt, py);
    z[i] = fmaf(new_vz, dt, pz);
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

    // The host owns the state. Keep it in pinned memory as one SoA block
    // [x | y | z | vx | vy | vz] so every step is one full-bandwidth DMA each way.
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t state_bytes = 6 * bytes;
    float* h_state = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_state, state_bytes));
    std::copy(h_x.begin(), h_x.end(), h_state + 0 * n);
    std::copy(h_y.begin(), h_y.end(), h_state + 1 * n);
    std::copy(h_z.begin(), h_z.end(), h_state + 2 * n);
    std::copy(h_vx.begin(), h_vx.end(), h_state + 3 * n);
    std::copy(h_vy.begin(), h_vy.end(), h_state + 4 * n);
    std::copy(h_vz.begin(), h_vz.end(), h_state + 5 * n);

    // allocate memories on device
    // Target 75 decimal GB across all device buffers. Store whole frames
    // in the history buffers and wrap when the simulation exceeds capacity.
    constexpr size_t target_device_bytes = 75'000'000'000ULL;
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

    float* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, state_bytes));
    float* d_x = d_state + 0 * n;
    float* d_y = d_state + 1 * n;
    float* d_z = d_state + 2 * n;
    float* d_vx = d_state + 3 * n;
    float* d_vy = d_state + 4 * n;
    float* d_vz = d_state + 5 * n;

    // Initial upload so the first history frame has data in it.
    CUDA_CHECK(cudaMemcpy(d_state, h_state, state_bytes, cudaMemcpyHostToDevice));

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

    // Per-phase timing: [0] H2D [1] kernel [2] D2H [3], plus snapshot start/stop.
    cudaEvent_t ev_step[4], ev_snap[2];
    for (cudaEvent_t& ev : ev_step) CUDA_CHECK(cudaEventCreate(&ev));
    for (cudaEvent_t& ev : ev_snap) CUDA_CHECK(cudaEventCreate(&ev));

    auto elapsed_ms = [](cudaEvent_t from, cudaEvent_t to) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, from, to));
        return static_cast<double>(ms);
    };

    double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0, d2d_ms = 0.0;
    unsigned long long h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0;
    unsigned long long steps = 0;

    // Copy current positions into the device history ring buffer.
    // Older snapshots are overwritten once history is full.
    auto snapshot = [&]() {
        size_t offset = (frame_count % total_frames) * static_cast<size_t>(n);
        CUDA_CHECK(cudaEventRecord(ev_snap[0]));
        CUDA_CHECK(cudaMemcpy(d_hist_x + offset, d_x, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_hist_y + offset, d_y, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_hist_z + offset, d_z, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(ev_snap[1]));
        CUDA_CHECK(cudaEventSynchronize(ev_snap[1]));
        d2d_ms += elapsed_ms(ev_snap[0], ev_snap[1]);
        d2d_bytes += 3 * bytes;
        frame_count++;
    };

    CUDA_CHECK(cudaEventRecord(gpu_start));

    while (t < END) {
        snapshot();

        //update velocity and position, step foward based on time interval
        float next_print = t + print_interval;
        while (t < next_print && t < END) {
            float step_dt = std::min(dt, END - t);

            // Host owns the state: upload all of it, take one cheap step on the
            // GPU, download all of it. Nothing stays resident between steps.
            CUDA_CHECK(cudaEventRecord(ev_step[0]));
            CUDA_CHECK(cudaMemcpy(d_state, h_state, state_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(ev_step[1]));
            step_central_star<<<blocks, threads>>>(n, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ev_step[2]));
            CUDA_CHECK(cudaMemcpy(h_state, d_state, state_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(ev_step[3]));
            CUDA_CHECK(cudaEventSynchronize(ev_step[3]));

            h2d_ms += elapsed_ms(ev_step[0], ev_step[1]);
            kernel_ms += elapsed_ms(ev_step[1], ev_step[2]);
            d2h_ms += elapsed_ms(ev_step[2], ev_step[3]);
            h2d_bytes += state_bytes;
            d2h_bytes += state_bytes;
            steps++;

            t += step_dt;
        }
    }

    // last frame needed
    snapshot();

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

    auto gb_per_s = [](unsigned long long b, double ms) { return ms > 0.0 ? b / (ms * 1e6) : 0.0; };
    const double transfer_ms = h2d_ms + d2h_ms + d2d_ms;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Steps: " << steps << ", history frames: " << frame_count << '\n';
    std::cout << "Host->Device:   " << h2d_bytes / 1e9 << " GB in " << h2d_ms << " ms ("
              << gb_per_s(h2d_bytes, h2d_ms) << " GB/s)\n";
    std::cout << "Device->Host:   " << d2h_bytes / 1e9 << " GB in " << d2h_ms << " ms ("
              << gb_per_s(d2h_bytes, d2h_ms) << " GB/s)\n";
    std::cout << "Device->Device: " << d2d_bytes / 1e9 << " GB in " << d2d_ms << " ms ("
              << gb_per_s(d2d_bytes, d2d_ms) << " GB/s)\n";
    std::cout << "Kernel compute: " << kernel_ms << " ms\n";
    if (transfer_ms + kernel_ms > 0.0) {
        std::cout << "Transfers are " << 100.0 * transfer_ms / (transfer_ms + kernel_ms)
                  << "% of measured GPU time\n";
    }
    std::cout << "GPU elapsed (transfers + kernels + snapshots): "
              << gpu_elapsed_ms << " ms\n";

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    for (cudaEvent_t ev : ev_step) CUDA_CHECK(cudaEventDestroy(ev));
    for (cudaEvent_t ev : ev_snap) CUDA_CHECK(cudaEventDestroy(ev));

    // free memories
    CUDA_CHECK(cudaFreeHost(h_state));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_hist_x)); CUDA_CHECK(cudaFree(d_hist_y)); CUDA_CHECK(cudaFree(d_hist_z));

    return 0;
}
//optimized version
//100 planet in 1 year 10 1 100: Execution time : 36136 ms ~= 36.136 seconds ~= 0.6 minutes
