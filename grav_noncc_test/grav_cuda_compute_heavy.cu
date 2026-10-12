#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
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
 * Compute-heavy version of grav_cuda_optimize.cu.
 *
 * The state is uploaded to the GPU once and never leaves it while the
 * simulation runs: no history buffers, no snapshots, no per-step copies.
 * Each step is a 4th-order Runge-Kutta update in double precision, i.e. four
 * O(N^2) all-pairs force evaluations per step (plus the central star).
 * The only data that comes back is a few KB of energy partial sums, computed
 * on the GPU before and after the run as a correctness check.
 */

constexpr double G = 5.67e-11;
constexpr double GM_SUN = 1.32712440018e20;    // central star at the origin; the input orbits assume one
constexpr double POSITION_CORRECTION = 1e-9;
constexpr int THREADS = 256;

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t __err = (expr);                                                  \
        if (__err != cudaSuccess) {                                                  \
            std::cerr << "CUDA error (" << #expr << "): "                           \
                      << cudaGetErrorString(__err) << '\n';                         \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

struct __align__(16) BodyTile {
    double x, y, z, gm;
};

__device__ __forceinline__ BodyTile load_body(int j, int n,
                                              const double* __restrict__ m,
                                              const double* __restrict__ x,
                                              const double* __restrict__ y,
                                              const double* __restrict__ z) {
    if (j < n) return {x[j], y[j], z[j], G * m[j]};
    return {0.0, 0.0, 0.0, 0.0};
}

// kernels
// Acceleration on every body from the central star plus all other bodies,
// tiled through shared memory like the original update_velocities_tiled,
// but in double precision.
__global__ void compute_accel_tiled(
    int n,
    const double* __restrict__ m,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    double* __restrict__ ax,
    double* __restrict__ ay,
    double* __restrict__ az) {
    __shared__ BodyTile tile[THREADS];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const bool active = i < n;

    double my_x = 0.0;
    double my_y = 0.0;
    double my_z = 0.0;
    if (active) {
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
    }

    const double inv_r_star = rsqrt(my_x * my_x + my_y * my_y + my_z * my_z + POSITION_CORRECTION);
    const double star_factor = -GM_SUN * inv_r_star * inv_r_star * inv_r_star;
    double acc_x = star_factor * my_x;
    double acc_y = star_factor * my_y;
    double acc_z = star_factor * my_z;

    for (int tile_start = 0; tile_start < n; tile_start += THREADS) {
        tile[tid] = load_body(tile_start + tid, n, m, x, y, z);

        __syncthreads();
        const int tile_count = min(THREADS, n - tile_start);

        if (active) {
            #pragma unroll 4
            for (int k = 0; k < tile_count; ++k) {
                if (tile_start + k == i) continue;

                const BodyTile body = tile[k];
                const double dx = body.x - my_x;
                const double dy = body.y - my_y;
                const double dz = body.z - my_z;
                const double dist_sq = dx * dx + dy * dy + dz * dz + POSITION_CORRECTION;
                const double inv_r = rsqrt(dist_sq);
                const double factor = body.gm * inv_r * inv_r * inv_r;

                acc_x = fma(factor, dx, acc_x);
                acc_y = fma(factor, dy, acc_y);
                acc_z = fma(factor, dz, acc_z);
            }
        }

        __syncthreads();
    }

    if (active) {
        ax[i] = acc_x;
        ay[i] = acc_y;
        az[i] = acc_z;
    }
}

// One RK4 stage over the packed [x | y | z] arrays (len = 3n). Folds this
// stage's derivatives (k_r = stage velocity, k_v = stage acceleration) into
// the running sums and builds the state for the next force evaluation:
//   stage_r = r + c * k_r,  stage_v = v + c * k_v
__global__ void rk4_stage(
    int len, bool first, double weight, double c,
    const double* __restrict__ r,
    const double* __restrict__ v,
    const double* __restrict__ a,
    double* __restrict__ stage_r,
    double* __restrict__ stage_v,
    double* __restrict__ sum_r,
    double* __restrict__ sum_v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const double k_r = first ? v[i] : stage_v[i];
    const double k_v = a[i];
    sum_r[i] = first ? weight * k_r : fma(weight, k_r, sum_r[i]);
    sum_v[i] = first ? weight * k_v : fma(weight, k_v, sum_v[i]);
    stage_r[i] = fma(c, k_r, r[i]);
    stage_v[i] = fma(c, k_v, v[i]);
}

// Final RK4 combine: r += h/6 (k1 + 2k2 + 2k3 + k4), same for v.
__global__ void rk4_finish(
    int len, double h,
    double* __restrict__ r,
    double* __restrict__ v,
    const double* __restrict__ stage_v,
    const double* __restrict__ a,
    const double* __restrict__ sum_r,
    const double* __restrict__ sum_v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const double sixth_h = h / 6.0;
    r[i] = fma(sixth_h, sum_r[i] + stage_v[i], r[i]);
    v[i] = fma(sixth_h, sum_v[i] + a[i], v[i]);
}

// Per-block partial sums of the total energy: kinetic + central-star
// potential + pairwise potential. Requires blockDim.x == THREADS.
__global__ void energy_partial_sums(
    int n,
    const double* __restrict__ m,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    double* __restrict__ block_sums) {
    __shared__ BodyTile tile[THREADS];
    __shared__ double partial[THREADS];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const bool active = i < n;

    double my_m = 0.0;
    double my_x = 0.0;
    double my_y = 0.0;
    double my_z = 0.0;
    double energy = 0.0;
    if (active) {
        my_m = m[i];
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        const double v_sq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        energy = 0.5 * my_m * v_sq
               - GM_SUN * my_m * rsqrt(my_x * my_x + my_y * my_y + my_z * my_z + POSITION_CORRECTION);
    }

    double pair_potential = 0.0;
    for (int tile_start = 0; tile_start < n; tile_start += THREADS) {
        tile[tid] = load_body(tile_start + tid, n, m, x, y, z);

        __syncthreads();
        const int tile_count = min(THREADS, n - tile_start);

        if (active) {
            for (int k = 0; k < tile_count; ++k) {
                if (tile_start + k == i) continue;

                const BodyTile body = tile[k];
                const double dx = body.x - my_x;
                const double dy = body.y - my_y;
                const double dz = body.z - my_z;
                pair_potential += body.gm * rsqrt(dx * dx + dy * dy + dz * dz + POSITION_CORRECTION);
            }
        }

        __syncthreads();
    }
    // Every pair is visited from both sides, hence the 1/2.
    energy -= 0.5 * my_m * pair_potential;

    partial[tid] = energy;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = partial[0];
}

//main function
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [dt] [END]\n";
        return 1;
    }
    constexpr double YEAR = 365.25 * 24.0 * 60.0 * 60.0; // matches gravsim0.cpp

    const double dt = atof(argv[1]);         // timestep in seconds
    const double END = atof(argv[2]) * YEAR; // years

    std::string input_filename = "planet_bodies_100k.dat";

    if (!std::isfinite(dt) || dt <= 0.0 ||
        !std::isfinite(END) || END < 0.0) {
        std::cerr << "Error: dt must be positive and finite; END must be nonnegative and finite.\n";
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
    std::vector<double> h_m, h_x, h_y, h_z, h_vx, h_vy, h_vz;

    std::cout << "Reading from " << input_filename << "...\n";

    while (infile.peek() != EOF) {
        // Skip 't=...' headers if present (e.g., from a restart file)
        if (infile.peek() == 't') {
            std::getline(infile, line);
            continue;
        }

        std::string name;
        double mass, x, y, z, vx, vy, vz;
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
    std::cerr << "dt=" << dt << "\tEND=" << (END / YEAR) << " years\n";

    // allocate memories on device
    // Positions, velocities and the RK4 scratch arrays are packed as [x | y | z].
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t vec_bytes = 3 * bytes;
    const int len = 3 * n;
    const int body_blocks = (n + THREADS - 1) / THREADS;
    const int vec_blocks = (len + THREADS - 1) / THREADS;

    double *d_m, *d_pos, *d_vel, *d_acc, *d_stage_pos, *d_stage_vel, *d_sum_r, *d_sum_v, *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMalloc(&d_pos, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_stage_pos, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_stage_vel, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum_r, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum_v, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_sums, body_blocks * sizeof(double)));

    // The only bulk transfer in the program: upload the initial state once.
    CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos + 0 * n, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos + 1 * n, h_y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos + 2 * n, h_z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel + 0 * n, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel + 1 * n, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel + 2 * n, h_vz.data(), bytes, cudaMemcpyHostToDevice));
    unsigned long long h2d_bytes = 7 * bytes;
    unsigned long long d2h_bytes = 0;

    std::cerr << "Using " << body_blocks << " blocks of " << THREADS << " threads\n";

    auto accel = [&](const double* pos) {
        compute_accel_tiled<<<body_blocks, THREADS>>>(
            n, d_m, pos, pos + n, pos + 2 * n, d_acc, d_acc + n, d_acc + 2 * n);
    };
    auto stage = [&](bool first, double weight, double c) {
        rk4_stage<<<vec_blocks, THREADS>>>(
            len, first, weight, c, d_pos, d_vel, d_acc, d_stage_pos, d_stage_vel, d_sum_r, d_sum_v);
    };
    std::vector<double> h_block_sums(body_blocks);
    auto total_energy = [&]() {
        energy_partial_sums<<<body_blocks, THREADS>>>(
            n, d_m, d_pos, d_pos + n, d_pos + 2 * n, d_vel, d_vel + n, d_vel + 2 * n, d_block_sums);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, body_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));
        d2h_bytes += body_blocks * sizeof(double);
        double sum = 0.0;
        for (double s : h_block_sums) sum += s;
        return sum;
    };

    const double energy_start = total_energy();

    cudaEvent_t gpu_start, gpu_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    CUDA_CHECK(cudaEventRecord(gpu_start));

    // Everything below stays on the GPU; the host only queues kernels.
    double t = 0.0;
    unsigned long long steps = 0;
    while (t < END) {
        const double h = std::min(dt, END - t);

        // RK4: k1 at the current state, k2 and k3 at the half step, k4 at the full step.
        accel(d_pos);
        stage(true, 1.0, 0.5 * h);
        accel(d_stage_pos);
        stage(false, 2.0, 0.5 * h);
        accel(d_stage_pos);
        stage(false, 2.0, h);
        accel(d_stage_pos);
        rk4_finish<<<vec_blocks, THREADS>>>(
            len, h, d_pos, d_vel, d_stage_vel, d_acc, d_sum_r, d_sum_v);
        CUDA_CHECK(cudaGetLastError());

        t += h;
        steps++;
    }

    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed_ms, gpu_start, gpu_stop));

    const double energy_end = total_energy();

    // 4 force evaluations per step, each ~n^2 body-body interactions.
    const double interactions = 4.0 * static_cast<double>(n) * static_cast<double>(n) * steps;

    std::cout << "Steps: " << steps << " (" << 4 * steps << " all-pairs force evaluations)\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU elapsed (RK4 steps only): " << gpu_elapsed_ms << " ms";
    if (steps > 0) std::cout << " (" << gpu_elapsed_ms / steps << " ms/step)";
    std::cout << '\n';
    if (gpu_elapsed_ms > 0.0f) {
        std::cout << "Throughput: " << interactions / (gpu_elapsed_ms * 1e-3) / 1e9
                  << " billion FP64 body-body interactions/s\n";
    }
    std::cout << "Host->Device: " << h2d_bytes / 1e6 << " MB (once, before the run)\n";
    std::cout << "Device->Host: " << d2h_bytes / 1e6 << " MB (energy check only)\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Total energy: start " << energy_start << " J, end " << energy_end
              << " J, relative drift " << (energy_end - energy_start) / std::abs(energy_start) << '\n';

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));

    // free memories
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_pos)); CUDA_CHECK(cudaFree(d_vel)); CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_stage_pos)); CUDA_CHECK(cudaFree(d_stage_vel));
    CUDA_CHECK(cudaFree(d_sum_r)); CUDA_CHECK(cudaFree(d_sum_v));
    CUDA_CHECK(cudaFree(d_block_sums));

    return 0;
}
