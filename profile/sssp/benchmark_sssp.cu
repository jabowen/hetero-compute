#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/benchmarks/gpu_benchmark.cuh"
#include "../../src/benchmarks/cpu_benchmark.h"
#include "../../src/benchmarks/heterogeneous_benchmark.cuh"
#include "../../src/kernels/cpu/sssp_pull.h"
#include "../../src/kernels/gpu/sssp_pull.cuh"

/******************************************************************************
 ***** Config *******************************************************************
 ******************************************************************************/

// Only get results for a particular depth (not the whole tree).
#define ONLY_LAYER
// Print results to std::cout.
#define PRINT_RESULTS
// Save results to YAML files.
#define SAVE_RESULTS
// Run full kernels.
/*#define FULL_KERNEL*/

#ifdef ONLY_LAYER
// Number of segments (NOT depth).
#define SEGMENTS 1 << 4
#else
// Current/Up to (inclusive) this depth.
#define DEPTH 6
#endif // ONLY_LAYER

/******************************************************************************
 ***** Helper Functions *******************************************************
 ******************************************************************************/

/** Identfier for devices. */
enum class Device {
    intel_i7_9700K, nvidia_quadro_rtx_4000
};

/**
 * Converts device ID to its name.
 * Parameters:
 *   - dev <- device ID.
 * Returns:
 *  device name.
 */
std::string to_string(Device dev) {
    switch (dev) {
        case Device::intel_i7_9700K:         return "Intel i7-9700K";
        case Device::nvidia_quadro_rtx_4000: return "NVIDIA Quadro RTX 4000";
    }
    return "";
}

/**
 * Save results to a file.
 * Parameters:
 *   - filename <- filename.
 *   - result <- result type (must have an operator<< implemented).
 */
#ifdef SAVE_RESULTS
template <typename ResT>
__inline__
void save_results(std::string filename, ResT &result) {
    std::ofstream ofs(filename, std::ofstream::out);
    ofs << result;
    ofs.close();
}
#endif // SAVE_RESULTS

/**
 * Generates an identifier for an arbitrary kernel.
 * Parameters:
 *   - ker  <- kernel ID.
 *   - args <- zero or more arguments to append to the identifer.
 * Returns:
 *   kernel identifier.
 */
template <typename IdT, typename ...OptArgsT>
__inline__
std::string get_identifier(IdT ker, OptArgsT ...args) {
    std::stringstream ss;
    ss << to_string(ker);
    if (sizeof...(args) > 0)
        volatile int unused[] = { (ss << "_" << args, 0)... };
    return ss.str();
}

/**
 * Runs a tree benchmark from an arbitrary device and kernel.
 * Parameters:
 *   - BenchmarkT <- tree benchmark class.
 *   - g          <- graph.
 *   - dev        <- device ID.
 *   - ker        <- kernel ID.
 *   - args       <- optional arguments for the identifier.
 */
template <class BenchmarkT, typename IdT, typename ...OptArgsT,
         typename = typename std::enable_if<
            std::is_base_of<TreeBenchmark, BenchmarkT>::value>>
void run_treebenchmark(CSRWGraph &g, Device dev, IdT ker, OptArgsT ...args) {
    // Run benchmark.
    // TOOD: this is a hacky way to pass arguments into the benchmark function.
    BenchmarkT bench(&g, get_kernel(ker), args...);

#ifdef ONLY_LAYER
    auto res = bench.layer_microbenchmark(SEGMENTS);
#else 
    auto res = bench.tree_microbenchmark(DEPTH);
#endif  // ONLY_LAYER

    // Configure metadata.
    std::string kerid = get_identifier(ker, args...);
    res.device_name = to_string(dev);
    res.kernel_name = kerid;

    // Output results appropriately.
#ifdef PRINT_RESULTS
    std::cout << res;
#endif // PRINT_RESULTS

#ifdef SAVE_RESULTS
    save_results(kerid + ".yaml", res);
#endif // SAVE_RESULTS
}       
 

/******************************************************************************
 ***** Main *******************************************************************
 ******************************************************************************/

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [graph.wsg]" << std::endl;
        return EXIT_FAILURE;
    }

    // Load in graph.
    CSRWGraph g;
    std::cout << "Loading graph ..." << std::endl;

    std::ifstream ifs(argv[1], std::ifstream::in | std::ifstream::binary);
    Timer timer; timer.Start(); // Start timing.
    ifs >> g;
    timer.Stop(); // Stop timing.
    ifs.close();

    std::cout << " > Loaded in " << timer.Millisecs() << " ms." << std::endl;

    // Run CPU benchmarks.
    run_treebenchmark<SSSPCPUTreeBenchmark>(g, Device::intel_i7_9700K, 
            SSSPCPU::one_to_one);

    // Run GPU benchmarks.
    const int block_count = 64;
    run_treebenchmark<SSSPGPUTreeBenchmark>(g, Device::nvidia_quadro_rtx_4000,
            SSSPGPU::one_to_one, block_count, 1024);
    run_treebenchmark<SSSPGPUTreeBenchmark>(g, Device::nvidia_quadro_rtx_4000,
            SSSPGPU::warp_min, block_count, 1024);

    std::vector<int> thread_counts = {64, 128, 256, 512, 1024};
    for (int thread_count : thread_counts) 
        run_treebenchmark<SSSPGPUTreeBenchmark>(g, 
                Device::nvidia_quadro_rtx_4000, SSSPGPU::block_min,
                block_count * (1024 / thread_count), thread_count);

    // Full kernel runs.
#ifdef FULL_KERNEL
    weight_t *ret_dist  = nullptr;
    weight_t *init_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        init_dist[i] = INF_WEIGHT;
    init_dist[0] = 0; // Arbitrarily set highest degree node to source.

    // Run CPU kernel.
    {
        std::cout << "SSSP CPU:" << std::endl;
        segment_res_t res = benchmark_sssp_cpu(g, epoch_sssp_pull_cpu,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    // Run GPU naive kernel.
    {
        std::cout << "SSSP GPU naive:" << std::endl;
        segment_res_t res = benchmark_sssp_gpu(g, epoch_sssp_pull_gpu_naive,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    // Run GPU warp min kernel.
    {
        std::cout << "SSSP GPU warp min:" << std::endl;
        segment_res_t res = benchmark_sssp_gpu(g, epoch_sssp_pull_gpu_warp_min,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    // Run GPU block min kernel.
    {
        std::cout << "SSSP GPU block min:" << std::endl;
        segment_res_t res = benchmark_sssp_gpu(g, epoch_sssp_pull_gpu_block_min,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
   }

    // Run heterogeneous kernel.
    {
        nid_t                      middle     = g.num_nodes / 16 * 2;
        graph_range_t              cpu_range  = { middle, g.num_nodes };
        std::vector<graph_range_t> gpu_ranges = { { 0, middle } };
        std::cout << "SSSP heterogeneous:" << std::endl;
        segment_res_t res = benchmark_sssp_heterogeneous(g,
                epoch_sssp_pull_cpu, epoch_sssp_pull_gpu_warp_min,
                cpu_range, gpu_ranges, init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    delete[] init_dist;
#endif // FULL_KERNEL

    return EXIT_SUCCESS;
}
