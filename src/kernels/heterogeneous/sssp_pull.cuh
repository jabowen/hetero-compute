/**
 * Heterogeneous implementations of SSSP pull kernel.
 * This is generated by util/scheduler/scheduler/kernelgen/sssp_hetero.py.
 */

#ifndef SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
#define SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH

#include <omp.h>
#include <vector>

#include "../kernel_types.h"
#include "../cpu/sssp_pull.h"
#include "../gpu/sssp_pull.cuh"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../util.h"

constexpr int num_gpus = 1;

/**
 * Runs SSSP kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. 
 * Configuration:
 *   - 1x 
 *   - 1x NVIDIA Quadro RTX 4000
 *
 * Parameters:
 *   - g         <- graph.
 *   - init_dist <- initial distance array.
 *   - ret_dist  <- pointer to the address of the return distance array.
 * Returns:
 *   Execution time in milliseconds.
 */
double sssp_pull_heterogeneous(const CSRWGraph &g, 
        const weight_t *init_dist, weight_t ** const ret_dist
) {
    // Configuration.
    constexpr int num_blocks   = 4;
    constexpr int num_segments = 16;
    
    // Copy graph.
    nid_t *seg_ranges = compute_equal_edge_ranges(g, num_segments);
    
    /// Block ranges to reduce irregular memory acceses.
    constexpr int gpu_blocks[] = {0, 4};
    nid_t block_ranges[num_blocks * 2];

    block_ranges[0] = seg_ranges[0]; // Block 0 Start 0
    block_ranges[1] = seg_ranges[1]; // Block 0 End 1 (excl.)
    block_ranges[2] = seg_ranges[1]; // Block 1 Start 1
    block_ranges[3] = seg_ranges[2]; // Block 1 End 2 (excl.)
    block_ranges[4] = seg_ranges[2]; // Block 2 Start 2
    block_ranges[5] = seg_ranges[15]; // Block 2 End 15 (excl.)
    block_ranges[6] = seg_ranges[15]; // Block 3 Start 15
    block_ranges[7] = seg_ranges[16]; // Block 3 End 16 (excl.)

    /// Actual graphs on GPU memory.
    offset_t *cu_indices[num_blocks];
    wnode_t  *cu_neighbors[num_blocks];

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++) 
            copy_subgraph_to_device(g,
                    &cu_indices[block], &cu_neighbors[block],
                    block_ranges[2 * block], block_ranges[2 * block + 1]);
    }

    // Distance.
    size_t   dist_size = g.num_nodes * sizeof(weight_t);
    weight_t *dist     = nullptr; 

    /// CPU Distance.
    CUDA_ERRCHK(cudaMallocHost((void **) &dist, dist_size));
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = init_dist[i];

    /// GPU Distances.
    weight_t *cu_dists[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_dists[gpu], dist_size));
        CUDA_ERRCHK(cudaMemcpy(cu_dists[gpu], init_dist, dist_size,
            cudaMemcpyHostToDevice));
    }

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updateds[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_updateds[gpu], 
                sizeof(nid_t)));
    }

    // Create streams.
    cudaStream_t streams[num_blocks];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++)
            CUDA_ERRCHK(cudaStreamCreate(&streams[b]));
    }

    // Get init vertex.
    nid_t start;
    for (nid_t i = 0; i < g.num_nodes; i++)
        if (init_dist[i] != INF_WEIGHT) start = i;

    // Start kernel!
    Timer timer; timer.Start();
    int epochs = 0;

    // Push for the first iteration.
    // TODO: implement push for more than one epoch. Requires parallel queue.
    for (wnode_t nei : g.get_neighbors(start)) {
        dist[nei.v] = nei.w;       
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                cu_dists[gpu] + nei.v, dist + nei.v,
                sizeof(weight_t), cudaMemcpyHostToDevice));
        }
    }
    epochs++;

    while (updated != 0) {
        // Reset update counters.
        updated = cpu_updated = 0;          
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemsetAsync(cu_updateds[gpu], 0, 
                    sizeof(nid_t)));
        }

        // Launch GPU epoch kernels.
        // Implicit CUDA device synchronize at the start of kernels.
        CUDA_ERRCHK(cudaSetDevice(0));
        epoch_sssp_pull_gpu_block_min<<<64, 1024, 0, streams[0]>>>(
                cu_indices[0], cu_neighbors[0],
                block_ranges[0], block_ranges[1],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_block_min<<<512, 128, 0, streams[1]>>>(
                cu_indices[1], cu_neighbors[1],
                block_ranges[2], block_ranges[3],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_warp_min<<<64, 1024, 0, streams[2]>>>(
                cu_indices[2], cu_neighbors[2],
                block_ranges[4], block_ranges[5],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_one_to_one<<<64, 1024, 0, streams[3]>>>(
                cu_indices[3], cu_neighbors[3],
                block_ranges[6], block_ranges[7],
                cu_dists[0], cu_updateds[0]);

        // Launch CPU epoch kernels.
                

        // Sync streams.
        for (int i = 0; i < num_blocks; i++)
            CUDA_ERRCHK(cudaStreamSynchronize(streams[i]));

        // Synchronize updates.
        nid_t gpu_updateds[num_gpus];
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                    &gpu_updateds[gpu], cu_updateds[gpu], 
                    sizeof(nid_t), cudaMemcpyDeviceToHost));
        }
        updated += cpu_updated;

        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaDeviceSynchronize());
            updated += gpu_updateds[gpu];
        }

        // Only update GPU distances if another epoch will be run.
        if (updated != 0) {
            // Copy CPU distances to all GPUs.
            

            // Copy GPU distances peer-to-peer.
            
        }
        epochs++;
    }
    // Copy GPU distances back to host.
    CUDA_ERRCHK(cudaSetDevice(0))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[0], cu_dists[0] + seg_ranges[0],
        (seg_ranges[16] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    // Wait for memops to complete.
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaDeviceSynchronize());
    }
    timer.Stop();
    std::cout << "Epochs: " << epochs << std::endl;

    // Copy output.
    *ret_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        (*ret_dist)[i] = dist[i];

    // Free streams.
    for (int b = 0; b < num_blocks; b++)
        CUDA_ERRCHK(cudaStreamDestroy(streams[b]));

    // Free memory.
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaFree(cu_updateds[gpu]));
        CUDA_ERRCHK(cudaFree(cu_dists[gpu]));
        
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++
        ) {
            CUDA_ERRCHK(cudaFree(cu_indices[block]));
            CUDA_ERRCHK(cudaFree(cu_neighbors[block]));
        }
    }
    CUDA_ERRCHK(cudaFreeHost(dist));
    delete[] seg_ranges;

    return timer.Millisecs();
}

/**
 * Enable peer access between all compatible GPUs.
 */
void enable_all_peer_access() {
    int can_access_peer;
    for (int from = 0; from < num_gpus; from++) {
        CUDA_ERRCHK(cudaSetDevice(from));

        for (int to = 0; to < num_gpus; to++) {
            if (from == to) continue;

            CUDA_ERRCHK(cudaDeviceCanAccessPeer(&can_access_peer, from, to));
            if(can_access_peer)
                CUDA_ERRCHK(cudaDeviceEnablePeerAccess(to, 0));
        }
    }
}

#endif // SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH