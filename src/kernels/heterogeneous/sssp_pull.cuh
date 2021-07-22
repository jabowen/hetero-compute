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

constexpr int num_gpus = 8;

/** Forward decl. */
void gpu_butterfly_P2P(nid_t *seg_ranges, weight_t **cu_dists);

/**
 * Runs SSSP kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. 
 * Configuration:
 *   - 1x 
 *   - 8x NVIDIA Quadro RTX 4000
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
    constexpr int num_blocks   = 20;
    constexpr int num_segments = 36;
    
    // Copy graph.
    nid_t *seg_ranges = compute_equal_edge_ranges(g, num_segments);
    
    /// Block ranges to reduce irregular memory acceses.
    constexpr int gpu_blocks[] = {0, 5, 7, 9, 12, 16, 18, 19, 20};
    nid_t block_ranges[num_blocks * 2];

    block_ranges[0] = seg_ranges[13]; // Block 0 Start 13
    block_ranges[1] = seg_ranges[14]; // Block 0 End 14 (excl.)
    block_ranges[2] = seg_ranges[0]; // Block 1 Start 0
    block_ranges[3] = seg_ranges[1]; // Block 1 End 1 (excl.)
    block_ranges[4] = seg_ranges[1]; // Block 2 Start 1
    block_ranges[5] = seg_ranges[2]; // Block 2 End 2 (excl.)
    block_ranges[6] = seg_ranges[2]; // Block 3 Start 2
    block_ranges[7] = seg_ranges[4]; // Block 3 End 4 (excl.)
    block_ranges[8] = seg_ranges[4]; // Block 4 Start 4
    block_ranges[9] = seg_ranges[7]; // Block 4 End 7 (excl.)
    block_ranges[10] = seg_ranges[18]; // Block 5 Start 18
    block_ranges[11] = seg_ranges[19]; // Block 5 End 19 (excl.)
    block_ranges[12] = seg_ranges[7]; // Block 6 Start 7
    block_ranges[13] = seg_ranges[13]; // Block 6 End 13 (excl.)
    block_ranges[14] = seg_ranges[23]; // Block 7 Start 23
    block_ranges[15] = seg_ranges[24]; // Block 7 End 24 (excl.)
    block_ranges[16] = seg_ranges[14]; // Block 8 Start 14
    block_ranges[17] = seg_ranges[18]; // Block 8 End 18 (excl.)
    block_ranges[18] = seg_ranges[28]; // Block 9 Start 28
    block_ranges[19] = seg_ranges[29]; // Block 9 End 29 (excl.)
    block_ranges[20] = seg_ranges[19]; // Block 10 Start 19
    block_ranges[21] = seg_ranges[22]; // Block 10 End 22 (excl.)
    block_ranges[22] = seg_ranges[22]; // Block 11 Start 22
    block_ranges[23] = seg_ranges[23]; // Block 11 End 23 (excl.)
    block_ranges[24] = seg_ranges[24]; // Block 12 Start 24
    block_ranges[25] = seg_ranges[25]; // Block 12 End 25 (excl.)
    block_ranges[26] = seg_ranges[25]; // Block 13 Start 25
    block_ranges[27] = seg_ranges[26]; // Block 13 End 26 (excl.)
    block_ranges[28] = seg_ranges[26]; // Block 14 Start 26
    block_ranges[29] = seg_ranges[27]; // Block 14 End 27 (excl.)
    block_ranges[30] = seg_ranges[27]; // Block 15 Start 27
    block_ranges[31] = seg_ranges[28]; // Block 15 End 28 (excl.)
    block_ranges[32] = seg_ranges[29]; // Block 16 Start 29
    block_ranges[33] = seg_ranges[31]; // Block 16 End 31 (excl.)
    block_ranges[34] = seg_ranges[31]; // Block 17 Start 31
    block_ranges[35] = seg_ranges[33]; // Block 17 End 33 (excl.)
    block_ranges[36] = seg_ranges[33]; // Block 18 Start 33
    block_ranges[37] = seg_ranges[35]; // Block 18 End 35 (excl.)
    block_ranges[38] = seg_ranges[35]; // Block 19 Start 35
    block_ranges[39] = seg_ranges[36]; // Block 19 End 36 (excl.)

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
    cudaStream_t memcpystreams[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaStreamCreate(&memcpystreams[gpu]));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_dists[gpu], dist_size));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_dists[gpu], dist, dist_size,
            cudaMemcpyHostToDevice, memcpystreams[gpu]));
    }
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_ERRCHK(cudaStreamSynchronize(memcpystreams[gpu]));
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
        if (nei.v == start) continue;

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
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[4]>>>(
                cu_indices[4], cu_neighbors[4],
                block_ranges[8], block_ranges[9],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_block_min<<<4096, 64, 0, streams[3]>>>(
                cu_indices[3], cu_neighbors[3],
                block_ranges[6], block_ranges[7],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_block_min<<<2048, 128, 0, streams[2]>>>(
                cu_indices[2], cu_neighbors[2],
                block_ranges[4], block_ranges[5],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_block_min<<<256, 1024, 0, streams[1]>>>(
                cu_indices[1], cu_neighbors[1],
                block_ranges[2], block_ranges[3],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[0]>>>(
                cu_indices[0], cu_neighbors[0],
                block_ranges[0], block_ranges[1],
                cu_dists[0], cu_updateds[0]);
        
        CUDA_ERRCHK(cudaSetDevice(1));
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[6]>>>(
                cu_indices[6], cu_neighbors[6],
                block_ranges[12], block_ranges[13],
                cu_dists[1], cu_updateds[1]);
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[5]>>>(
                cu_indices[5], cu_neighbors[5],
                block_ranges[10], block_ranges[11],
                cu_dists[1], cu_updateds[1]);
        
        CUDA_ERRCHK(cudaSetDevice(2));
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[8]>>>(
                cu_indices[8], cu_neighbors[8],
                block_ranges[16], block_ranges[17],
                cu_dists[2], cu_updateds[2]);
        epoch_sssp_pull_gpu_block_min<<<2048, 128, 0, streams[7]>>>(
                cu_indices[7], cu_neighbors[7],
                block_ranges[14], block_ranges[15],
                cu_dists[2], cu_updateds[2]);
        
        CUDA_ERRCHK(cudaSetDevice(3));
        epoch_sssp_pull_gpu_block_min<<<2048, 128, 0, streams[11]>>>(
                cu_indices[11], cu_neighbors[11],
                block_ranges[22], block_ranges[23],
                cu_dists[3], cu_updateds[3]);
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[10]>>>(
                cu_indices[10], cu_neighbors[10],
                block_ranges[20], block_ranges[21],
                cu_dists[3], cu_updateds[3]);
        epoch_sssp_pull_gpu_block_min<<<4096, 64, 0, streams[9]>>>(
                cu_indices[9], cu_neighbors[9],
                block_ranges[18], block_ranges[19],
                cu_dists[3], cu_updateds[3]);
        
        CUDA_ERRCHK(cudaSetDevice(4));
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[15]>>>(
                cu_indices[15], cu_neighbors[15],
                block_ranges[30], block_ranges[31],
                cu_dists[4], cu_updateds[4]);
        epoch_sssp_pull_gpu_block_min<<<2048, 128, 0, streams[14]>>>(
                cu_indices[14], cu_neighbors[14],
                block_ranges[28], block_ranges[29],
                cu_dists[4], cu_updateds[4]);
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[13]>>>(
                cu_indices[13], cu_neighbors[13],
                block_ranges[26], block_ranges[27],
                cu_dists[4], cu_updateds[4]);
        epoch_sssp_pull_gpu_block_min<<<2048, 128, 0, streams[12]>>>(
                cu_indices[12], cu_neighbors[12],
                block_ranges[24], block_ranges[25],
                cu_dists[4], cu_updateds[4]);
        
        CUDA_ERRCHK(cudaSetDevice(5));
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[17]>>>(
                cu_indices[17], cu_neighbors[17],
                block_ranges[34], block_ranges[35],
                cu_dists[5], cu_updateds[5]);
        epoch_sssp_pull_gpu_block_min<<<4096, 64, 0, streams[16]>>>(
                cu_indices[16], cu_neighbors[16],
                block_ranges[32], block_ranges[33],
                cu_dists[5], cu_updateds[5]);
        
        CUDA_ERRCHK(cudaSetDevice(6));
        epoch_sssp_pull_gpu_warp_min<<<256, 1024, 0, streams[18]>>>(
                cu_indices[18], cu_neighbors[18],
                block_ranges[36], block_ranges[37],
                cu_dists[6], cu_updateds[6]);
        
        CUDA_ERRCHK(cudaSetDevice(7));
        epoch_sssp_pull_gpu_one_to_one<<<256, 1024, 0, streams[19]>>>(
                cu_indices[19], cu_neighbors[19],
                block_ranges[38], block_ranges[39],
                cu_dists[7], cu_updateds[7]);

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
            gpu_butterfly_P2P(seg_ranges, cu_dists); // Not implmented if INTERLEAVE=true.
        }
        epochs++;
    }
    // Copy GPU distances back to host.
    CUDA_ERRCHK(cudaSetDevice(0))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[13], cu_dists[0] + seg_ranges[13],
        (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[0], cu_dists[0] + seg_ranges[0],
        (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(1))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[18], cu_dists[1] + seg_ranges[18],
        (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[7], cu_dists[1] + seg_ranges[7],
        (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(2))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[23], cu_dists[2] + seg_ranges[23],
        (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[14], cu_dists[2] + seg_ranges[14],
        (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(3))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[28], cu_dists[3] + seg_ranges[28],
        (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[19], cu_dists[3] + seg_ranges[19],
        (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(4))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[24], cu_dists[4] + seg_ranges[24],
        (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(5))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[29], cu_dists[5] + seg_ranges[29],
        (seg_ranges[33] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(6))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[33], cu_dists[6] + seg_ranges[33],
        (seg_ranges[35] - seg_ranges[33]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
    CUDA_ERRCHK(cudaSetDevice(7))
    CUDA_ERRCHK(cudaMemcpyAsync(
        dist + seg_ranges[35], cu_dists[7] + seg_ranges[35],
        (seg_ranges[36] - seg_ranges[35]) * sizeof(weight_t), cudaMemcpyDeviceToHost));
    
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

/**
 * Butterfly GPU P2P transfer.
 */
void gpu_butterfly_P2P(nid_t *seg_ranges, weight_t **cu_dists) {
    // Butterfly Iteration 0
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[13], cu_dists[0] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[0], cu_dists[0] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[18], cu_dists[1] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[7], cu_dists[1] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[23], cu_dists[2] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[14], cu_dists[2] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[28], cu_dists[3] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[19], cu_dists[3] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[24], cu_dists[4] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[29], cu_dists[5] + seg_ranges[29],
            (seg_ranges[33] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[33], cu_dists[6] + seg_ranges[33],
            (seg_ranges[35] - seg_ranges[33]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[35], cu_dists[7] + seg_ranges[35],
            (seg_ranges[36] - seg_ranges[35]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    
    // Butterfly Iteration 1
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[13], cu_dists[0] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[0], cu_dists[0] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[18], cu_dists[0] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[7], cu_dists[0] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[13], cu_dists[1] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[0], cu_dists[1] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[18], cu_dists[1] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[7], cu_dists[1] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[23], cu_dists[2] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[14], cu_dists[2] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[28], cu_dists[2] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[19], cu_dists[2] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[23], cu_dists[3] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[14], cu_dists[3] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[28], cu_dists[3] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[19], cu_dists[3] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[24], cu_dists[4] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[29], cu_dists[4] + seg_ranges[29],
            (seg_ranges[33] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[24], cu_dists[5] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[29], cu_dists[5] + seg_ranges[29],
            (seg_ranges[33] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[33], cu_dists[6] + seg_ranges[33],
            (seg_ranges[36] - seg_ranges[33]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[33], cu_dists[7] + seg_ranges[33],
            (seg_ranges[36] - seg_ranges[33]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    
    // Butterfly Iteration 2
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[13], cu_dists[0] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[0], cu_dists[0] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[18], cu_dists[0] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[7], cu_dists[0] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[23], cu_dists[0] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[14], cu_dists[0] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[28], cu_dists[0] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[4] + seg_ranges[19], cu_dists[0] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[13], cu_dists[1] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[0], cu_dists[1] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[18], cu_dists[1] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[7], cu_dists[1] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[23], cu_dists[1] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[14], cu_dists[1] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[28], cu_dists[1] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[5] + seg_ranges[19], cu_dists[1] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[13], cu_dists[2] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[0], cu_dists[2] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[18], cu_dists[2] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[7], cu_dists[2] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[23], cu_dists[2] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[14], cu_dists[2] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[28], cu_dists[2] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[6] + seg_ranges[19], cu_dists[2] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[13], cu_dists[3] + seg_ranges[13],
            (seg_ranges[14] - seg_ranges[13]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[0], cu_dists[3] + seg_ranges[0],
            (seg_ranges[7] - seg_ranges[0]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[18], cu_dists[3] + seg_ranges[18],
            (seg_ranges[19] - seg_ranges[18]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[7], cu_dists[3] + seg_ranges[7],
            (seg_ranges[13] - seg_ranges[7]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[23], cu_dists[3] + seg_ranges[23],
            (seg_ranges[24] - seg_ranges[23]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[14], cu_dists[3] + seg_ranges[14],
            (seg_ranges[18] - seg_ranges[14]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[28], cu_dists[3] + seg_ranges[28],
            (seg_ranges[29] - seg_ranges[28]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[7] + seg_ranges[19], cu_dists[3] + seg_ranges[19],
            (seg_ranges[23] - seg_ranges[19]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[24], cu_dists[4] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[0] + seg_ranges[29], cu_dists[4] + seg_ranges[29],
            (seg_ranges[36] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[24], cu_dists[5] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[1] + seg_ranges[29], cu_dists[5] + seg_ranges[29],
            (seg_ranges[36] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[24], cu_dists[6] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[2] + seg_ranges[29], cu_dists[6] + seg_ranges[29],
            (seg_ranges[36] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[24], cu_dists[7] + seg_ranges[24],
            (seg_ranges[28] - seg_ranges[24]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_dists[3] + seg_ranges[29], cu_dists[7] + seg_ranges[29],
            (seg_ranges[36] - seg_ranges[29]) * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    
}

#endif // SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH