/**
 * Heterogeneous implementation of the PR pull kernel.
 * This is generated by util/scheduler/scheduler/kernelgen/pr_hetero.py.
 */

#ifndef SRC_KERNELS_HETEROGENEOUS__PR_CUH
#define SRC_KERNELS_HETEROGENEOUS__PR_CUH

#include <omp.h>
#include <vector>

#include "../kernel_types.cuh"
#include "../cpu/pr.cuh"
#include "../gpu/pr.cuh"
#include "../../cuda.cuh"
#include "../../graph.cuh"
#include "../../util.h"

constexpr int num_gpus_pr = 1;

/** Forward decl. */
void gpu_butterfly_P2P_pr(nid_t *seg_ranges, weight_t **cu_scores, 
        cudaStream_t *memcpy_streams);

/**
 * Runs PR kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. 
 * Configuration:
 *   - 1x Intel i7-9700K
 *   - 1x NVIDIA Quadro RTX 4000
 *
 * Parameters:
 *   - g         <- graph.
 *   - init_score <- initial score array.
 *   - ret_score  <- pointer to the address of the return score array.
 * Returns:
 *   Execution time in milliseconds.
 */
double pr_pull_heterogeneous(const CSRWGraph &g, 
        const weight_t *init_score, weight_t ** const ret_score
) {
    // Configuration.
    constexpr int num_blocks   = 1;
    constexpr int num_segments = 8;
    
    // Copy graph.
    nid_t *seg_ranges = compute_equal_edge_ranges(g, num_segments);
    
    /// Block ranges to reduce irregular memory acceses.
    constexpr int gpu_blocks[] = {0, 1};
    nid_t block_ranges[num_blocks * 2];

    block_ranges[0] = seg_ranges[0]; // Block 0 Start 0
    block_ranges[1] = seg_ranges[4]; // Block 0 End 4 (excl.)

    //degrees
    offset_t *cu_degrees      = nullptr;

    size_t deg_size = g.num_nodes * sizeof(offset_t);
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_degrees, deg_size));
    for(int i=0; i<g.num_nodes; i++){
        cu_degrees[i]=g.get_degree(i);
    }

    /// Actual graphs on GPU memory.
    offset_t *cu_indices[num_blocks];
    wnode_t  *cu_neighbors[num_blocks];

    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++) 
            copy_subgraph_to_device(g,
                    &cu_indices[block], &cu_neighbors[block],
                    block_ranges[2 * block], block_ranges[2 * block + 1]);
    }

    // Initialize memcopy streams.
    // idx = from_gpu * num_gpus_pr + to_gpu;
    cudaStream_t memcpy_streams[num_gpus_pr * num_gpus_pr];
    for (int from = 0; from < num_gpus_pr; from++) {
        CUDA_ERRCHK(cudaSetDevice(from));
        for (int to = 0; to < num_gpus_pr; to++)
            CUDA_ERRCHK(cudaStreamCreate(&memcpy_streams[from * num_gpus_pr + to]));
    }

    // score.
    size_t   score_size = g.num_nodes * sizeof(weight_t);
    /*weight_t *score     = nullptr; 

    /// CPU score.
    CUDA_ERRCHK(cudaMallocHost((void **) &score, score_size));
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        score[i] = init_score[i]; 
    */
     
    /// GPU scores.
    weight_t *cu_scores1[num_gpus_pr];
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMallocManaged((void **) &cu_scores1[gpu], score_size));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_scores1[gpu], init_score, score_size,
            cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr]));
    }
	weight_t *cu_scores2[num_gpus_pr];
	for (int gpu = 0; gpu < num_gpus_pr; gpu++) {        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMallocManaged((void **) &cu_scores2[gpu], score_size));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_scores2[gpu], init_score, score_size,
            cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr]));
    }
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr]));
    }

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updateds[num_gpus_pr];
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_updateds[gpu], 
                sizeof(nid_t)));
    }

    // Create compute streams and markers.
    cudaStream_t compute_streams[num_blocks]; // Streams for compute.
    cudaEvent_t  compute_markers[num_blocks]; // Compute complete indicators.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++) {
            CUDA_ERRCHK(cudaStreamCreate(&compute_streams[b]));
            CUDA_ERRCHK(cudaEventCreate(&compute_markers[b]));
        }
    }

    // Get init vertex.
    // TODO: add this as a parameter.
    nid_t start;
    for (nid_t i = 0; i < g.num_nodes; i++)
        if (init_score[i] != 1.0f/g.num_nodes) start = i;

    // Start kernel!
    Timer timer; timer.Start();
    int epochs = 0;

    /*
    // Push for the first iteration.
    // TODO: implement push for more than one epoch. Requires parallel queue.
    for (wnode_t nei : g.get_neighbors(start)) {
        if (nei.v == start) continue;

        score[nei.v] = nei.w;       
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                cu_scores[gpu] + nei.v, score + nei.v,
                sizeof(weight_t), cudaMemcpyHostToDevice));
        }
    }
    epochs++;
    */

    int iters=0;
    while (updated != 0) {
        if(iters>200){
            break;
        }
        iters++;
        // Reset update counters.
        updated = cpu_updated = 0;          
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemsetAsync(cu_updateds[gpu], 0, 
                    sizeof(nid_t)));
        }

        // Launch GPU epoch kernels.
        // Implicit CUDA device synchronize at the start of kernels.
        CUDA_ERRCHK(cudaSetDevice(0));
        epoch_pr_pull_gpu_block_red<<<512, 512, 0, compute_streams[0]>>>(
                cu_indices[0], cu_neighbors[0],
                block_ranges[0], block_ranges[1],
                cu_scores1[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[0], compute_streams[0]));
        /*
        CUDA_ERRCHK(cudaMemcpyAsync(
                score + block_ranges[0], cu_scores[0] + block_ranges[0],
                (block_ranges[1] - block_ranges[0]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[0]));
        */

        // Launch CPU epoch kernels.
        #pragma omp parallel
        {
            cudaDeviceSynchronize();
            epoch_pr_pull_cpu_one_to_one(g, cu_scores2[0], 
                    seg_ranges[4], seg_ranges[8],
                    omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
        }

        // Sync compute streams.
        for (int b = 0; b < num_blocks; b++)
            CUDA_ERRCHK(cudaEventSynchronize(compute_markers[b]));

        // Synchronize updates.
        nid_t gpu_updateds[num_gpus_pr];
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                    &gpu_updateds[gpu], cu_updateds[gpu],  sizeof(nid_t), 
                    cudaMemcpyDeviceToHost, memcpy_streams[gpu * num_gpus_pr + gpu]));
        }
        updated += cpu_updated;

        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr + gpu]));
            updated += gpu_updateds[gpu];
        }

        // Only update GPU scores if another epoch will be run.
        if (updated != 0) {
            // Copy CPU scores to all GPUs.
            /*for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
                CUDA_ERRCHK(cudaMemcpyAsync(
                    cu_scores[gpu] + seg_ranges[4],
                    score + seg_ranges[4],
                    (seg_ranges[8] - seg_ranges[4]) * sizeof(weight_t),
                    cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr + gpu]));
            }*/
			
			weight_t *cu_temp[num_gpus_pr];
			temp=cu_scores[1];
			cu_scores[1]=cu_scores[2];
			cu_scores[2]=temp;
			free(temp);

            // Copy GPU scores peer-to-peer.
            // Not implmented if INTERLEAVE=true.
            gpu_butterfly_P2P_pr(seg_ranges, cu_scores, memcpy_streams); 

            // Synchronize HtoD async calls.
            for (int gpu = 0; gpu < num_gpus_pr; gpu++)
                CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr + gpu]));
		    
        }

        // Sync DtoH copies.
        for (int b = 0; b < num_blocks; b++)
            CUDA_ERRCHK(cudaStreamSynchronize(compute_streams[b]));
        

        
        epochs++;
    }
    
    timer.Stop();

    // Copy output.
    *ret_score = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        (*ret_score)[i] = cu_scores1[0][i];

    // Free streams.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++) {
            CUDA_ERRCHK(cudaStreamDestroy(compute_streams[b]));
            CUDA_ERRCHK(cudaEventDestroy(compute_markers[b]));
        }

        for (int to = 0; to < num_gpus_pr; to++)
            CUDA_ERRCHK(cudaStreamDestroy(memcpy_streams[gpu * num_gpus_pr + to]));
    }

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_degrees));
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaFree(cu_updateds[gpu]));
        CUDA_ERRCHK(cudaFree(cu_scores1[gpu]));
		CUDA_ERRCHK(cudaFree(cu_scores2[gpu]));
        
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++
        ) {
            CUDA_ERRCHK(cudaFree(cu_indices[block]));
            CUDA_ERRCHK(cudaFree(cu_neighbors[block]));
        }
    }
    //CUDA_ERRCHK(cudaFreeHost(score));
    delete[] seg_ranges;

    return timer.Millisecs();
}

/**
 * Enable peer access between all compatible GPUs.
 */
void enable_all_peer_access_pr() {
    int can_access_peer;
    for (int from = 0; from < num_gpus_pr; from++) {
        CUDA_ERRCHK(cudaSetDevice(from));

        for (int to = 0; to < num_gpus_pr; to++) {
            if (from == to) continue;

            CUDA_ERRCHK(cudaDeviceCanAccessPeer(&can_access_peer, from, to));
            if(can_access_peer) {
                CUDA_ERRCHK(cudaDeviceEnablePeerAccess(to, 0));
                std::cout << from << " " << to << " yes" << std::endl;
            } else {
                std::cout << from << " " << to << " no" << std::endl;
            }
        }
    }
}

/**
 * Butterfly GPU P2P transfer.
 */
void gpu_butterfly_P2P_pr(nid_t *seg_ranges, weight_t **cu_scores, 
    cudaStream_t *memcpy_streams
) {
    
}

#endif // SRC_KERNELS_HETEROGENEOUS__PR_CUH