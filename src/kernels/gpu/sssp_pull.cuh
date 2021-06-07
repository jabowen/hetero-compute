/**
 * GPU implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH
#define SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH

#include <iostream>
#include <omp.h> 

#include "../kernel_types.h"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../util.h"

/** Forward decl. */
__global__ 
void epoch_sssp_pull_gpu_naive(const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *dist, nid_t *updated
);

/******************************************************************************
 ***** SSSP Kernel ************************************************************
 ******************************************************************************/

/**
 * Runs SSSP kernel on GPU. Synchronization occurs in serial.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- gpu epoch kernel.
 *   - init_dist    <- initial distance array.
 *   - ret_dist     <- pointer to the address of the return distance array.
 *   - block_count  <- (optional) number of blocks.
 *   - thread_count <- (optional) number of threads.
 * Returns:
 *   Execution time in millisecods.
 */
double sssp_pull_gpu(const CSRWGraph &g, sssp_gpu_epoch_func epoch_kernel, 
        const weight_t *init_dist, weight_t **ret_dist, 
        int block_count = 64, int thread_count = 1024
) {
    CONDCHK(epoch_kernel != epoch_sssp_pull_gpu_naive and thread_count % 32 != 0, 
            "thread count must be divisible by 32")

    // Copy graph.
    offset_t *cu_index      = nullptr;
    wnode_t  *cu_neighbors  = nullptr;
    size_t   index_size     = g.num_nodes * sizeof(offset_t);
    size_t   neighbors_size = g.num_edges * sizeof(wnode_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMalloc((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, g.index, index_size, 
            cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g.neighbors, neighbors_size, 
            cudaMemcpyHostToDevice));
    
    // Distance.
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = init_dist[i];

    weight_t *cu_dist = nullptr;
    size_t dist_size = g.num_nodes * sizeof(weight_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, dist_size));

    // Update counter.
    nid_t updated     = 1;
    nid_t *cu_updated = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_updated, sizeof(nid_t)));
    CUDA_ERRCHK(cudaMemcpy(cu_dist, dist, dist_size, 
            cudaMemcpyHostToDevice));

    // Start kernel!
    Timer timer; timer.Start();
    while (updated != 0) {
        CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

        // Note: Must run with thread count % 32 == 0 since warp level
        //       synchronization could performed.
        (*epoch_kernel)<<<block_count, thread_count>>>(cu_index, 
                cu_neighbors, 0, g.num_nodes, cu_dist, cu_updated);

        CUDA_ERRCHK(cudaMemcpy(&updated, cu_updated, sizeof(nid_t),
                cudaMemcpyDeviceToHost));
    }
    timer.Stop();

    // Copy distances.
    CUDA_ERRCHK(cudaMemcpy(dist, cu_dist, dist_size, cudaMemcpyDeviceToHost));
    *ret_dist = dist;

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_updated));
    CUDA_ERRCHK(cudaFree(cu_dist));

    return timer.Millisecs();
}

/******************************************************************************
 ***** Epoch Kernels **********************************************************
 ******************************************************************************/

/**
 * Runs SSSP pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each block is assigned to a single node. To compute min distance, a 
 * block-level min is executed.
 *
 * Conditions:
 *   - blockDim.x % warpSize == 0 (i.e., number of threads per block is some 
 *                                 multiple of the warp size)
 *   - thread count == 1024       (TODO: make this variable with templating?)
 *   - warpSize == 32             (depends on thread count)
 * Parameters:
 *   - index     <- graph index returned by deconstruct_wgraph().
 *   - neighbors <- graph neighbors returned by deconstruct_wgraph().
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - dist      <- input distance and output distances computed this epoch.
 *   - updated   <- global counter on number of nodes updated.
 */
__global__
void epoch_sssp_pull_gpu_block_min(const offset_t *index,
        const wnode_t *neighbors, const nid_t start_id, const nid_t end_id,
        weight_t *dist, nid_t *updated
) {
    __shared__ weight_t block_dist[32];

    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid % warpSize; // ID within a warp.

    nid_t local_updated = 0;

    for (int nid = start_id + blockIdx.x; nid < end_id; nid += gridDim.x) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (int i = index[nid] + threadIdx.x; i < index[nid + 1];
                i += blockDim.x
        ) {
            weight_t prop_dist = dist[neighbors[i].v] + neighbors[i].w;
            new_dist = min(prop_dist, new_dist);
        }

        // Warp-level min.
        new_dist = warp_min(new_dist);
        if (warpid == 0) { block_dist[threadIdx.x / warpSize] = new_dist; }

        // Block level min (using warp min).
        __syncthreads();
        // If first warp.
        if (threadIdx.x / warpSize == 0) {
            new_dist = block_dist[warpid];
            new_dist = warp_min(new_dist);
        }

        // Update distance if applicable.
        if (threadIdx.x == 0 and new_dist != dist[nid]) {
            dist[nid] = new_dist;
            local_updated++;
        }
    }

    // Push update count.
    if (threadIdx.x == 0)
        atomicAdd(updated, local_updated);
}

/**
 * Runs SSSP pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each warp is assigned to a single node. To compute min distance, a warp-level
 * min is executed.
 *
 * Conditions:
 *   - blockDim.x % warpSize == 0 (i.e., number of threads per block is some 
 *                                 multiple of the warp size).
 * Parameters:
 *   - index     <- graph index returned by deconstruct_wgraph().
 *   - neighbors <- graph neighbors returned by deconstruct_wgraph().
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - dist      <- input distance and output distances computed this epoch.
 *   - updated   <- global counter on number of nodes updated.
 */
__global__ 
void epoch_sssp_pull_gpu_warp_min(const offset_t *index, 
        const wnode_t *neighbors, const nid_t start_id, 
        const nid_t end_id, weight_t *dist, nid_t *updated
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid % warpSize; // ID within a warp.
    int num_threads = gridDim.x * blockDim.x;

    nid_t local_updated = 0;

    for (int nid = start_id + tid / warpSize; nid < end_id; 
            nid += (num_threads / warpSize)
    ) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (int i = index[nid] + warpid; i < index[nid + 1]; i += warpSize) {
            weight_t prop_dist = dist[neighbors[i].v] + neighbors[i].w;
            new_dist = min(prop_dist, new_dist);
        }

        new_dist = warp_min(new_dist);

        // Update distance if applicable.
        if (warpid == 0 and new_dist != dist[nid]) {
            dist[nid] = new_dist;
            local_updated++;
        }
    }

    // Push update count.
    if (warpid == 0)
        atomicAdd(updated, local_updated);
}

/**
 * Runs SSSP pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each thread is assigned to a single node.
 *
 * Parameters:
 *   - index     <- graph index returned by deconstruct_wgraph().
 *   - neighbors <- graph neighbors returned by deconstruct_wgraph().
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - dist      <- input distance and output distances computed this epoch.
 *   - updated   <- global counter on number of nodes updated.
 */
__global__ 
void epoch_sssp_pull_gpu_naive(const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *dist, nid_t *updated
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    int local_updated = 0;

    for (int nid = start_id + tid; nid < end_id; nid += num_threads) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (int i = index[nid]; i < index[nid + 1]; i++) {
            weight_t prop_dist = dist[neighbors[i].v] + neighbors[i].w;
            new_dist = min(prop_dist, new_dist);
        }

        // Update distance if applicable.
        if (new_dist != dist[nid]) {
            dist[nid] = new_dist;
            local_updated++;
        }
    }

    // Push update count.
    atomicAdd(updated, local_updated);
}

#endif // SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH
