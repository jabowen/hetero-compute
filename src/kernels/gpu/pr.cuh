/**
 * GPU implementations of PR pull kernels.
 */

#ifndef SRC_KERNELS_GPU__KERNEL_PR_CUH
#define SRC_KERNELS_GPU__KERNEL_PR_CUH

#include <iostream>
#include <omp.h> 
#include <ostream>
#include <vector>

#include "../kernel_types.cuh"
#include "../../cuda.cuh"
#include "../../devices.h"
#include "../../graph.cuh"
#include "../../util.h"

/** Forward decl. */
__global__ 
void epoch_pr_pull_gpu_one_to_one(
        const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *score, 
        nid_t *updated, int numNodes, offset_t *degrees);

/*****************************************************************************
 ***** SSSP Kernel ***********************************************************
 *****************************************************************************/

/**
 * Runs SSSP kernel on GPU. Synchronization occurs in serial.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- gpu epoch kernel.
 *   - init_score   <- initial score array.
 *   - ret_score     <- pointer to the address of the return score array.
 *   - block_count  <- (optional) number of blocks.
 *   - thread_count <- (optional) number of threads.
 * Returns:
 *   Execution time in milliseconds.
 */
double pr_pull_gpu(
        const CSRWGraph &g, pr_gpu_epoch_func epoch_kernel, 
        weight_t *init_score, weight_t ** const ret_score, 
        int block_count = 64, int thread_count = 1024
) {
    CONDCHK(epoch_kernel != epoch_pr_pull_gpu_one_to_one 
                and thread_count % 32 != 0, 
            "thread count must be divisible by 32");

    // Copy graph.
    offset_t *cu_index      = nullptr;
    wnode_t  *cu_neighbors  = nullptr;
    size_t   index_size     = (g.num_nodes + 1) * sizeof(offset_t);
    size_t   neighbors_size = g.num_edges * sizeof(wnode_t);
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, g.index, index_size, 
            cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g.neighbors, neighbors_size, 
            cudaMemcpyHostToDevice));
     
    //degrees
    offset_t *cu_degrees      = nullptr;
    offset_t *degrees = new offset_t[g.num_nodes];
    for(int i=0; i<g.num_nodes; i++){
        degrees[i]=g.get_degree(i);
    }
    size_t deg_size = g.num_nodes * sizeof(offset_t);
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_degrees, deg_size));
    CUDA_ERRCHK(cudaMemcpy(cu_degrees, degrees, deg_size,
	    cudaMemcpyHostToDevice));

    // Score
    weight_t *cu_score = nullptr;
    size_t score_size = g.num_nodes * sizeof(weight_t);
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_score, score_size));
    CUDA_ERRCHK(cudaMemcpy(cu_score, init_score, score_size, 
            cudaMemcpyHostToDevice));

    // Update counter.
    nid_t updated     = 1;
    nid_t *cu_updated = nullptr;
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_updated, sizeof(nid_t)));

    // Start kernel!
    Timer timer; timer.Start();
    int iters=0;
    while (updated != 0) {
        CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

        (*epoch_kernel)<<<block_count, thread_count>>>(cu_index, 
                cu_neighbors, 0, g.num_nodes, cu_score, cu_updated, 
		g.num_nodes, cu_degrees);

        CUDA_ERRCHK(cudaMemcpy(&updated, cu_updated, sizeof(nid_t),
                cudaMemcpyDeviceToHost));
	iters++;
	if(iters>200){
	    break;
	}
    }
    timer.Stop();

    // Copy scores.
    *ret_score = new weight_t[g.num_nodes];
    CUDA_ERRCHK(cudaMemcpy(*ret_score, cu_score, score_size, 
                cudaMemcpyDeviceToHost));

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_updated));
    CUDA_ERRCHK(cudaFree(cu_score));
    CUDA_ERRCHK(cudaFree(cu_degrees));

    return timer.Millisecs();
}

/******************************************************************************
 ***** Epoch Kernels **********************************************************
 ******************************************************************************/

/**
 * Runs PR pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each thread is assigned to a single node.
 *
 * Parameters:
 *   - index     <- graph index where index 0 = start_id, 1 = start_id + 1,
 *                  ... , (end_id - start_id - 1) = end_id - 1.
 *   - neighbors <- graph neighbors corresponding to the indexed nodes.
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - score      <- input score and output score computed this epoch
 *                  including nodes that \notin [start_id, end_id).
 *   - updated   <- global counter on number of nodes updated.
 *   - numNodes  <- the number of nodes
 *   - degrees   <- the degree of each node
 */
__global__ 
void epoch_pr_pull_gpu_one_to_one(
        const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, 
        weight_t *score, nid_t *updated, int numNodes, offset_t* degrees
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    float kDamp=.85;
    int local_updated = 0;
    float epsilon=.00000005;
    weight_t base_score=(1.0f - kDamp) / numNodes;

    for (nid_t nid = start_id + tid; nid < end_id; nid += num_threads) {
	weight_t incoming_total=0;

        // Find sum of the scores.
        nid_t index_id = nid - start_id;
        for (offset_t i = index[index_id]; i < index[index_id + 1]; i++) {
	    nid_t v=neighbors[i].v;
            incoming_total+=score[v]/degrees[v];
	}
	weight_t new_score=base_score+kDamp*incoming_total;

        // Update score if applicable.
        if (abs(new_score-score[nid])>epsilon) {
            score[nid] = new_score;
            local_updated++;
        }
    }

    // Push update count.
    atomicAdd(updated, local_updated);
}

/**
 * Runs PR pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each warp is assigned to a single node. To compute sum of score, a warp-level
 * sum is executed.
 *
 * Conditions:
 *   - blockDim.x % warpSize == 0 (i.e., number of threads per block is some 
 *                                 multiple of the warp size).
 * Parameters:
 *   - index     <- graph index where index 0 = start_id, 1 = start_id + 1,
 *                  ... , (end_id - start_id - 1) = end_id - 1.
 *   - neighbors <- graph neighbors corresponding to the indexed nodes.
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - score     <- input score and output scores computed this epoch
 *                  including nodes that \notin [start_id, end_id).
 *   - updated   <- global counter on number of nodes updated.
 *   - numNodes  <- the number of nodes
 *   - degrees   <- the degree of each node
 */
__global__ 
void epoch_pr_pull_gpu_warp_red(
        const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, 
        weight_t *score, nid_t *updated, int numNodes, offset_t* degrees
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid & (warpSize - 1); // ID within a warp.
    int num_threads = gridDim.x * blockDim.x;

    float kDamp=.85;
    float epsilon=.00000005;
    weight_t base_score=(1.0f - kDamp) / numNodes;

    nid_t local_updated = 0;

    for (nid_t nid = start_id + tid / warpSize; nid < end_id; 
            nid += (num_threads / warpSize)
    ) {
        weight_t incoming_total=0;

        // Find sum of score.
        nid_t index_id = nid - start_id;
        for (offset_t i = index[index_id] + warpid; i < index[index_id + 1]; 
                i += warpSize
        ) {
	    nid_t v=neighbors[i].v;
            incoming_total+=score[v]/degrees[v];
        }

        weight_t new_score = base_score+kDamp*warp_sum(incoming_total);

        // Update score if applicable.
	if (warpid==0 && abs(new_score-score[nid])>epsilon) {
	    score[nid] = new_score;
	    local_updated++;
	}
	
    }
    // Push update count.
    if (warpid == 0){
        atomicAdd(updated, local_updated);
    }
}

/**
 * Runs PR pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each block is assigned to a single node. To compute sum of score, a 
 * block-level sum is executed.
 *
 * Conditions:
 *   - warpSize == 32             
 *   - blockDim.x % warpSize == 0 (i.e., number of threads per block is some 
 *                                 multiple of the warp size)
     - thread count % warpSize == 0
 * Parameters:
 *   - index     <- graph index where index 0 = start_id, 1 = start_id + 1,
 *                  ... , (end_id - start_id - 1) = end_id - 1.
 *   - neighbors <- graph neighbors corresponding to the indexed nodes.
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - score      <- input score and output scores computed this epoch
 *                  including nodes that \notin [start_id, end_id).
 *   - updated   <- global counter on number of nodes updated.
 *   - numNodes  <- the number of nodes
 *   - degrees   <- the degree of each node
 */
__global__
void epoch_pr_pull_gpu_block_red(
        const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id,
        weight_t *score, nid_t *updated, int numNodes, offset_t* degrees
) {
    __shared__ weight_t block_score[32];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid = tid & (warpSize - 1); // ID within a warp.
    
    // Initialize block scores.
    if (threadIdx.x / warpSize == 0)
        block_score[warpid] =0;

    float kDamp=.85;
    float epsilon=0.00000005;
    weight_t base_score=(1.0f - kDamp) / numNodes;

    nid_t local_updated = 0;

    for (nid_t nid = start_id + blockIdx.x; nid < end_id; nid += gridDim.x) {
	weight_t incoming_total=0;

        // Find sum of scores.
        nid_t index_id = nid - start_id;
        for (offset_t i = index[index_id] + threadIdx.x; 
                i < index[index_id + 1]; i += blockDim.x
        ) {
	    nid_t v=neighbors[i].v;
	    incoming_total+=score[v]/degrees[v];
        }

        // Warp-level sum.
        weight_t new_score = warp_sum(incoming_total);
        if (warpid == 0) { block_score[threadIdx.x / warpSize] = new_score; }

        // Block level sum (using warp sum).
        __syncthreads();
        if (threadIdx.x / warpSize == 0) { // If first warp.
            new_score = block_score[warpid];
            // TODO: optimize this to only use the necssary number of shuffles.
            new_score = base_score+kDamp*warp_sum(new_score);
        }

        // Update scores if applicable.
        if (threadIdx.x == 0 and abs(new_score-score[nid])>epsilon) {
            score[nid] = new_score;
            local_updated++;
        }
    }

    // Push update count.
    if (threadIdx.x == 0)
        atomicAdd(updated, local_updated);
}

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

/** Identifier for epoch kernels. */
enum class PRGPU {
    one_to_one, warp_red, block_red, undefined
};

/** List of kernels available (no good iterator for enum classes). */
std::vector<PRGPU> pr_gpu_kernels = {
    PRGPU::one_to_one, PRGPU::warp_red, PRGPU::block_red
};

std::vector<PRGPU> get_kernels(UNUSED PRGPU unused) {
    // Using hack to overload function by return type.
    return pr_gpu_kernels;
}

/** 
 * Convert epoch kernel ID to its representation name (not as human-readable). 
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_repr(PRGPU ker) {
    switch (ker) {
        case PRGPU::one_to_one: return "pr_gpu_onetoone";
        case PRGPU::warp_red:   return "pr_gpu_warp_red";
        case PRGPU::block_red:  return "pr_gpu_block_red";
        case PRGPU::undefined:  
        default:                  return "";
    }
}

/** 
 * Convert epoch kernel ID to its human-readable name. 
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_string(PRGPU ker) {
    switch (ker) {
        case PRGPU::one_to_one: return "PR GPU one-to-one";
        case PRGPU::warp_red:   return "PR GPU warp-red";
        case PRGPU::block_red:  return "PR GPU block-red";
        case PRGPU::undefined:  
        default:                  return "undefined PR GPU kernel";
    }
}

/**
 * Convert epoch kernel ID to kernel function pointer.
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel function pointer.
 */
pr_gpu_epoch_func get_kernel(PRGPU ker) {
    switch (ker) {
        case PRGPU::one_to_one: return epoch_pr_pull_gpu_one_to_one;
        case PRGPU::warp_red:   return epoch_pr_pull_gpu_warp_red;
        case PRGPU::block_red:  return epoch_pr_pull_gpu_block_red;
        case PRGPU::undefined:  
        default:                  return nullptr;
    }
}

std::ostream &operator<<(std::ostream &os, PRGPU ker) {
    os << to_string(ker);
    return os;
}

#endif // SRC_KERNELS_GPU__KERNEL_SSSP_CUH
