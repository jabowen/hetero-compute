/**
 * GPU implementations of PR pull kernels.
 */

#include <iostream>
#include <omp.h> 
#include <ostream>
#include <vector>

#include "../../src/kernels/kernel_types.cuh"
#include "../../src/cuda.cuh"
#include "../../src/devices.h"
#include "../../src/graph.cuh"
#include "../../src/util.h"

__global__
void test(
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

