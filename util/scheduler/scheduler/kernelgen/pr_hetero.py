###############################################################################
# C++ Heterogeneous PR "compiler"                                           #
#   Generates a heterogeneous PR kernel given a particular schedule and a   #
#   particular hardware configuration (e.g., 1 CPU 1 GPU, 1 CPU 6 GPU, etc).  #
#   The schedule (kernels for each device) will be hardcoded but the size of  #
#   the input graph will be dynamic.                                          #
###############################################################################

import enum
from functools import reduce
from math import log
from typing import *
from scheduler.scheduler import (
    DeviceSchedule, 
    KernelSegment
)

# Interleave computation and memory transfers.
INTERLEAVE = False
# GPUs work on highest degree nodes first.
HIGH_DEGREE_FIRST = True

###############################################################################
##### Enums ###################################################################
###############################################################################

class PRKernels(enum.Enum):
    CPU_one_to_one = enum.auto()
    GPU_one_to_one = enum.auto()
    GPU_warp_red   = enum.auto()
    GPU_block_red  = enum.auto()

def to_epochkernel_funcname(kernel: PRKernels) -> str:
    func_names = {
        PRKernels.CPU_one_to_one: 'epoch_pr_pull_cpu_one_to_one',
        PRKernels.GPU_one_to_one: 'epoch_pr_pull_gpu_one_to_one',
        PRKernels.GPU_warp_red:   'epoch_pr_pull_gpu_warp_red',
        PRKernels.GPU_block_red:  'epoch_pr_pull_gpu_block_red',
    }
    return func_names[kernel]

def ker_to_string(kernel: PRKernels) -> str:
    string_names = {
        PRKernels.CPU_one_to_one: 'PR CPU one-to-one',
        PRKernels.GPU_one_to_one: 'PR GPU one-to-one',
        PRKernels.GPU_warp_red:   'PR GPU warp-red',
        PRKernels.GPU_block_red:  'PR GPU block-red',
    }
    return string_names[kernel]

class Kernel:
    def __init__(self, kerid: PRKernels, block_count=64, thread_count=64):
        self.kerid        = kerid
        self.block_count  = block_count
        self.thread_count = thread_count
        self.is_gpu       = kerid not in [PRKernels.CPU_one_to_one]

    def __repr__(self) -> str:
        return f'Kernel(kernel={ker_to_string(self.kerid)}, '\
            f'block_count={self.block_count}, '\
            f'thread_count={self.thread_count})'

def parse_kernel(kerstr: str) -> Kernel:
    """Returns corresponding kernel object based on string description."""
    # Figure out kernel.
    kerid = None
    for k in PRKernels:
        ks = ker_to_string(k)
        if kerstr[:len(ks)] == ks:
            kerid = k
            break

    kernel = Kernel(kerid)
    
    # If it's a GPU kernel, figure out block and thread counts.
    if kernel.is_gpu:
        bt_str              = kerstr[len(ker_to_string(k)) + 1:]
        b_str, t_str        = bt_str.split(' ')
        kernel.block_count  = int(b_str)
        kernel.thread_count = int(t_str)

    return kernel

###############################################################################
##### PR Heterogenous Generator #############################################
###############################################################################

def generate_pr_hetero_source_code(scheds: List[DeviceSchedule]) -> str:
    ###########################################################################
    ##    Configuration Information                                          ##
    ###########################################################################
    gpu_segments: List[List[KernelSegment]] = [
        devsched.schedule
        for devsched in scheds
        if is_gpu(devsched.device_name)
    ]
    gpu_prefix_sum = reduce(lambda acc, devsched: 
                                acc + [acc[-1] + len(devsched)],
                            gpu_segments[:-1], [0])
    gpu_contig_mem: List[List[List[int, int]]] = list()
    for segs in gpu_segments:
        contig_mem = [[seg.seg_start, seg.seg_end] for seg in segs]
        idx = 1
        while idx < len(contig_mem):
            # Merge if necessary.
            if contig_mem[idx - 1][1]  + 1 == contig_mem[idx][0]:
                contig_mem[idx - 1][1] = contig_mem[idx][1]
                del contig_mem[idx]
            else:
                idx += 1
        gpu_contig_mem.append(contig_mem)

    num_blocks   = sum(len(sched) for sched in gpu_segments)
    num_segments = max(kerseg.seg_end 
                       for devsched in scheds
                       for kerseg in devsched.schedule) + 1
    num_gpus_pr     = len(gpu_segments)
    has_cpu      = len(scheds) != num_gpus_pr

    cpu_name = ""
    gpu_name = ""
    for sched in scheds:
        if is_gpu(sched.device_name):
            gpu_name = sched.device_name
        else:
            cpu_name = sched.device_name

    ###########################################################################
    ##    Helper Generator Functions                                         ##
    ###########################################################################

    ###########################################################################
    ##### GPU Blocks ##########################################################
    ###########################################################################
    def generate_gpu_blocks() -> str:
        gpu_blocks = gpu_prefix_sum + \
            [gpu_prefix_sum[-1] + len(gpu_segments[-1])]
        gpu_blocks = [str(block) for block in gpu_blocks]
        return f'constexpr int gpu_blocks[] = {{{", ".join(gpu_blocks)}}};'

    ###########################################################################
    ##### Block Range Generation ##############################################
    ###########################################################################
    def generate_block_ranges() -> str:
        code = ''
        for devid, devsched in enumerate(gpu_segments):
            for kerid, kerseg in enumerate(devsched):
                idx = gpu_prefix_sum[devid] + kerid
                code += \
f"""
block_ranges[{2 * idx}] = seg_ranges[{kerseg.seg_start}]; // Block {idx} Start {kerseg.seg_start}
block_ranges[{2 * idx + 1}] = seg_ranges[{kerseg.seg_end + 1}]; // Block {idx} End {kerseg.seg_end + 1} (excl.)
""".strip() + '\n'
        return code.strip()

    ###########################################################################
    ##### GPU Kernel Launch ###################################################
    ###########################################################################
    def generate_gpu_kernel_launches() -> str:
        code = ''
        for devid, devsched in enumerate(gpu_segments):
            if devid != 0: code += '\n'
            code += f'CUDA_ERRCHK(cudaSetDevice({devid}));' + '\n'
            kernel_launches = list()
            for kerid, kerseg in enumerate(devsched):
                idx    = gpu_prefix_sum[devid] + kerid
                kernel = parse_kernel(kerseg.kernel_name)

                # Launch kernel.
                segcode = \
f"""
{to_epochkernel_funcname(kernel.kerid)}<<<{kernel.block_count}, {kernel.thread_count}, 0, compute_streams[{idx}]>>>(
        cu_indices[{idx}], cu_neighbors[{idx}],
        block_ranges[{2 * idx}], block_ranges[{2 * idx + 1}],
        cu_scores[{devid}], cu_updateds[{devid}], g.num_nodes, cu_degrees);
CUDA_ERRCHK(cudaEventRecord(compute_markers[{idx}], compute_streams[{idx}]));
""".strip()

                # Device to Host memcpy if needed.
                if has_cpu:
                    segcode += '\n' + \
f"""
CUDA_ERRCHK(cudaMemcpyAsync(
        score + block_ranges[{2 * idx}], cu_scores[{devid}] + block_ranges[{2 * idx}],
        (block_ranges[{2 * idx + 1}] - block_ranges[{2 * idx}]) * sizeof(weight_t),
        cudaMemcpyDeviceToHost, compute_streams[{idx}]));
""".strip()

                # Peer to Peer memcpy if needed.
                if INTERLEAVE and num_gpus_pr > 1:
                    segcode += '\n' + \
f"""
for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
    if (gpu == {devid}) continue;

    CUDA_ERRCHK(cudaStreamWaitEvent(memcpy_streams[{devid} * num_gpus_pr + gpu], 
            compute_markers[{idx}], 0));
    CUDA_ERRCHK(cudaMemcpyAsync(
            cu_scores[gpu] + block_ranges[{2 * idx}], cu_scores[{devid}] + block_ranges[{2  * idx}],
            (block_ranges[{2 * idx + 1}] - block_ranges[{2 * idx}]) * sizeof(weight_t),
            cudaMemcpyDeviceToDevice, memcpy_streams[{devid} * num_gpus_pr + gpu]));
}}
""".strip()
                kernel_launches.append(segcode)
            
            if HIGH_DEGREE_FIRST:
                code += '\n'.join(kernel_launches) + '\n'
            else:
                code += '\n'.join(reversed(kernel_launches)) + '\n'

        return code.strip()

    ###########################################################################
    ##### CPU Kernel Launch ###################################################
    ###########################################################################
    def generate_cpu_kernel_launches() -> str:
        if not has_cpu: return ''

        code = ''
        cpu_schedule = next(filter(lambda devsched: 
                                      not is_gpu(devsched.device_name),
                                   scheds))
        for kerseg in cpu_schedule.schedule:
            kernel = parse_kernel(kerseg.kernel_name)
            code += \
f""" 
#pragma omp parallel
{{
    {to_epochkernel_funcname(kernel.kerid)}(g, score, 
            seg_ranges[{kerseg.seg_start}], seg_ranges[{kerseg.seg_end + 1}],
            omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
}}
""".strip() + '\n'
        return code.strip()

    ###########################################################################
    ##### Host to Device Sync #################################################
    ###########################################################################
    def generate_score_HtoD_synchronize() -> str:
        if not has_cpu: return ''
        code = ''
        cpu_schedule = next(filter(lambda devsched: 
                                      not is_gpu(devsched.device_name),
                                   scheds))
        for kerseg in cpu_schedule.schedule:
            code += \
f""" 
for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
    CUDA_ERRCHK(cudaMemcpyAsync(
        cu_scores[gpu] + seg_ranges[{kerseg.seg_start}],
        score + seg_ranges[{kerseg.seg_start}],
        (seg_ranges[{kerseg.seg_end + 1}] - seg_ranges[{kerseg.seg_start}]) * sizeof(weight_t),
        cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr + gpu]));
}}
""".strip() + '\n'
        return code.strip()

    def generate_score_HtoD_synchronize_sync() -> str:
        if not has_cpu: return ''
        code = \
f""" 
for (int gpu = 0; gpu < num_gpus_pr; gpu++)
    CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr + gpu]));
""".strip()
        return code

    ###########################################################################
    ##### P2P Butterfly Sync ##################################################
    ###########################################################################
    def generate_butterfly_transfer() -> str:
        """Performs a butterfly pattern."""
        if num_gpus_pr <= 1 or INTERLEAVE: return ''

        # Only for base 2 for now.
        assert(log(num_gpus_pr, 2).is_integer()) 

        chunks = gpu_contig_mem

        def merge(segs1, segs2):
            merged = list()
            idx1 = 0
            idx2 = 0
            while idx1 < len(segs1) and idx2 < len(segs2):
                s1, e1 = segs1[idx1]
                s2, e2 = segs2[idx2]
                if s1 < s2:
                    merged.append([s1, e1])
                    idx1 += 1
                else:
                    merged.append([s2, e2])
                    idx2 += 1
            while idx1 < len(segs1):
                s, e = segs1[idx1]
                merged.append([s, e])
                idx1 += 1
            while idx2 < len(segs2):
                s, e = segs2[idx2]
                merged.append([s, e])
                idx2 += 1
            return merged

        def contiguify(segs):
            idx = 1
            while idx < len(segs):
                if segs[idx - 1][1] + 1 == segs[idx][0]:
                    segs[idx - 1][1] = segs[idx][1]
                    del segs[idx]
                else:
                    idx += 1
            return segs

        code = ''
        for i in range(int(log(num_gpus_pr, 2))):
            # Determine new blocks.
            if i != 0:
                chunks = [contiguify(merge(l, r))
                          for l, r in zip(chunks[::2], chunks[1::2])]
            if i != 0: code += '\n'
            code += f'// Butterfly Iteration {i}' + '\n'

            streams: List[Tuple[int, int]] = list()
            prefix = 0
            block = 2 ** i
            # Mem copies.
            for chunk_id, chunk in enumerate(chunks):
                stride = (1 if chunk_id % 2 == 0 else -1) * 2 ** i
                for gpu in range(block):
                    from_gpu = prefix + gpu
                    to_gpu = (from_gpu + stride) % num_gpus_pr
                    streams.append((from_gpu, to_gpu))

                    for seg in chunk:
                        start, end = seg
                        code += \
f"""
CUDA_ERRCHK(cudaMemcpyAsync(
    cu_scores[{to_gpu}] + seg_ranges[{start}], cu_scores[{from_gpu}] + seg_ranges[{start}],
    (seg_ranges[{end + 1}] - seg_ranges[{start}]) * sizeof(weight_t), 
    cudaMemcpyDeviceToDevice, memcpy_streams[{from_gpu} * num_gpus_pr + {to_gpu}]));
""".strip() + '\n'
                
                prefix += block

            code += '\n'

            # Synchronize streams.
            for from_gpu, to_gpu in streams:
                code += \
f"""
CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[{from_gpu} * num_gpus_pr + {to_gpu}]));
""".strip() + '\n'

        return code

    ###########################################################################
    ##### Device to Host Sync #################################################
    ###########################################################################
    def generate_score_DtoH_synchronize() -> str:
        if has_cpu: return ''

        code = '// Copy GPU scores back to host.' + '\n'
        for devid, segs in enumerate(gpu_contig_mem):
            if devid != 0: code += '\n'
            code += f'CUDA_ERRCHK(cudaSetDevice({devid}))' + '\n'
            for seg in segs:
                code += \
f"""
CUDA_ERRCHK(cudaMemcpyAsync(
    score + seg_ranges[{seg[0]}], cu_scores[{devid}] + seg_ranges[{seg[0]}],
    (seg_ranges[{seg[1] + 1}] - seg_ranges[{seg[0]}]) * sizeof(weight_t), 
    cudaMemcpyDeviceToHost, memcpy_streams[{devid} * num_gpus_pr + {devid}]));
""".strip() + '\n'
        
        code += \
"""
// Wait for memops to complete.
for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
    CUDA_ERRCHK(cudaSetDevice(gpu));
    CUDA_ERRCHK(cudaDeviceSynchronize());
}
""".strip() + '\n'
        
        return code

    ###########################################################################
    ##### Interleave Sync #####################################################
    ###########################################################################
    def generate_interleave_synchronize() -> str:
        if not INTERLEAVE or num_gpus_pr == 1: return ''
        
        code = \
"""
// Synchronize interleave streams.
for (int i = 0; i < num_gpus_pr * num_gpus; i++)
    CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[i]));    
"""
        return code.strip() + '\n'

    def generate_interleave_DtoH_synchronize() -> str:
        if not has_cpu: return ''
        code = \
"""
// Sync DtoH copies.
for (int b = 0; b < num_blocks; b++)
    CUDA_ERRCHK(cudaStreamSynchronize(compute_streams[b]));
"""
        return code.strip() + '\n'

    ###########################################################################
    ##    Main Source Code Generation                                        ##
    ###########################################################################
    source_code = \
f"""
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

constexpr int num_gpus_pr = {num_gpus_pr};

/** Forward decl. */
void gpu_butterfly_P2P_pr(nid_t *seg_ranges, weight_t **cu_scores, 
        cudaStream_t *memcpy_streams);

/**
 * Runs PR kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. 
 * Configuration:
 *   - 1x {cpu_name}
 *   - {num_gpus_pr}x {gpu_name}
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
) {{
    // Configuration.
    constexpr int num_blocks   = {num_blocks};
    constexpr int num_segments = {num_segments};
    
    // Copy graph.
    nid_t *seg_ranges = compute_equal_edge_ranges(g, num_segments);
    
    /// Block ranges to reduce irregular memory acceses.
    {indent_after(generate_gpu_blocks())}
    nid_t block_ranges[num_blocks * 2];

    {indent_after(generate_block_ranges())}

    //degrees
    offset_t *cu_degrees      = nullptr;
    offset_t *degrees = new offset_t[g.num_nodes];
    for(int i=0; i<g.num_nodes; i++){{
        degrees[i]=g.get_degree(i);
    }}
    size_t deg_size = g.num_nodes * sizeof(offset_t);
    CUDA_ERRCHK(cudaMallocManaged((void **) &cu_degrees, deg_size));
    CUDA_ERRCHK(cudaMemcpy(cu_degrees, degrees, deg_size,
            cudaMemcpyHostToDevice));

    /// Actual graphs on GPU memory.
    offset_t *cu_indices[num_blocks];
    wnode_t  *cu_neighbors[num_blocks];

    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++) 
            copy_subgraph_to_device(g,
                    &cu_indices[block], &cu_neighbors[block],
                    block_ranges[2 * block], block_ranges[2 * block + 1]);
    }}

    // Initialize memcopy streams.
    // idx = from_gpu * num_gpus_pr + to_gpu;
    cudaStream_t memcpy_streams[num_gpus_pr * num_gpus_pr];
    for (int from = 0; from < num_gpus_pr; from++) {{
        CUDA_ERRCHK(cudaSetDevice(from));
        for (int to = 0; to < num_gpus_pr; to++)
            CUDA_ERRCHK(cudaStreamCreate(&memcpy_streams[from * num_gpus_pr + to]));
    }}

    // score.
    size_t   score_size = g.num_nodes * sizeof(weight_t);
    weight_t *score     = nullptr; 

    /// CPU score.
    CUDA_ERRCHK(cudaMallocHost((void **) &score, score_size));
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        score[i] = init_score[i];

    /// GPU scores.
    weight_t *cu_scores[num_gpus_pr];
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMallocManaged((void **) &cu_scores[gpu], score_size));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_scores[gpu], score, score_size,
            cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr]));
    }}
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
        CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr]));
    }}

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updateds[num_gpus_pr];
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMallocManaged((void **) &cu_updateds[gpu], 
                sizeof(nid_t)));
    }}

    // Create compute streams and markers.
    cudaStream_t compute_streams[num_blocks]; // Streams for compute.
    cudaEvent_t  compute_markers[num_blocks]; // Compute complete indicators.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++) {{
            CUDA_ERRCHK(cudaStreamCreate(&compute_streams[b]));
            CUDA_ERRCHK(cudaEventCreate(&compute_markers[b]));
        }}
    }}

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
    for (wnode_t nei : g.get_neighbors(start)) {{
        if (nei.v == start) continue;

        score[nei.v] = nei.w;       
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                cu_scores[gpu] + nei.v, score + nei.v,
                sizeof(weight_t), cudaMemcpyHostToDevice));
        }}
    }}
    epochs++;
    */

    int iters=0;
    while (updated != 0) {{
        if(iters>200){{
            break;
        }}
        iters++;
        // Reset update counters.
        updated = cpu_updated = 0;          
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemsetAsync(cu_updateds[gpu], 0, 
                    sizeof(nid_t)));
        }}

        // Launch GPU epoch kernels.
        // Implicit CUDA device synchronize at the start of kernels.
        {indent_after(generate_gpu_kernel_launches(), 8)}

        // Launch CPU epoch kernels.
        {indent_all(generate_cpu_kernel_launches(), 8)}

        // Sync compute streams.
        for (int b = 0; b < num_blocks; b++)
            CUDA_ERRCHK(cudaEventSynchronize(compute_markers[b]));

        // Synchronize updates.
        nid_t gpu_updateds[num_gpus_pr];
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                    &gpu_updateds[gpu], cu_updateds[gpu],  sizeof(nid_t), 
                    cudaMemcpyDeviceToHost, memcpy_streams[gpu * num_gpus_pr + gpu]));
        }}
        updated += cpu_updated;

        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr + gpu]));
            updated += gpu_updateds[gpu];
        }}

        // Only update GPU scores if another epoch will be run.
        /*if (updated != 0) {{
            // Copy CPU scores to all GPUs.
            {indent_after(generate_score_HtoD_synchronize(), 12)}

            // Copy GPU scores peer-to-peer.
            // Not implmented if INTERLEAVE=true.
            gpu_butterfly_P2P_pr(seg_ranges, cu_scores, memcpy_streams); 

            // Synchronize HtoD async calls.
            {indent_after(generate_score_HtoD_synchronize_sync(), 12)}
        }}*/

        {indent_after(generate_interleave_DtoH_synchronize(), 8)}

        {indent_after(generate_interleave_synchronize(), 8)}
        epochs++;
    }}
    {indent_after(generate_score_DtoH_synchronize())}
    timer.Stop();

    // Copy output.
    *ret_score = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        (*ret_score)[i] = score[i];

    // Free streams.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++) {{
            CUDA_ERRCHK(cudaStreamDestroy(compute_streams[b]));
            CUDA_ERRCHK(cudaEventDestroy(compute_markers[b]));
        }}

        for (int to = 0; to < num_gpus_pr; to++)
            CUDA_ERRCHK(cudaStreamDestroy(memcpy_streams[gpu * num_gpus_pr + to]));
    }}

    // Free memory.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaFree(cu_updateds[gpu]));
        CUDA_ERRCHK(cudaFree(cu_scores[gpu]));
        
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++
        ) {{
            CUDA_ERRCHK(cudaFree(cu_indices[block]));
            CUDA_ERRCHK(cudaFree(cu_neighbors[block]));
        }}
    }}
    CUDA_ERRCHK(cudaFreeHost(score));
    delete[] seg_ranges;

    return timer.Millisecs();
}}

/**
 * Enable peer access between all compatible GPUs.
 */
void enable_all_peer_access_pr() {{
    int can_access_peer;
    for (int from = 0; from < num_gpus_pr; from++) {{
        CUDA_ERRCHK(cudaSetDevice(from));

        for (int to = 0; to < num_gpus_pr; to++) {{
            if (from == to) continue;

            CUDA_ERRCHK(cudaDeviceCanAccessPeer(&can_access_peer, from, to));
            if(can_access_peer) {{
                CUDA_ERRCHK(cudaDeviceEnablePeerAccess(to, 0));
                std::cout << from << " " << to << " yes" << std::endl;
            }} else {{
                std::cout << from << " " << to << " no" << std::endl;
            }}
        }}
    }}
}}

/**
 * Butterfly GPU P2P transfer.
 */
void gpu_butterfly_P2P_pr(nid_t *seg_ranges, weight_t **cu_scores, 
    cudaStream_t *memcpy_streams
) {{
    {indent_after(generate_butterfly_transfer())}
}}

#endif // SRC_KERNELS_HETEROGENEOUS__PR_CUH
"""
    # Remove empty lines and spaces at start and end.
    return source_code.strip()

###############################################################################
##### Helper Functions ########################################################
###############################################################################

def indent_after(block: str, indent=4) -> str:
    """Indent all lines except for the first one with {indent} number of 
    spaces. Ignores #if #endif preprocessor statements."""
    lines = block.split('\n')
    indstr = ' ' * indent
    return '\n'.join([lines[0]] + [('' if is_ifendif(line) else indstr) + line 
                                   for line in lines[1:]])

def indent_all(block: str, indent=4) -> str:
    """Indent all lines with {indent} number of spaces. Ignores #if #endif 
    preprocessor statements."""
    return ('' if is_ifendif(block) else ' ' * indent) \
        + indent_after(block, indent)

def is_ifendif(line: str) -> bool:
    return line[:3] == '#if' or line[:6] == '#endif' or line[:7] == '#pragma'

def is_gpu(device_name: str) -> bool:
    return device_name in [
        'NVIDIA Quadro RTX 4000',
        'NVIDIA Tesla V100',
        'NVIDIA Tesla K80',
        'NVIDIA Tesla M60'
    ]
