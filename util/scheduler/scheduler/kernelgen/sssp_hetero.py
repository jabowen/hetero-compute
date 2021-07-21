###############################################################################
# C++ Heterogeneous SSSP "compiler                                            #
#   Generates a heterogeneous SSSP kernel given a particular schedule and a   #
#   particular hardware configuration (e.g., 1 CPU 1 GPU, 1 CPU 6 GPU, etc).  #
#   The schedule (kernels for each device) will be hardcoded but the size of  #
#   the input graph will be dynamic.                                          #
###############################################################################

import enum
from functools import reduce
from typing import *
from scheduler.scheduler import (
    DeviceSchedule, 
    KernelSegment
)

# Interleave computation and memory transfers.
INTERLEAVE = True

###############################################################################
##### Enums ###################################################################
###############################################################################

class SSSPKernels(enum.Enum):
    CPU_one_to_one = enum.auto()
    GPU_one_to_one = enum.auto()
    GPU_warp_min   = enum.auto()
    GPU_block_min  = enum.auto()

def to_epochkernel_funcname(kernel: SSSPKernels) -> str:
    func_names = {
        SSSPKernels.CPU_one_to_one: 'epoch_sssp_pull_cpu_one_to_one',
        SSSPKernels.GPU_one_to_one: 'epoch_sssp_pull_gpu_one_to_one',
        SSSPKernels.GPU_warp_min:   'epoch_sssp_pull_gpu_warp_min',
        SSSPKernels.GPU_block_min:  'epoch_sssp_pull_gpu_block_min',
    }
    return func_names[kernel]

def ker_to_string(kernel: SSSPKernels) -> str:
    string_names = {
        SSSPKernels.CPU_one_to_one: 'SSSP CPU one-to-one',
        SSSPKernels.GPU_one_to_one: 'SSSP GPU one-to-one',
        SSSPKernels.GPU_warp_min:   'SSSP GPU warp-min',
        SSSPKernels.GPU_block_min:  'SSSP GPU block-min',
    }
    return string_names[kernel]

class Kernel:
    def __init__(self, kerid: SSSPKernels, block_count=64, thread_count=64):
        self.kerid        = kerid
        self.block_count  = block_count
        self.thread_count = thread_count
        self.is_gpu       = kerid not in [SSSPKernels.CPU_one_to_one]

    def __repr__(self) -> str:
        return f'Kernel(kernel={ker_to_string(self.kerid)}, '\
            f'block_count={self.block_count}, '\
            f'thread_count={self.thread_count})'

def parse_kernel(kerstr: str) -> Kernel:
    """Returns corresponding kernel object based on string description."""
    # Figure out kernel.
    kerid = None
    for k in SSSPKernels:
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
##### SSSP Heterogenous Generator #############################################
###############################################################################

def generate_sssp_hetero_source_code(scheds: List[DeviceSchedule]) -> str:
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

    num_blocks   = sum(len(sched) for sched in gpu_segments)
    num_segments = max(kerseg.seg_end 
                       for devsched in scheds
                       for kerseg in devsched.schedule) + 1
    num_gpus     = len(gpu_segments)

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
    def generate_gpu_blocks() -> str:
        gpu_blocks = gpu_prefix_sum + \
            [gpu_prefix_sum[-1] + len(gpu_segments[-1])]
        gpu_blocks = [str(block) for block in gpu_blocks]
        return f'constexpr int gpu_blocks[] = {{{", ".join(gpu_blocks)}}};'

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

    def generate_gpu_kernel_launches() -> str:
        code = ''
        for devid, devsched in enumerate(gpu_segments):
            code += f'CUDA_ERRCHK(cudaSetDevice({devid}));' + '\n'
            kernel_launches = list()
            for kerid, kerseg in enumerate(devsched):
                idx    = gpu_prefix_sum[devid] + kerid
                kernel = parse_kernel(kerseg.kernel_name)
                segcode = \
f"""
{to_epochkernel_funcname(kernel.kerid)}<<<{kernel.block_count}, {kernel.thread_count}, 0, streams[{idx}]>>>(
        cu_indices[{idx}], cu_neighbors[{idx}],
        block_ranges[{2 * idx}], block_ranges[{2 * idx + 1}],
        cu_dists[{devid}], cu_updateds[{devid}]);
CUDA_ERRCHK(cudaMemcpyAsync(
    dist + block_ranges[{2 * idx}], cu_dists[{devid}] + block_ranges[{2 * idx}],
    (block_ranges[{2 * idx + 1}] - block_ranges[{2 * idx}]) * sizeof(weight_t),
    cudaMemcpyDeviceToHost, streams[{idx}]));
""".strip() + '\n'
                kernel_launches.append(segcode)
            #code += '\n'.join(reversed(kernel_launches))
            code += '\n'.join(kernel_launches)
        return code.strip()

    def generate_cpu_kernel_launches() -> str:
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
    {to_epochkernel_funcname(kernel.kerid)}(g, dist, 
            seg_ranges[{kerseg.seg_start}], seg_ranges[{kerseg.seg_end + 1}],
            omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
}}
""".strip() + '\n'
        return code.strip()

    def generate_distance_HtoD_synchronize() -> str:
        code = ''
        cpu_schedule = next(filter(lambda devsched: 
                                      not is_gpu(devsched.device_name),
                                   scheds))
        for kerseg in cpu_schedule.schedule:
            code += \
f""" 
for (int gpu = 0; gpu < num_gpus; gpu++) {{
    CUDA_ERRCHK(cudaMemcpyAsync(
        cu_dists[gpu] + seg_ranges[{kerseg.seg_start}],
        dist + seg_ranges[{kerseg.seg_start}],
        (seg_ranges[{kerseg.seg_end + 1}] - seg_ranges[{kerseg.seg_start}]) * sizeof(weight_t),
        cudaMemcpyHostToDevice));
}}
""".strip() + '\n'
        return code.strip()

    ###########################################################################
    ##    Main Source Code Generation                                        ##
    ###########################################################################
    source_code = \
f"""
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

constexpr int num_gpus = {num_gpus};

/**
 * Runs SSSP kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. 
 * Configuration:
 *   - 1x {cpu_name}
 *   - {num_gpus}x {gpu_name}
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

    /// Actual graphs on GPU memory.
    offset_t *cu_indices[num_blocks];
    wnode_t  *cu_neighbors[num_blocks];

    for (int gpu = 0; gpu < num_gpus; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++) 
            copy_subgraph_to_device(g,
                    &cu_indices[block], &cu_neighbors[block],
                    block_ranges[2 * block], block_ranges[2 * block + 1]);
    }}

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
    for (int gpu = 0; gpu < num_gpus; gpu++) {{        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_dists[gpu], dist_size));
        CUDA_ERRCHK(cudaMemcpy(cu_dists[gpu], init_dist, dist_size,
            cudaMemcpyHostToDevice));
    }}

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updateds[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_updateds[gpu], 
                sizeof(nid_t)));
    }}

    // Create streams.
    cudaStream_t streams[num_blocks];
    for (int gpu = 0; gpu < num_gpus; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++)
            CUDA_ERRCHK(cudaStreamCreate(&streams[b]));
    }}

    // Get init vertex.
    nid_t start;
    for (nid_t i = 0; i < g.num_nodes; i++)
        if (init_dist[i] != INF_WEIGHT) start = i;

    // Start kernel!
    Timer timer; timer.Start();
    int epochs = 0;

    // Push for the first iteration.
    // TODO: implement push for more than one epoch. Requires parallel queue.
    for (wnode_t nei : g.get_neighbors(start)) {{
        dist[nei.v] = nei.w;       
        for (int gpu = 0; gpu < num_gpus; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                cu_dists[gpu] + nei.v, dist + nei.v,
                sizeof(weight_t), cudaMemcpyHostToDevice));
        }}
    }}
    epochs++;

    while (updated != 0) {{
        // Reset update counters.
        updated = cpu_updated = 0;          
        for (int gpu = 0; gpu < num_gpus; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemsetAsync(cu_updateds[gpu], 0, 
                    sizeof(nid_t)));
        }}

        // Launch GPU epoch kernels.
        // Implicit CUDA device synchronize at the start of kernels.
        {indent_after(generate_gpu_kernel_launches(), 8)}

        // Launch CPU epoch kernels.
        {indent_all(generate_cpu_kernel_launches(), 8)}

        // Sync streams.
        for (int i = 0; i < num_blocks; i++)
            CUDA_ERRCHK(cudaStreamSynchronize(streams[i]));

        // Synchronize updates.
        nid_t gpu_updateds[num_gpus];
        for (int gpu = 0; gpu < num_gpus; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                    &gpu_updateds[gpu], cu_updateds[gpu], 
                    sizeof(nid_t), cudaMemcpyDeviceToHost));
        }}
        updated += cpu_updated;

        for (int gpu = 0; gpu < num_gpus; gpu++) {{
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaDeviceSynchronize());
            updated += gpu_updateds[gpu];
        }}

        // Only update GPU distances if another epoch will be run.
        if (updated != 0) {{
            // Copy CPU distances to all GPUs.
            {indent_after(generate_distance_HtoD_synchronize(), 12)}

            // Copy GPU distances peer-to-peer.
            for (int src_gpu = 0; src_gpu < num_gpus; src_gpu++) {{
                CUDA_ERRCHK(cudaSetDevice(src_gpu));
                for (int dst_gpu = 0; dst_gpu < num_gpus; dst_gpu++) {{
                    if (src_gpu == dst_gpu) continue;
                    
                    for (int block = gpu_blocks[src_gpu];
                            block < gpu_blocks[src_gpu + 1]; block++
                    ) {{
                        int start = block_ranges[2 * block];
                        int end   = block_ranges[2 * block + 1];
                        CUDA_ERRCHK(cudaMemcpyAsync(
                                    cu_dists[dst_gpu] + start,
                                    cu_dists[src_gpu] + start,
                                    (end - start) * sizeof(weight_t),
                                    cudaMemcpyDeviceToDevice));
                    }}
                }}
            }}
        }}
        epochs++;
    }}
    // Wait for memops to complete.
    for (int gpu = 0; gpu < num_gpus; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaDeviceSynchronize());
    }}
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
    for (int gpu = 0; gpu < num_gpus; gpu++) {{
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaFree(cu_updateds[gpu]));
        CUDA_ERRCHK(cudaFree(cu_dists[gpu]));
        
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++
        ) {{
            CUDA_ERRCHK(cudaFree(cu_indices[block]));
            CUDA_ERRCHK(cudaFree(cu_neighbors[block]));
        }}
    }}
    CUDA_ERRCHK(cudaFreeHost(dist));
    delete[] seg_ranges;

    return timer.Millisecs();
}}

/**
 * Enable peer access between all compatible GPUs.
 */
void enable_all_peer_access() {{
    int can_access_peer;
    for (int from = 0; from < num_gpus; from++) {{
        CUDA_ERRCHK(cudaSetDevice(from));

        for (int to = 0; to < num_gpus; to++) {{
            if (from == to) continue;

            CUDA_ERRCHK(cudaDeviceCanAccessPeer(&can_access_peer, from, to));
            if(can_access_peer)
                CUDA_ERRCHK(cudaDeviceEnablePeerAccess(to, 0));
        }}
    }}
}}

#endif // SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
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
    return line[:3] == '#if' or line[:6] == '#endif'

def is_gpu(device_name: str) -> bool:
    return device_name in [
        'NVIDIA Quadro RTX 4000',
        'NVIDIA Tesla V100',
        'NVIDIA Tesla M60'
    ]
