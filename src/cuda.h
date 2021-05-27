/**
 * NVIDIA helper functions.
 *
 * Toucan specs:
 *   - Quadro RTX 4000 (compute capability 7.5) [cannot use certain warp level
 *     functionality]
 */

#ifndef CUDA_H
#define CUDA_H

#include <type_traits>

#define ALLWARP (1 << warpSize - 1) // Mask for all warps.

// https://stackoverflow.com/questions/16252902/sfinae-set-of-types-contains-the-type
template <typename T, typename ...> struct is_contained : std::false_type {};
template <typename T, typename Head, typename ...Tail>
struct is_contained<T, Head, Tail...> : std::integral_constant<bool,
    std::is_same<T, Head>::value || is_contained<T, Tail...>::value> {}; 

/**
 * Only thread of warp id = 0 receive a warp-level minimum.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to minimize.
 * Returns:
 *   minimum value across all warps (only for thread of warp id = 0).
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__inline__ __device__
T warp_min(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val = min(val, __shfl_down_sync(ALLWARP, val, offset));
    return val;
}

/**
 * Each thread in a warp will receive a warp-level minimum.
 * __shfl_xor_sync creates a butterfly pattern.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to minimize.
 * Returns:
 *   minimum value across all warps.
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__inline__ __device__
T warp_all_min(T val) {
    for (int mask = warpSize >> 1; mask > 0; mask >>= 1) 
        val = min(val, __shfl_xor_sync(ALLWARP, val, mask));
    return val;
}

#endif // CUDA_H
