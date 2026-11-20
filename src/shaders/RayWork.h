#pragma once
#include "gpu/RayTracingRecord.h"

// One atomic reservation per contributing SIMD group, including divergent primary shading lanes.
inline void RayAppendPixel(device uint *work, uint pixel, bool append) {
    const uint rank = simd_prefix_exclusive_sum(uint(append));
    const uint count = simd_sum(uint(append));
    uint first = 0u;
    if (simd_is_first() && count != 0u)
        first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(work), count, memory_order_relaxed);
    first = simd_broadcast_first(first);
    if (append) work[RayPixelOffset + first + rank] = pixel;
}

inline uint2 RayWorkPixel(device const uint *work, uint index, uint width) {
    const uint pixel = work[RayPixelOffset + index];
    return uint2(pixel % width, pixel / width);
}

inline uint2 RayTilePixel(uint tile, uint2 local, uint width) {
    const uint columns = (width + 7u) / 8u;
    return uint2(tile % columns, tile / columns) * 8u + local;
}
