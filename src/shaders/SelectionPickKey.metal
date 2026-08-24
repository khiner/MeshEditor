#ifndef SELECTIONPICKKEY_MSL
#define SELECTIONPICKKEY_MSL

#include "Bindless.metal"
#include "ElementSelectQuery.metal"

// Nearest first, then shallowest. One atomic minimum folds every covering fragment into this key
// the same way whatever order they arrive in, and the id pass breaks the remaining ties.
// Depth keeps its float bit pattern, which orders like the value it holds and keeps its precision
// where geometry sits.
inline uint PackElementPickKey(uint distance_sq, float depth) {
    const uint depth_bits = as_type<uint>(metal::clamp(depth, 0.0f, 1.0f)) >> 13u;
    return (distance_sq << 19) | depth_bits;
}

// One fragment's contribution to a pick, run twice over the same raster: with no id slot it reduces
// to the winning key, and with one it takes the lowest id that reported that key.
inline void WriteElementPick(
    device const BindlessSet &bindless, constant ElementSelectQuery &q, uint2 pixel, float depth, uint id
) {
    if (id == 0u || q.KeySlot == INVALID_SLOT) return;
    const int2 delta = int2(pixel) - int2(q.TargetPx);
    const uint distance_sq = uint(delta.x * delta.x + delta.y * delta.y);
    if (distance_sq > q.RadiusSq) return;

    device atomic_uint *key = BindlessBufferMutable(atomic_uint, bindless.Buffer, q.KeySlot);
    const uint packed = PackElementPickKey(distance_sq, depth);
    if (q.IdSlot == INVALID_SLOT) {
        atomic_fetch_min_explicit(key, packed, memory_order_relaxed);
        return;
    }
    if (packed != atomic_load_explicit(key, memory_order_relaxed)) return;
    atomic_fetch_min_explicit(BindlessBufferMutable(atomic_uint, bindless.Buffer, q.IdSlot), id, memory_order_relaxed);
}

#endif
