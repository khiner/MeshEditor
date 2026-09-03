#ifndef SELECTIONPICKKEY_MSL
#define SELECTIONPICKKEY_MSL

#include "Bindless.metal"
#include "ElementSelectQuery.metal"

// Orders candidates by radial distance, depth bits, and then ID through atomic minimum operations.
inline uint PackElementPickKey(uint distance_sq, float depth) {
    const uint depth_bits = as_type<uint>(metal::clamp(depth, 0.0f, 1.0f)) >> 13u;
    return (distance_sq << 19) | depth_bits;
}

// Reduces the first pass to a key and the second pass to the lowest ID with that key.
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
