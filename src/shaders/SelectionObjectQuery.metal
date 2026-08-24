#ifndef SELECTIONOBJECTQUERY_MSL
#define SELECTIONOBJECTQUERY_MSL

#include "Bindless.metal"
#include "ObjectSelectQuery.metal"

// Radial distance first, then depth, within this click's epoch, so keys an older click left behind
// sort after every candidate this click rasters.
inline uint PackObjectPickKey(uint epoch_inv, uint dist_sq, float depth) {
    // Radial distance, not its square, so the field separates candidates a pixel apart out to its
    // own end rather than collapsing everything past sixteen pixels into one bucket.
    const uint dist_u8 = metal::min(uint(metal::sqrt(float(dist_sq))), 0xffu);
    const uint depth_u16 = uint(metal::clamp(depth, 0.0f, 1.0f) * 65535.0f + 0.5f);
    return (epoch_inv << 24u) | (dist_u8 << 16u) | depth_u16;
}

// One fragment's contribution to a box select, a pick, or both. Every write is a minimum or an or,
// so the answer does not depend on the order fragments arrive in.
inline void WriteObjectSelect(
    device const BindlessSet &bindless, constant ObjectSelectQuery &q, uint2 pixel, float depth, uint id
) {
    if (id == 0u || id > q.MaxId) return; // Ids past the cap address neither bitset nor the key array.
    const uint bit = id - 1u;
    if (q.BoxResultSlot != INVALID_SLOT &&
        pixel.x >= q.Box.x && pixel.x <= q.Box.z && pixel.y >= q.Box.y && pixel.y <= q.Box.w) {
        device atomic_uint *bits = BindlessBufferMutable(atomic_uint, bindless.Buffer, q.BoxResultSlot);
        atomic_fetch_or_explicit(&bits[bit >> 5u], 1u << (bit & 31u), memory_order_relaxed);
    }
    if (q.BestKeySlot == INVALID_SLOT) return;
    const int2 delta = int2(pixel) - int2(q.TargetPx);
    const uint dist_sq = uint(delta.x * delta.x + delta.y * delta.y);
    if (dist_sq > q.RadiusSq) return;
    device atomic_uint *keys = BindlessBufferMutable(atomic_uint, bindless.Buffer, q.BestKeySlot);
    atomic_fetch_min_explicit(&keys[bit], PackObjectPickKey(q.EpochInv, dist_sq, depth), memory_order_relaxed);
    device atomic_uint *seen = BindlessBufferMutable(atomic_uint, bindless.Buffer, q.SeenBitsSlot);
    atomic_fetch_or_explicit(&seen[bit >> 5u], 1u << (bit & 31u), memory_order_relaxed);
}

#endif
