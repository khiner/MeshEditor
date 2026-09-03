#ifndef SELECTIONOBJECTQUERY_MSL
#define SELECTIONOBJECTQUERY_MSL

#include "Bindless.metal"
#include "ObjectSelectQuery.metal"

// Sorts by current click epoch, radial distance, and depth so stale keys follow current candidates.
inline uint PackObjectPickKey(uint epoch_inv, uint dist_sq, float depth) {
    // Store radial distance to preserve one-pixel resolution beyond sixteen pixels.
    const uint dist_u8 = metal::min(uint(metal::sqrt(float(dist_sq))), 0xffu);
    const uint depth_u16 = uint(metal::clamp(depth, 0.0f, 1.0f) * 65535.0f + 0.5f);
    return (epoch_inv << 24u) | (dist_u8 << 16u) | depth_u16;
}

// Accumulates order-independent fragment contributions through atomic minimum and OR operations.
inline void WriteObjectSelect(
    device const BindlessSet &bindless, constant ObjectSelectQuery &q, uint2 pixel, float depth, uint id
) {
    if (id == 0u || id > q.MaxId) return;
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
