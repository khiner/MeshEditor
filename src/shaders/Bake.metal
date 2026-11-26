// The texels of a grid over a record's UV layout that each triangle covers, and the nearest triangle of the texels near the layout.
// The triangles' bounds span fewer than 2^32 rows of texels in all.
#include "BakeTexels.metal"
#include "gpu/BakeJob.h"

// The address of the live triangle in slot i of the record's triangle blocks, or InvalidOffset.
inline uint BakeTriangleAddress(device const BindlessSet &b, DomainBlocks blocks, uint i) {
    const uint k = i / MeshElementBlockSize, lane = i % MeshElementBlockSize;
    if (k >= blocks.Count) return InvalidOffset;
    const uint block = BindlessBuffer(uint, b.Buffer, blocks.Blocks.Slot)[blocks.Blocks.Offset + k];
    device const MeshElementBlock &membership = BindlessBuffer(MeshElementBlock, b.Buffer, blocks.MembershipSlot)[block];
    return SlotLive(membership, lane) ? block * MeshElementBlockSize + lane : InvalidOffset;
}
inline BakeTriangle BakeTriangleAt(device const BindlessSet &b, thread const BakeJob &job, uint address) {
    return BakeTriangleOf(b, job.Uvs, uint3(BindlessBuffer(packed_uint3, b.Buffer, job.TrianglesSlot)[address]), job.Width, job.Height);
}
// The first and last column and row of the texels whose centers lie in the triangle's bounds, empty when the last precedes the first.
inline float4 BakeBounds(BakeTriangle t, uint width, uint height) {
    const float2 low = min(min(t.Q[0], t.Q[1]), t.Q[2]), high = max(max(t.Q[0], t.Q[1]), t.Q[2]);
    return float4(max(ceil(low - 0.5f), 0.0f), min(floor(high - 0.5f), float2(width, height) - 1.0f));
}
// The first and last column of the texels of a row of the triangle's bounds whose centers can lie in the triangle, one texel past where its edges cross the row's centers, empty when the last precedes the first.
inline float2 BakeSpan(BakeTriangle t, float4 bounds, float row) {
    const float y = row + 0.5f;
    float low = INFINITY, high = -INFINITY;
    for (uint k = 0u; k < 3u; ++k) {
        const float2 a = t.Q[k], b = t.Q[(k + 1u) % 3u];
        if ((a.y - y) * (b.y - y) > 0.0f) continue;
        // An edge along the row crosses it at both ends.
        const float x = a.y == b.y ? a.x : a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x);
        low = min(low, a.y == b.y ? min(a.x, b.x) : x);
        high = max(high, a.y == b.y ? max(a.x, b.x) : x);
    }
    return float2(max(bounds.x, ceil(low - 0.5f) - 1.0f), min(bounds.z, floor(high - 0.5f) + 1.0f));
}
inline device uint *BakeWords(device const BindlessSet &b, SlotOffset words) { return BindlessBufferMutable(uint, b.Buffer, words.Slot) + words.Offset; }
// The distance in texels from the position to the triangle.
inline float BakeDistance(device const BindlessSet &b, thread const BakeJob &job, uint address, float2 c) {
    const BakeTriangle t = BakeTriangleAt(b, job, address);
    return distance(BakePosition(t, BakeNearest(t, c)), c);
}

// Counts the rows of texels in each live triangle's bounds, and none for a triangle with no area, then ends the counts with zero.
kernel void BakeTriangleCounts(JOB_TABLE_ARGS(BakeJob)) {
    JOB_TILE_THREAD(tiled, i, BakeTile);
    const BakeJob job = tiled;
    if (i > job.Triangles.Count * MeshElementBlockSize) return;
    const uint address = BakeTriangleAddress(b, job.Triangles, i);
    uint count = 0u;
    if (address != InvalidOffset) {
        const BakeTriangle t = BakeTriangleAt(b, job, address);
        const float4 bounds = BakeBounds(t, job.Width, job.Height);
        if (BakeArea(t) != 0.0f && all(bounds.zw >= bounds.xy)) count = uint(bounds.w - bounds.y + 1.0f);
    }
    BakeWords(b, job.Counts)[i] = count;
}

// Marks each texel whose center lies in a triangle with the largest such triangle's address plus one, from the scanned counts.
// A thread visits every thread-count-th row of the triangles' bounds, and one thread runs per texel or per triangle slot, whichever are more.
kernel void BakeCoverTexels(JOB_TABLE_ARGS(BakeJob)) {
    JOB_TILE_THREAD(tiled, i, BakeTile);
    const BakeJob job = tiled;
    const uint slots = job.Triangles.Count * MeshElementBlockSize, threads = max(job.Width * job.Height, slots);
    if (i >= threads) return;
    device const uint *first = BakeWords(b, job.Counts);
    device atomic_uint *covered = reinterpret_cast<device atomic_uint *>(BakeWords(b, job.Texels));
    for (uint k = i; k < first[slots]; k += threads) {
        const uint slot = PartitionPoint(slots, [&](uint m) { return first[m + 1u] <= k; });
        const uint address = BakeTriangleAddress(b, job.Triangles, slot);
        const BakeTriangle t = BakeTriangleAt(b, job, address);
        const float4 bounds = BakeBounds(t, job.Width, job.Height);
        const uint row = uint(bounds.y) + k - first[slot];
        const float2 span = BakeSpan(t, bounds, float(row));
        for (float column = span.x; column <= span.y; column += 1.0f) {
            if (BakeCovers(t, float2(column, float(row)) + 0.5f)) atomic_fetch_max_explicit(covered + row * job.Width + uint(column), address + 1u, memory_order_relaxed);
        }
    }
}

// One jump of the texels' search for their nearest triangle: each texel takes the nearest of its own and the triangles of the texels Step away along each axis and diagonal, the larger address on a tie.
kernel void BakeMarginJump(JOB_TABLE_ARGS(BakeJob)) {
    JOB_TILE_THREAD(tiled, i, BakeTile);
    const BakeJob job = tiled;
    if (i >= job.Width * job.Height) return;
    device const uint *from = BakeWords(b, job.Texels);
    const int2 texel{int(i % job.Width), int(i / job.Width)};
    const float2 center = float2(texel) + 0.5f;
    uint nearest = from[i];
    float distance = nearest ? BakeDistance(b, job, nearest - 1u, center) : INFINITY;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const int2 at = texel + int2(x, y) * int(job.Step);
            if (any(at < 0) || at.x >= int(job.Width) || at.y >= int(job.Height)) continue;
            const uint held = from[uint(at.y) * job.Width + uint(at.x)];
            if (!held || held == nearest) continue;
            const float d = BakeDistance(b, job, held - 1u, center);
            if (d < distance || (d == distance && held > nearest)) {
                nearest = held;
                distance = d;
            }
        }
    }
    BakeWords(b, job.Jumped)[i] = nearest;
}
