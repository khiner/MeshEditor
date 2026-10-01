#ifndef MESHLETBUILD_MSL
#define MESHLETBUILD_MSL

#include "Bindless.metal"
#include "CornerRenderKey.metal"
#include "BlockScan.metal"
#include "RadixSort.metal"
#include "ElementWorkShared.metal"
#include "EnclosingSphere.metal"
#include "ConnectivityRead.metal"
#include "gpu/CornerClass.h"
#include "gpu/CornerClassMode.h"
#include "gpu/MeshletBuildJob.h"
#include "gpu/MeshletBuildPushConstants.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletRecord.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/LodNode.h"

constant uint BuildHashSlots = 2048u;
constant uint BuildClusterThreads = MeshletBuildClusterThreads;
constant uint BuildRecordWords = sizeof(MeshletRecord) / sizeof(uint);

// Stats: ordered-float bounds[6], tiles, record bound, emitted records,
// emitted vertices, with the remaining words reserved for validation.
struct MeshletBuilder {
    device const BindlessSet &B;
    constant MeshletBuildPushConstants &Pc;
    MeshletBuildJob J;
    device uint *S() const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot); }
    device atomic_uint *A() const { return reinterpret_cast<device atomic_uint *>(S()); }
    device uint *Stats() const { return S() + J.StatsOffset; }
    device uint *Prim(uint p) const { return S() + J.PrimitiveScratchOffset + 8u * p; }
    device uint *Tile(uint t) const { return S() + J.TileScratchOffset + 4u * t; }
    device uint *Counts(uint t) const { return S() + J.TileCountsOffset + 4u * t; }
    device MeshletRecord *Records() const { return reinterpret_cast<device MeshletRecord *>(S() + J.RecordScratchOffset); }
    ConnectivityView Conn() const { return {B, J.Mesh.Connectivity, J.Mesh.FaceCount}; }
    uint Element(uint e) const {
        return J.Elements.Storage.Slot != InvalidSlot ? WorkGroupElement(B,J.Elements,e) :
            J.Mesh.TriangleSlot != InvalidSlot ? J.Mesh.TriangleOffset+e : e;
    }
    uint TriangleCorner(uint t, uint c) const {
        return J.Mesh.TriangleSlot != InvalidSlot ? TriangleCornerHandle(B,J.Mesh.TriangleSlot,c,t) : J.Mesh.IndexSlotOffset.Offset+t*3u+c;
    }
    bool PhysicalBoundary(uint t,uint c) const {
        if (J.Mesh.TriangleSlot==InvalidSlot) return false;
        const uint a=TriangleCorner(t,c),d=TriangleCorner(t,(c+1u)%3u);
        return Conn().Next(a)==d && Conn().Opposite(d)==InvalidOffset;
    }
    uint TriangleVertex(uint t, uint c) const {
        return BindlessBuffer(uint,B.IndexBuffer,J.Mesh.IndexSlotOffset.Slot)[TriangleCorner(t,c)]-J.Mesh.VertexOffset;
    }
    uint VertexIndex(uint corner) const {
        if (J.Topology == 2u) return (J.Elements.Storage.Slot != InvalidSlot ? Element(corner) - J.Mesh.VertexOffset : corner);
        if (J.Topology == 0u) return TriangleVertex(Element(corner/3u),corner%3u);
        // A line element's first endpoint starts its first halfedge, and its second ends it.
        const uint h = Conn().EdgeHalfedge(Element(corner/2u));
        return BindlessBuffer(uint,B.IndexBuffer,J.Mesh.IndexSlotOffset.Slot)[corner%2u == 0u ? Conn().Opposite(h) : h]-J.Mesh.VertexOffset;
    }
    float3 Position(uint v) const { return float3(BindlessBuffer(Vertex, B.VertexBuffer, J.Mesh.VertexSlot)[J.Mesh.VertexOffset + v].Position); }
    uint CornersPerElement() const { return J.Topology == 0u ? 3u : J.Topology == 1u ? 2u : 1u; }
    float3 Center(uint e) const {
        float3 p = float3(0);
        const uint count = CornersPerElement();
        for (uint c = 0u; c < count; ++c) p += Position(VertexIndex(e * count + c));
        return p / float(count);
    }
    uint SourcePrimitive(uint e) const {
        if (J.Mesh.ElementPrimitives.ValuesSlot == InvalidSlot) return 0u;
        const uint owner = J.Topology == 0u ? TriangleFaceHandle(B,J.Mesh.Connectivity,J.Mesh.TriangleSlot,Element(e)) :
            J.Mesh.VertexOffset + VertexIndex(e * CornersPerElement());
        return BindlessBuffer(uint,B.ElementPrimitiveBuffer,J.Mesh.ElementPrimitives.ValuesSlot)[ElementAttributeIndex(B,J.Mesh.ElementPrimitives,owner)];
    }
    uint Primitive(uint e) const { return WorkRank(B,J.Materials,SourcePrimitive(e)); }
    CornerRenderKey Keys() const { return {B,J.Mesh,Pc.CornerSectors,Pc.FaceSharpnessSlot}; }
    bool Flat(uint t) const {
        const uint element=Element(t);
        return Keys().Flat(uint3(TriangleCorner(element,0u),TriangleCorner(element,1u),TriangleCorner(element,2u)));
    }
    uint AttributeHandle(uint corner) const {
        if (J.Topology != 0u) return J.Mesh.VertexOffset + VertexIndex(corner);
        return TriangleCorner(Element(corner/3u),corner%3u);
    }
    uint Hash(uint c, bool flat) const { return Keys().Hash(AttributeHandle(c),flat); }
    bool Equal(uint a, bool af, uint b, bool bf) const { return Keys().Equal(AttributeHandle(a),af,AttributeHandle(b),bf); }
    uint ElementAt(uint sorted) const { return S()[((J.RadixPassCount & 1u) ? J.TempOrderOffset : J.OrderOffset) + sorted]; }
};

inline uint OrderedFloat(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : u ^ 0x80000000u; }
inline float FromOrderedFloat(uint u) { return as_type<float>((u & 0x80000000u) ? u ^ 0x80000000u : ~u); }
inline uint SpatialKeyPart(uint value, uint mask) {
    uint result = 0u;
    while (mask) {
        const uint bit = mask & (~mask + 1u);
        if (value & 1u) result |= bit;
        value >>= 1u;
        mask &= mask - 1u;
    }
    return result;
}

inline uint SpatialKeyCoordinate(float value, float lo, float hi, uint mask) {
    const uint bits = popcount(mask), used = min(bits, 24u);
    if (!used) return 0u;
    const uint cells = 1u << used;
    const float fraction = clamp((value - lo) / max(hi - lo, 1e-20f), 0.f, 1.f);
    return min(uint(fraction * float(cells)), cells - 1u) << (bits - used);
}

#define BUILD_ARGS uint batch_group [[threadgroup_position_in_grid]], device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]], constant MeshletBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
#define BUILD_CONTEXT const uint2 tile_entry = BindlessBuffer(uint2,bindless.Buffer,pc.TilesSlot)[pc.FirstTile+batch_group]; const MeshletBuilder b{bindless, pc, BindlessBuffer(MeshletBuildJob, bindless.Buffer, pc.JobsSlot)[tile_entry.x]}; const auto j = b.J

kernel void MeshletBuildMaterials(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint i=tile_entry.y*256u+lane;
    if (i >= j.ElementCount) return;
    if (j.Elements.Storage.Slot != InvalidSlot) {
        const uint count = BindlessBuffer(uint,bindless.Buffer,j.Elements.Storage.Slot)[j.Elements.Storage.Offset+5u];
        if (count != j.ElementCount && i == 0u) {
            atomic_store_explicit(BindlessBufferMutable(atomic_uint,bindless.Buffer,j.Materials.Storage.Slot)+j.Materials.Storage.Offset+1u,1u,memory_order_relaxed);
        }
        if (i >= count) return;
    }
    // The host seeds a fragment's one bound primitive.
    if (j.ExistingPrimitive != InvalidOffset) return;
    const uint material = b.SourcePrimitive(i);
    if (material >= j.Materials.Count) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,bindless.Buffer,j.Materials.Storage.Slot)+j.Materials.Storage.Offset+1u,1u,memory_order_relaxed);
    } else MarkWork(bindless,j.Materials,material);
}

kernel void MeshletBuildInit(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint i=tile_entry.y*256u+lane;
    if (i < 16u) b.Stats()[i] = i < 3u ? OrderedFloat(INFINITY) : i < 6u ? OrderedFloat(-INFINITY) : 0u;
    if (i < j.PrimitiveCount) for (uint k = 0u; k < 8u; ++k) b.Prim(i)[k] = 0u;
}

kernel void MeshletBuildBounds(uint lane [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint i=tile_entry.y*256u+lane;
    threadgroup float3 lower[8], upper[8];
    const float3 p = i < j.ElementCount ? b.Center(i) : float3(0.f);
    const float3 lo = simd_min(i < j.ElementCount ? p : float3(INFINITY));
    const float3 hi = simd_max(i < j.ElementCount ? p : float3(-INFINITY));
    if (sl == 0u) { lower[sg] = lo; upper[sg] = hi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0u) return;
    float3 minimum = lower[0], maximum = upper[0];
    for (uint g = 1u; g < 8u; ++g) { minimum = min(minimum, lower[g]); maximum = max(maximum, upper[g]); }
    for (uint k = 0u; k < 3u; ++k) {
        atomic_fetch_min_explicit(b.A() + j.StatsOffset + k, OrderedFloat(minimum[k]), memory_order_relaxed);
        atomic_fetch_max_explicit(b.A() + j.StatsOffset + 3u + k, OrderedFloat(maximum[k]), memory_order_relaxed);
    }
}

kernel void MeshletBuildKeys(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint i=tile_entry.y*256u+lane;
    float3 lo, hi;
    for (uint k = 0u; k < 3u; ++k) { lo[k] = FromOrderedFloat(b.Stats()[k]); hi[k] = FromOrderedFloat(b.Stats()[3u + k]); }
    // Longest-axis bisection gives each key bit the largest remaining
    // object-space cell dimension. One schedule is shared by the whole tile.
    threadgroup uint masks[3];
    if (lane == 0u) {
        float spans[3]{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
        masks[0] = 0u; masks[1] = 0u; masks[2] = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const uint axis = spans[0] >= spans[1] && spans[0] >= spans[2] ? 0u : spans[1] >= spans[2] ? 1u : 2u;
            masks[axis] |= 1u << uint(bit);
            spans[axis] *= 0.5f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= j.ElementCount) return;
    const float3 center = b.Center(i);
    const uint key = SpatialKeyPart(SpatialKeyCoordinate(center.x, lo.x, hi.x, masks[0]), masks[0]) |
        SpatialKeyPart(SpatialKeyCoordinate(center.y, lo.y, hi.y, masks[1]), masks[1]) |
        SpatialKeyPart(SpatialKeyCoordinate(center.z, lo.z, hi.z, masks[2]), masks[2]);
    uint p = b.Primitive(i);
    if (p >= j.PrimitiveCount) { atomic_store_explicit(b.A()+j.StatsOffset+13u,1u,memory_order_relaxed); p = 0u; }
    b.S()[j.KeysOffset + 2u * i] = key;
    b.S()[j.KeysOffset + 2u * i + 1u] = p;
    b.S()[j.OrderOffset + i] = i;
    atomic_fetch_add_explicit(b.A() + j.PrimitiveScratchOffset + 8u * p, 1u, memory_order_relaxed);

}

inline RadixSortView MeshletSort(MeshletBuilder b) {
    const auto j = b.J;
    const uint shift = b.Pc.PassParameter;
    return {b.S() + j.KeysOffset, b.S() + j.OrderOffset, b.S() + j.TempOrderOffset,
            b.S() + j.HistogramOffset, b.S() + j.DigitTotalsOffset, j.ElementCount, j.BlockCount,
            2u, shift / 32u, shift % 32u, ((shift / 4u) & 1u) != 0u};
}
kernel void MeshletBuildHistogram(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint group=tile_entry.y;
    if (pc.PassParameter>=j.RadixPassCount*4u) return;
    threadgroup atomic_uint counts[16];
    RadixHistogram(MeshletSort(b), lane, group, counts);
}
kernel void MeshletBuildHistogramPrefix(uint lane [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint digit=tile_entry.y;
    if (pc.PassParameter>=j.RadixPassCount*4u) return;
    threadgroup uint sums[9];
    RadixPrefix(MeshletSort(b), lane, digit, sl, sg, sums);
}
kernel void MeshletBuildScatter(uint lane [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint group=tile_entry.y;
    if (pc.PassParameter>=j.RadixPassCount*4u) return;
    threadgroup uint groups[128];
    RadixScatter(MeshletSort(b), lane, group, sl, sg, groups);
}

kernel void MeshletBuildSegments(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    if (lane != 0u) return;
    uint elements = 0u, tiles = 0u, records = 0u;
    for (uint p = 0u; p < j.PrimitiveCount; ++p) {
        device uint *prim = b.Prim(p);
        const uint n = prim[0], count = (n + MeshletBuildTileElements - 1u) / MeshletBuildTileElements;
        const uint bound = j.Topology == 0u ? (n / MeshletBuildTileElements) * MeshletBuildTileRecords + ((n % MeshletBuildTileElements) + 20u) / 21u : (n + 15u) / 16u;
        prim[1] = elements; prim[2] = tiles; prim[3] = count; prim[4] = records; prim[5] = bound;
        elements += n; tiles += count; records += bound;
    }
    b.Stats()[6] = tiles; b.Stats()[7] = records;
}

kernel void MeshletBuildTiles(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint i=tile_entry.y*256u+lane;
    if (i >= b.Stats()[6]) return;
    uint lo = 0u, hi = j.PrimitiveCount;
    while (lo + 1u < hi) {
        const uint mid = (lo + hi) / 2u;
        if (b.Prim(mid)[2] <= i) lo = mid; else hi = mid;
    }
    device const uint *prim = b.Prim(lo);
    const uint tile = i - prim[2], first = tile * MeshletBuildTileElements;
    device uint *out = b.Tile(i);
    out[0] = lo; out[1] = prim[1] + first; out[2] = min(MeshletBuildTileElements, prim[0] - first);
    out[3] = prim[4] + tile * (j.Topology == 0u ? MeshletBuildTileRecords : MeshletBuildTileElements / 16u);
}

inline float3 BuildVertexPosition(MeshletBuilder b, uint source) {
    return b.Position(b.J.Topology == 0u ? b.VertexIndex(source) : source);
}

inline void BuildClusterBarrier(uint simd_size, mem_flags flags) {
    // A tile uses one SIMD group on Apple GPUs with 32-lane execution. Keep
    // the full-group fence when a device divides this threadgroup differently.
    if (simd_size == BuildClusterThreads) simdgroup_barrier(flags);
    else threadgroup_barrier(flags);
}

// Two 16-bit tile-local entries share one atomic word.
// Every corner index fits in 11 bits.
// The value 0xffff marks an empty hash or incidence slot.
inline ushort ClusterTableLoad(threadgroup atomic_uint *table, uint slot) {
    return ushort(atomic_load_explicit(table + (slot >> 1u), memory_order_relaxed) >> ((slot & 1u) * 16u));
}
inline ushort ClusterTableInsert(threadgroup atomic_uint *table, uint slot, ushort value) {
    const uint index=slot >> 1u, shift=(slot & 1u) * 16u, mask=0xffffu << shift;
    uint word=atomic_load_explicit(table + index, memory_order_relaxed);
    for (;;) {
        const ushort old=ushort(word >> shift);
        if (old!=0xffffu) return old;
        const uint next=(word & ~mask) | (uint(value) << shift);
        if (atomic_compare_exchange_weak_explicit(table + index,&word,next,memory_order_relaxed,memory_order_relaxed)) return old;
    }
}
inline void ClusterTableMin(threadgroup atomic_uint *table, uint slot, ushort value) {
    const uint index=slot >> 1u, shift=(slot & 1u) * 16u, mask=0xffffu << shift;
    uint word=atomic_load_explicit(table + index, memory_order_relaxed);
    while (value<ushort(word >> shift)) {
        const uint next=(word & ~mask) | (uint(value) << shift);
        if (atomic_compare_exchange_weak_explicit(table + index,&word,next,memory_order_relaxed,memory_order_relaxed)) break;
    }
}
inline ushort ClusterTableExchange(threadgroup atomic_uint *table, uint slot, ushort value) {
    const uint index=slot >> 1u, shift=(slot & 1u) * 16u, mask=0xffffu << shift;
    uint word=atomic_load_explicit(table + index, memory_order_relaxed);
    for (;;) {
        const uint next=(word & ~mask) | (uint(value) << shift);
        if (atomic_compare_exchange_weak_explicit(table + index,&word,next,memory_order_relaxed,memory_order_relaxed)) return ushort(word >> shift);
    }
}

kernel void MeshletBuildClusters(uint lane [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint tile_id=tile_entry.y;
    if (tile_id >= b.Stats()[6]) return;
    device const uint *tile = b.Tile(tile_id);
    const uint primitive = tile[0], first = tile[1], count = tile[2];
    const uint vertex_base = first * b.CornersPerElement();
    device uint *refs = b.S() + j.VertexScratchOffset + vertex_base;
    device uint *triangles = BindlessBufferMutable(uint, bindless.Buffer, pc.TriangleIdsSlot) + j.TriangleOffset + first;
    device uchar *local = BindlessBufferMutable(uchar, bindless.Buffer, pc.LocalTrianglesSlot) + j.LocalTriangleOffset + first * 3u;
    if (j.Topology != 0u) {
        const uint endpoints = b.CornersPerElement();
        for (uint i = lane; i < count; i += BuildClusterThreads) {
            const uint e = b.ElementAt(first + i);
            triangles[i] = j.Topology == 2u ? b.VertexIndex(e) : b.Element(e) - j.Mesh.Connectivity.Edges.Offset;
            for (uint c = 0u; c < endpoints; ++c) refs[i * endpoints + c] = b.VertexIndex(e * endpoints + c);
        }
        BuildClusterBarrier(simd_size,mem_flags::mem_device);
        if (lane == 0u) {
            const uint records = (count + 15u) / 16u;
            for (uint m = 0u; m < records; ++m) {
                const uint n = min(16u, count - m * 16u);
                MeshletRecord record{.TriangleOffset = first + m * 16u, .TriangleCount = n, .VertexOffset = vertex_base + m * 16u * endpoints, .VertexCount = n * endpoints, .LocalTriangleOffset = 0u, .Primitive = primitive, .GroupIndex = InvalidOffset, .RefinedGroup = InvalidOffset, .Topology = j.Topology};
                device const uint *vertices = b.S() + j.VertexScratchOffset + record.VertexOffset;
                FitClusterBounds(record, local, [&](uint i) { return BuildVertexPosition(b, vertices[i]); });
                b.Records()[tile[3] + m] = record;
            }
            b.Counts(tile_id)[0] = records; b.Counts(tile_id)[1] = count * endpoints;
        }
        return;
    }
    threadgroup atomic_uint table[BuildHashSlots/2u];
    // Tile-local corner and triangle indices fit in 11 and 9 bits.
    // The hash table probes on its low 11 bits.
    // A 16-bit fingerprint plus the full render-key comparison preserves collision handling.
    threadgroup ushort hashes[MeshletBuildTileElements*3u], representative[MeshletBuildTileElements*3u], candidates[MeshletBuildTileElements];
    // Each tile triangle's sorted source and emitted element, read once so the serial greedy step touches no device state.
    threadgroup uint sources[MeshletBuildTileElements], elements[MeshletBuildTileElements];
    // Triangle state bits: flat (1), chosen (2), candidate (4), and a physical boundary after corner c (8 << c).
    threadgroup uchar vertex_map[MeshletBuildTileElements*3u], state[MeshletBuildTileElements];
    threadgroup uint scores[BuildClusterThreads/32u];
    threadgroup float3 centers[MeshletBuildTileElements];
    threadgroup float3 bound_positions[64];
    threadgroup uint chosen, vertex_count, triangle_count, total_vertices, total_triangles, record_count, candidate_count;
    threadgroup float3 anchor;
    threadgroup float scale;
    const uint corners = count * 3u;
    for (uint i = lane; i < BuildHashSlots/2u; i += BuildClusterThreads) atomic_store_explicit(table + i, InvalidOffset, memory_order_relaxed);
    for (uint i = lane; i < count; i += BuildClusterThreads) {
        const uint t = b.ElementAt(first + i), element = b.Element(t);
        uchar flags = b.Flat(t) ? 1u : 0u;
        for (uint c = 0u; c < 3u; ++c) if (b.PhysicalBoundary(element, c)) flags |= uchar(8u << c);
        sources[i] = t;
        elements[i] = element;
        state[i] = flags;
        centers[i] = b.Center(t);
    }
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    const auto corner_at = [&](uint c) { return sources[c / 3u] * 3u + c % 3u; };
    for (uint c = lane; c < corners; c += BuildClusterThreads) hashes[c] = ushort(b.Hash(corner_at(c), (state[c / 3u] & 1u) != 0u));
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    for (uint c = lane; c < corners; c += BuildClusterThreads) {
        uint slot = hashes[c] & (BuildHashSlots - 1u);
        for (;;) {
            const ushort other = ClusterTableInsert(table,slot,ushort(c));
            if (other == 0xffffu) break;
            if (hashes[c] == hashes[other] && b.Equal(corner_at(c), (state[c / 3u] & 1u) != 0u, corner_at(other), (state[other / 3u] & 1u) != 0u)) {
                ClusterTableMin(table,slot,ushort(c)); break;
            }
            slot = (slot + 1u) & (BuildHashSlots - 1u);
        }
        representative[c] = slot;
    }
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    for (uint c = lane; c < corners; c += BuildClusterThreads) representative[c] = ClusterTableLoad(table,representative[c]);
    if (lane == 0u) { total_vertices = 0u; total_triangles = 0u; record_count = 0u; }
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    // The representative hash table becomes an incidence head per welded corner, and hashes become linked-list next entries.
    for (uint c = lane; c < (corners+1u)/2u; c += BuildClusterThreads) atomic_store_explicit(table + c, InvalidOffset, memory_order_relaxed);
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    for (uint c = lane; c < corners; c += BuildClusterThreads)
        hashes[c] = ClusterTableExchange(table,representative[c],ushort(c));
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float3 lower = float3(INFINITY), upper = float3(-INFINITY);
        for (uint i = 0u; i < count; ++i) { lower = min(lower, centers[i]); upper = max(upper, centers[i]); }
        scale = 1.f / max(dot(upper - lower, upper - lower), 1e-20f);
    }
    BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    while (total_triangles < count) {
        for (uint c = lane; c < corners; c += BuildClusterThreads) vertex_map[c] = 0xffu;
        for (uint i = lane; i < count; i += BuildClusterThreads) state[i] &= ~uchar(4u);
        if (lane == 0u) { vertex_count = 0u; triangle_count = 0u; candidate_count = 0u; anchor = float3(0); }
        BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
        for (;;) {
            // Reuse dominates proximity in the score.
            // Every triangle with a reused corner is in candidates.
            // Scan the tile only to seed a new meshlet or when no connected triangle fits its vertex cap.
            uint best = 0u;
            for (uint scan = 0u; scan < 2u; ++scan) {
                const bool connected = scan == 0u && candidate_count != 0u;
                if (scan == 1u && (candidate_count == 0u || best != 0u)) break;
                uint score = 0u;
                const uint limit = connected ? candidate_count : count;
                for (uint entry = lane; entry < limit && triangle_count < 48u; entry += BuildClusterThreads) {
                    const uint i = connected ? candidates[entry] : entry;
                    if ((state[i] & 2u) != 0u) continue;
                    const uint3 r{representative[i * 3u], representative[i * 3u + 1u], representative[i * 3u + 2u]};
                    const uint a = vertex_map[r.x], c = vertex_map[r.y], d = vertex_map[r.z];
                    const uint added = uint(a == 0xffu) + uint(c == 0xffu && r.y != r.x) + uint(d == 0xffu && r.z != r.x && r.z != r.y);
                    if (vertex_count + added <= 64u) {
                        const uint reuse = uint(a != 0xffu) + uint(c != 0xffu) + uint(d != 0xffu);
                        const uint near = triangle_count == 0u ? 0u : uint((1.f - clamp(dot(centers[i] - anchor / float(triangle_count), centers[i] - anchor / float(triangle_count)) * scale, 0.f, 1.f)) * 1048575.f);
                        score = max(score, ((reuse + 1u) << 29u) | (near << 9u) | (511u - i));
                    }
                }
                best = simd_max(score);
            }
            if (sl == 0u) scores[sg] = best;
            BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
            if (lane == 0u) {
                uint best_score = 0u;
                for (uint g = 0u; g < BuildClusterThreads/32u; ++g) best_score = max(best_score, scores[g]);
                chosen = best_score == 0u ? InvalidOffset : 511u - (best_score & 511u);
                if (chosen != InvalidOffset) {
                    state[chosen] |= 2u;
                    triangles[total_triangles + triangle_count] = elements[chosen];
                    for (uint c = 0u; c < 3u; ++c) {
                        const uint rep = representative[chosen * 3u + c];
                        uint v = vertex_map[rep];
                        if (v == 0xffu) {
                            v = vertex_count++; vertex_map[rep] = v;
                            refs[total_vertices + v] = corner_at(rep);
                            for (ushort corner = ClusterTableLoad(table,rep); corner != 0xffffu; corner = hashes[corner]) {
                                const uint neighbor = corner / 3u;
                                if ((state[neighbor] & 6u) == 0u) {
                                    state[neighbor] |= 4u;
                                    candidates[candidate_count++] = neighbor;
                                }
                            }
                        }
                        local[(total_triangles + triangle_count) * 3u + c] = uchar(v | (c == 0u && (state[chosen] & 1u) ? 0x80u : 0u) | ((state[chosen] & (8u << c)) ? 0x40u : 0u));
                    }
                    anchor += centers[chosen];
                    ++triangle_count;
                }
            }
            BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
            if (chosen == InvalidOffset) break;
        }
        BuildClusterBarrier(simd_size,mem_flags::mem_device);
        for (uint v = lane; v < vertex_count; v += BuildClusterThreads)
            bound_positions[v] = BuildVertexPosition(b, refs[total_vertices + v]);
        BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
        if (lane == 0u) {
            MeshletRecord record{.TriangleOffset = first + total_triangles, .TriangleCount = triangle_count, .VertexOffset = vertex_base + total_vertices, .VertexCount = vertex_count, .LocalTriangleOffset = (first + total_triangles) * 3u, .Primitive = primitive, .GroupIndex = InvalidOffset, .RefinedGroup = InvalidOffset};
            FitClusterBounds(record, local + total_triangles * 3u, [&](uint i) { return bound_positions[i]; });
            b.Records()[tile[3] + record_count++] = record;
            total_vertices += vertex_count; total_triangles += triangle_count;
        }
        BuildClusterBarrier(simd_size,mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { b.Counts(tile_id)[0] = record_count; b.Counts(tile_id)[1] = total_vertices; }
}

kernel void MeshletBuildOffsets(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    if (lane != 0u) return;
    uint records = 0u, vertices = 0u;
    for (uint p = 0u; p < j.PrimitiveCount; ++p) {
        device uint *prim = b.Prim(p);
        prim[6] = records;
        for (uint t = prim[2]; t < prim[2] + prim[3]; ++t) {
            device uint *count = b.Counts(t);
            count[2] = records; count[3] = vertices;
            records += count[0]; vertices += count[1];
        }
        prim[7] = records - prim[6];
    }
    b.Stats()[8] = records; b.Stats()[9] = vertices;
}

kernel void MeshletBuildEmit(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint tile_id=tile_entry.y;
    if (tile_id >= b.Stats()[6]) return;
    device const uint *tile = b.Tile(tile_id), *counts = b.Counts(tile_id);
    const uint primitive = tile[0], vertex_base = tile[1] * b.CornersPerElement();
    device uint *vertices = BindlessBufferMutable(uint, bindless.Buffer, pc.VertexRefsSlot);
    for (uint v = lane; v < counts[1]; v += 256u) {
        const uint source = b.S()[j.VertexScratchOffset + vertex_base + v];
        vertices[j.VertexOffset + counts[3] + v] = j.Topology == 0u ?
            (j.Mesh.TriangleSlot != InvalidSlot ? b.AttributeHandle(source) : j.Mesh.IndexSlotOffset.Offset + source) : source;
    }
    for (uint m = lane; m < counts[0]; m += 256u) {
        auto record = b.Records()[tile[3] + m];
        record.VertexOffset = j.VertexOffset + counts[3] + record.VertexOffset - vertex_base;
        record.TriangleOffset += j.TriangleOffset;
        record.LocalTriangleOffset += j.LocalTriangleOffset;
        record.Topology = j.Topology;
        // A fragment's adopter names its traversal leaf.
        if (j.ExistingPrimitive == InvalidOffset) BindlessBufferMutable(uint,bindless.Buffer,pc.LodLeavesSlot)[j.MeshletOffset+counts[2]+m] = j.NodeOffset+record.Primitive;
        record.Primitive = j.ExistingPrimitive == InvalidOffset ? record.Primitive+j.PrimitiveOffset : j.ExistingPrimitive;
        record.GroupIndex = j.ExistingGroup;
        BindlessBufferMutable(MeshletRecord, bindless.Buffer, pc.MeshletsSlot)[j.MeshletOffset + counts[2] + m] = record;

    }
}

kernel void MeshletBuildPrimitives(uint lane [[thread_index_in_threadgroup]], BUILD_ARGS) {
    BUILD_CONTEXT;
    const uint p=tile_entry.y*256u+lane;
    if (p >= j.PrimitiveCount || j.ExistingPrimitive != InvalidOffset) return;
    device const uint *prim = b.Prim(p);
    const uint node = prim[7] ? j.NodeOffset + p : InvalidOffset;
    BindlessBufferMutable(uint,bindless.Buffer,pc.LodParentsSlot)[j.NodeOffset+p] = InvalidOffset;
    const uint source_primitive = WorkGroupElement(bindless,j.Materials,p);
    BindlessBufferMutable(uint,bindless.Buffer,pc.PrimitiveRoutesSlot)[j.PrimitiveRoutes + source_primitive] = j.PrimitiveOffset+p;
    BindlessBufferMutable(PrimitiveRecord, bindless.Buffer, pc.PrimitivesSlot)[j.PrimitiveOffset + p] = {
        .AuxIndices = j.AuxIndices, .PrimitiveIndex = source_primitive, .PrimitiveMaterialOffset = j.Mesh.PrimitiveMaterialOffset,
        .TriangleOffset = j.TriangleOffset + prim[1], .TriangleCount = prim[0],
        .MeshletCount = prim[7], .Level0Count = prim[7], .LodRootNode = node, .LodFinestNode = node,
        .LodAttributes = ~0u,
    };
    BindlessBufferMutable(LodNode, bindless.Buffer, pc.NodesSlot)[j.NodeOffset + p] = {.Error = INFINITY, .FirstMeshlet = j.MeshletOffset + prim[6], .MeshletCount = prim[7], .MeshletRoot = InvalidOffset};
}

#endif
