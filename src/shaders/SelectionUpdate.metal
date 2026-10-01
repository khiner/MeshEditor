#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "gpu/SelectionAggregate.h"
#include "gpu/SelectionUpdatePushConstants.h"

// Every handle is a canonical arena handle.
// Derivation reads only the source domain's words, so blocks of other domains update independently.
struct SelectionContext {
    device const BindlessSet &B;
    constant SelectionUpdatePushConstants &Pc;

    ConnectivityView Connectivity() const { return {B, Pc.Connectivity, Pc.FaceCount}; }
    uint Corner(uint h) const { return BindlessBuffer(uint, B.IndexBuffer, Pc.CornersSlot)[h]; }
    bool Live(uint domain, uint handle) const {
        const uint word = BindlessBuffer(MeshElementBlock, B.Buffer, Pc.Blocks[domain])[handle / MeshElementBlockSize].Live[(handle % MeshElementBlockSize) / 32u];
        return (word & (1u << (handle % 32u))) != 0u;
    }
    bool Selected(uint domain, uint handle) const {
        return handle != InvalidOffset && (BindlessBuffer(uint, B.Buffer, Pc.Masks[domain])[handle / 32u] & (1u << (handle % 32u))) != 0u;
    }
    // Both endpoints, with an invalid second endpoint for an unpaired face-less halfedge.
    uint2 EdgeVertices(uint edge) const {
        const auto conn = Connectivity();
        const uint h = conn.EdgeHalfedge(edge), opposite = conn.Opposite(h);
        const uint from = opposite != InvalidOffset ? opposite : Pc.FaceCount ? conn.Previous(h) : InvalidOffset;
        return uint2(Corner(h), from == InvalidOffset ? InvalidOffset : Corner(from));
    }
    bool Derived(uint domain, uint handle) const {
        const auto conn = Connectivity();
        if (domain == 0u) {
            bool selected = false;
            if (Pc.Source == 2u) {
                for (const auto item : conn.Fan(handle)) selected = selected || Selected(2u, item.y);
            } else conn.ForEachIncidentEdge(handle, [&](uint edge) { selected = selected || Selected(1u, edge); });
            return selected;
        }
        if (domain == 1u) {
            if (Pc.Source == 0u) {
                const uint2 v = EdgeVertices(handle);
                return Selected(0u, v.x) && Selected(0u, v.y);
            }
            const uint h = conn.EdgeHalfedge(handle), opposite = conn.Opposite(h);
            return Selected(2u, conn.HalfedgeFace(h)) || (opposite != InvalidOffset && Selected(2u, conn.HalfedgeFace(opposite)));
        }
        const uint2 halfedges = conn.FaceHalfedges(handle);
        for (uint h = halfedges.x; h < halfedges.y; ++h)
            if (!Selected(Pc.Source, Pc.Source == 0u ? Corner(h) : conn.Edge(h))) return false;
        return true;
    }

    // The first mark of a block appends it to the dirty list.
    void Mark(uint domain, uint handle) const {
        if (handle == InvalidOffset) return;
        const uint block = handle / MeshElementBlockSize, bit = 1u << (block % 32u);
        device atomic_uint *word = BindlessBufferMutable(atomic_uint, B.Buffer, Pc.DirtySlot) + SelectionDirtyWord(domain, block);
        if (atomic_fetch_or_explicit(word, bit, memory_order_relaxed) & bit) return;
        device atomic_uint *count = BindlessBufferMutable(atomic_uint, B.Buffer, Pc.List.Slot) + Pc.List.Offset;
        const uint at = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        BindlessBufferMutable(uint, B.Buffer, Pc.List.Slot)[Pc.List.Offset + 3u + at] = (domain << 30u) | block;
    }
};

// One thread per seed bit marks the seed's own block and the blocks of its incident elements.
kernel void MarkSelectionNeighbors(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SelectionUpdatePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Count * 32u) return;
    const SelectionContext ctx{b, pc};
    device const uint *words = BindlessBuffer(uint, b.Buffer, pc.Items.Slot) + pc.Items.Offset + 3u * (i / 32u);
    const SelectionSeed seed{words[0], words[1], words[2]};
    const uint handle = seed.Word * 32u + i % 32u;
    if (i % 32u == 0u && seed.Domain != SelectionHalfedgeDomain) ctx.Mark(seed.Domain, handle);
    if (!(seed.Bits & (1u << (i % 32u))) || !ctx.Live(seed.Domain, handle)) return;
    const auto conn = ctx.Connectivity();
    if (seed.Domain == 0u) {
        for (const auto item : conn.Fan(handle)) ctx.Mark(2u, item.y);
        conn.ForEachIncidentEdge(handle, [&](uint edge) { ctx.Mark(1u, edge); });
    } else if (seed.Domain == 1u) {
        const uint2 v = ctx.EdgeVertices(handle);
        ctx.Mark(0u, v.x);
        ctx.Mark(0u, v.y);
        const uint h = conn.EdgeHalfedge(handle), opposite = conn.Opposite(h);
        ctx.Mark(2u, conn.HalfedgeFace(h));
        if (opposite != InvalidOffset) ctx.Mark(2u, conn.HalfedgeFace(opposite));
    } else if (seed.Domain == 2u) {
        const uint2 halfedges = conn.FaceHalfedges(handle);
        for (uint h = halfedges.x; h < halfedges.y; ++h) {
            ctx.Mark(0u, ctx.Corner(h));
            ctx.Mark(1u, conn.Edge(h));
        }
    } else {
        ctx.Mark(0u, ctx.Corner(handle));
        ctx.Mark(1u, conn.Edge(handle));
        ctx.Mark(2u, conn.HalfedgeFace(handle));
    }
}

// Both block and root aggregation use the same fixed SIMD/word fold order.
// Only the first lane of each SIMD group publishes its partial.
inline SelectionAggregate ReduceSelectionAggregate(
    threadgroup SelectionAggregate *scratch, uint lane, uint simd_group,
    float3 sum, float3 low, float3 high, uint selected, uint live, uint flags
) {
    sum = float3(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z));
    low = float3(simd_min(low.x), simd_min(low.y), simd_min(low.z));
    high = float3(simd_max(high.x), simd_max(high.y), simd_max(high.z));
    selected = simd_sum(selected);
    live = simd_sum(live);
    flags = simd_or(flags);
    if ((lane & 31u) == 0u) {
        scratch[simd_group] = {packed_float3(sum), selected, {packed_float3(low), packed_float3(high)}, live, flags};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SelectionAggregate result{};
    if (lane != 0u) return result;
    float3 total = 0.0f, lo = FLT_MAX, hi = -FLT_MAX;
    for (uint w = 0u; w < MeshElementBlockWords; ++w) {
        total += float3(scratch[w].PositionSum);
        lo = min(lo, float3(scratch[w].Bounds.Min));
        hi = max(hi, float3(scratch[w].Bounds.Max));
        result.Selected += scratch[w].Selected;
        result.LiveCount += scratch[w].LiveCount;
        result.Flags |= scratch[w].Flags;
    }
    result.PositionSum = packed_float3(total);
    result.Bounds = {packed_float3(lo), packed_float3(hi)};
    return result;
}

// One 256-lane group per entry: each SIMD group owns one mask word.
// The group clears the entry's dirty bit and refreshes the block when the mesh owns it.
// Lane partials fold in a fixed order, so a block's aggregate depends only on its contents.
kernel void UpdateSelectionBlocks(
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SelectionUpdatePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup SelectionAggregate scratch[8];
    const SelectionContext ctx{b, pc};
    const uint entry = BindlessBuffer(uint, b.Buffer, pc.Items.Slot)[pc.Items.Offset + group];
    const uint domain = entry >> 30u, block = entry & ((1u << 30u) - 1u);
    if (lane == 0u) atomic_fetch_and_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, pc.DirtySlot) + SelectionDirtyWord(domain, block), ~(1u << (block % 32u)), memory_order_relaxed);
    if (BindlessBuffer(MeshElementBlock, b.Buffer, pc.Blocks[domain])[block].Owner != pc.Owners[domain]) return;
    const uint handle = block * MeshElementBlockSize + lane;
    const bool live = ctx.Live(domain, handle);
    bool selected = ctx.Selected(domain, handle);
    if (pc.Source != InvalidOffset && domain != pc.Source) {
        selected = live && ctx.Derived(domain, handle);
        const uint word = uint((simd_vote::vote_t)simd_ballot(selected));
        if (simd_lane == 0u) BindlessBufferMutable(uint, b.Buffer, pc.Masks[domain])[block * MeshElementBlockWords + simd_group] = word;
    }
    float3 sum = 0.0f, low = FLT_MAX, high = -FLT_MAX;
    uint flags = 0u;
    if (live && domain == 0u) {
        const float3 position = float3(BindlessBuffer(Vertex, b.VertexBuffer, pc.VerticesSlot)[handle].Position);
        low = position;
        high = position;
        if (selected) {
            sum = position;
            device const uchar *sharpness = BindlessBuffer(uchar, b.Buffer, pc.EdgeSharpnessSlot);
            ctx.Connectivity().ForEachIncidentEdge(handle, [&](uint edge) { flags |= sharpness[edge] ? SelectionSelectedSharp : SelectionSelectedSmooth; });
        }
    } else if (live) {
        const bool sharp = BindlessBuffer(uchar, b.Buffer, domain == 1u ? pc.EdgeSharpnessSlot : pc.FaceSharpnessSlot)[handle] != 0u;
        flags = (sharp ? SelectionLiveSharp : SelectionLiveSmooth) | (selected ? (sharp ? SelectionSelectedSharp : SelectionSelectedSmooth) : 0u);
        if (domain == 1u) {
            const auto conn = ctx.Connectivity();
            const uint h = conn.EdgeHalfedge(handle);
            if (h != InvalidOffset && conn.Opposite(h) == InvalidOffset) flags |= SelectionBoundary;
        }
    }
    const auto aggregate = ReduceSelectionAggregate(scratch, lane, simd_group, sum, low, high, uint(selected && live), uint(live), flags);
    if (lane == 0u) BindlessBufferMutable(SelectionAggregate, b.Buffer, pc.Leaves[domain])[block] = aggregate;
}

// One group per domain folds its leaves in ascending block order.
// Each lane owns a fixed contiguous span, so the sum is independent of which blocks changed.
kernel void ReduceSelectionRoots(
    uint domain [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SelectionReducePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup SelectionAggregate scratch[8];
    device const uint *blocks = BindlessBuffer(uint, b.Buffer, pc.Lists[domain].Slot) + pc.Lists[domain].Offset;
    device const SelectionAggregate *leaves = BindlessBuffer(SelectionAggregate, b.Buffer, pc.Leaves[domain]);
    const uint count = pc.Counts[domain], span = (count + 255u) / 256u;
    float3 sum = 0.0f, low = FLT_MAX, high = -FLT_MAX;
    uint selected = 0u, live = 0u, flags = 0u;
    for (uint i = lane * span, end = min(count, i + span); i < end; ++i) {
        const SelectionAggregate leaf = leaves[blocks[i]];
        sum += float3(leaf.PositionSum);
        low = min(low, float3(leaf.Bounds.Min));
        high = max(high, float3(leaf.Bounds.Max));
        selected += leaf.Selected;
        live += leaf.LiveCount;
        flags |= leaf.Flags;
    }
    const auto root = ReduceSelectionAggregate(scratch, lane, simd_group, sum, low, high, selected, live, flags);
    if (lane == 0u) BindlessBufferMutable(SelectionAggregate, b.Buffer, pc.RootsSlot)[pc.Root + domain] = root;
}

// One 256-lane group per listed block writes its selected handles after the block's first output index.
kernel void GatherSelectedElements(
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SelectionGatherPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (group >= pc.Count) return;
    device const uint *pairs = BindlessBuffer(uint, b.Buffer, pc.Blocks.Slot) + pc.Blocks.Offset + 2u * group;
    const uint2 entry = uint2(pairs[0], pairs[1]);
    device const uint *words = BindlessBuffer(uint, b.Buffer, pc.MaskSlot) + entry.x * MeshElementBlockWords;
    const uint word = lane / 32u, bit = 1u << (lane % 32u);
    if (!(words[word] & bit)) return;
    uint rank = popcount(words[word] & (bit - 1u));
    for (uint w = 0u; w < word; ++w) rank += popcount(words[w]);
    BindlessBufferMutable(uint, b.Buffer, pc.Output.Slot)[pc.Output.Offset + entry.y + rank] = entry.x * MeshElementBlockSize + lane;
}
