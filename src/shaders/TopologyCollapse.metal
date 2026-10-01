#include "TopologyCollapseShared.metal"

#define COLLAPSE_ARGS device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]], constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
#define COLLAPSE_CONTEXT const TopoContext ctx{bindless, pc}; const uint2 tile = ctx.Tile(group); const MeshTopologyJob job = ctx.Jobs()[tile.x]

kernel void TopologyCollapseKeys(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    const uint i = tile.y * 256u + lane;
    if (i >= job.CollapseCount) return;
    const uint v = CollapseVertex(ctx, job, i);
    device uint *keys = ctx.Scratch() + job.CollapseOffset;
    keys[i] = ctx.VertexTargets(job)[v];
    keys[job.CollapseCount + i] = i;
}

kernel void TopologyCollapseHistogram(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    threadgroup atomic_uint counts[16];
    RadixHistogram(CollapseSort(ctx, job, pc.PassParameter), lane, tile.y, counts);
}
kernel void TopologyCollapsePrefix(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    threadgroup uint sums[9];
    RadixPrefix(CollapseSort(ctx, job, pc.PassParameter), lane, tile.y, sl, sg, sums);
}
kernel void TopologyCollapseScatter(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    threadgroup uint groups[128];
    RadixScatter(CollapseSort(ctx, job, pc.PassParameter), lane, tile.y, sl, sg, groups);
}

// A fixed tree scans each tile.
// Only tile tails advance to the next level.
// Sorted component keys let the same reduction handle isolated vertices, disconnected components, and components crossing any number of tiles.
kernel void TopologyCollapseReduce(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    const uint level = pc.PassParameter, count = CollapseLevelCount(job.CollapseCount, level);
    const uint i = tile.y * 256u + lane;
    const auto sort = CollapseSort(ctx, job);
    const uint key = i < count ? CollapseKey(sort, i, level) : InvalidOffset;
    CollapseSum sum{packed_float3(0), 0u};
    if (i < count) {
        if (level == 0u) {
            const uint v = CollapseVertex(ctx, job, sort.Order[i]);
            sum = {packed_float3(ctx.SrcPosition(job, v) - ctx.SrcPosition(job, key)), 1u};
        } else {
            const uint previous_count = CollapseLevelCount(job.CollapseCount, level - 1u);
            sum = CollapseSums(ctx, job, level - 1u)[uint(min((ulong(i) + 1u) * 256u, ulong(previous_count)) - 1u)];
        }
    }
    threadgroup CollapseSum values[256];
    threadgroup uint keys[256];
    values[lane] = sum;
    keys[lane] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1u; step < 256u; step <<= 1u) {
        const bool combine = lane >= step && key == keys[lane - step];
        const CollapseSum previous = combine ? values[lane - step] : CollapseSum{packed_float3(0), 0u};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (combine) sum = AddCollapseSum(previous, sum);
        values[lane] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < count) CollapseSums(ctx, job, level)[i] = sum;
}

// Parent prefixes already contain preceding tiles. Only a tile's leading
// component can continue across its boundary.
kernel void TopologyCollapseCarry(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    const uint level = pc.PassParameter, count = CollapseLevelCount(job.CollapseCount, level);
    const uint i = tile.y * 256u + lane;
    if (count <= 256u || i >= count || tile.y == 0u) return;
    const auto sort = CollapseSort(ctx, job);
    if (CollapseKey(sort, i, level) != CollapseKey(sort, tile.y * 256u - 1u, level)) return;
    device CollapseSum *sums = CollapseSums(ctx, job, level);
    sums[i] = AddCollapseSum(CollapseSums(ctx, job, level + 1u)[tile.y - 1u], sums[i]);
}

// A component's final sorted item owns its destination position. Publishing
// through the count scan avoids a center lookup or a second position write.
kernel void TopologyCollapseCenters(uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]], COLLAPSE_ARGS) {
    COLLAPSE_CONTEXT;
    const uint i = tile.y * 256u + lane;
    if (i >= job.CollapseCount) return;
    const auto sort = CollapseSort(ctx, job);
    const uint root = sort.Keys[sort.Order[i]];
    if (i + 1u < job.CollapseCount && sort.Keys[sort.Order[i + 1u]] == root) return;
    const CollapseSum sum = CollapseSums(ctx, job, 0u)[i];
    const uint destination = ctx.Counts(job, TopoCountVertices)[ctx.VertexEntry(root)];
    ctx.DstVertices(job)[destination].Position = ctx.SrcPosition(job, root) + float3(sum.Delta) / float(sum.Count);
}
