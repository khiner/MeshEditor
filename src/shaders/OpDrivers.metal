#include "BakeTexels.metal"
#include "OpCall.metal"

kernel void OpElementsDriver(JOB_TABLE_ARGS(OpElementsJob), OpPrograms programs [[buffer(BufferIndex_Programs)]]) {
    JOB_THREAD(job, i);
    const DomainElement e = ElementAt(b, job.Call.Blocks, i);
    if (!e.Live) return;
    const OpPoint p = OpPointAt(b, job.Call, e);
    uint out[OpMaxOutputWords];
    OpRun(programs, b, job.Call.Program, p, out);
    for (uint o = 0u, word = 0u; o < job.OutputCount; ++o) {
        const OpOutput output = job.Outputs[o];
        const uint n = AttributeComponents(output.Type);
        uint4 value = uint4(0u);
        for (uint c = 0u; c < n; ++c) value[c] = out[word + c];
        WriteNamedAttribute(b, output.Attribute, output.Type, e.Handle, value);
        word += n;
    }
}

kernel void OpMasksDriver(JOB_TABLE_ARGS(OpMasksJob), OpPrograms programs [[buffer(BufferIndex_Programs)]]) {
    JOB_THREAD(job, i);
    const DomainElement e = ElementAt(b, job.Call.Blocks, i);
    if (!e.Live) return;
    const OpPoint p = OpPointAt(b, job.Call, e);
    uint out[OpMaxOutputWords];
    OpRun(programs, b, job.Call.Program, p, out);
    if (out[0] != 0u) WriteMaskBit(BindlessBufferMutable(uint, b.Buffer, job.Masks.Slot) + job.Masks.Offset, e.Address, true);
}

kernel void OpPointDriver(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant OpPointPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    OpPrograms programs [[buffer(BufferIndex_Programs)]]
) {
    OpPoint p;
    OpStart(p, b, pc.Program, pc.Frame, pc.Time);
    uint out[OpMaxOutputWords];
    OpRun(programs, b, pc.Program, p, out);
    device uint *result = BindlessBufferMutable(uint, b.Buffer, pc.Result.Slot) + pc.Result.Offset;
    for (uint w = 0u; w < pc.Words; ++w) result[w] = out[w];
}

// A texel's point is its triangle's point under its center, or the triangle's nearest point within the margin, and its neighbors move its weights by one texel along each axis.
// Inside the triangle, the baked set's texture coordinate is the texel's center.
kernel void OpTexelsDriver(JOB_TABLE_ARGS(OpTexelsJob), OpPrograms programs [[buffer(BufferIndex_Programs)]]) {
    JOB_THREAD(job, i);
    if (i >= job.Width * job.Height) return;
    const uint2 texel{i % job.Width, i / job.Width};
    const float2 center = float2(texel) + 0.5f, size = float2(job.Width, job.Height);
    uint out[OpMaxOutputWords], components = 0u;
    if (const uint held = OpWords(b, job.Texels)[i]) {
        const ConnectivityView conn{b, job.Call.Geometry.Connectivity};
        const uint3 corners = uint3(BindlessBuffer(packed_uint3, b.Buffer, job.TrianglesSlot)[held - 1u]);
        const BakeTriangle t = BakeTriangleOf(b, job.Call.Geometry.Texcoords[job.UvSet], corners, job.Width, job.Height);
        // Without a margin, every texel that holds a triangle is one the triangle covers.
        const bool inside = job.Margin == 0u || BakeCovers(t, center);
        const float3 w = inside ? BakeWeights(t, center) : BakeNearest(t, center);
        if (inside || distance(BakePosition(t, w), center) <= float(job.Margin)) {
            OpPoint p = OpTrianglePoint(b, job.Call, conn, corners, w);
            OpPoint neighbors[2];
            if (job.Call.Program.Needs & OpNeedsNeighbors) {
                neighbors[0] = OpTrianglePoint(b, job.Call, conn, corners, w + BakeWeightsX(t));
                neighbors[1] = OpTrianglePoint(b, job.Call, conn, corners, w + BakeWeightsY(t));
                if (inside) {
                    neighbors[0].Texcoords[job.UvSet] = float2(center.x + 1.0f, size.y - center.y) / size;
                    neighbors[1].Texcoords[job.UvSet] = float2(center.x, size.y - center.y - 1.0f) / size;
                }
                p.Neighbors = neighbors;
            }
            if (inside) p.Texcoords[job.UvSet] = float2(center.x, size.y - center.y) / size;
            OpRun(programs, b, job.Call.Program, p, out);
            components = job.Components;
        }
    }
    OpWriteTexel(b, job.Output, i, job.Stride, out, components);
}

// The texture coordinates of a position on the grid, in texels from its top left.
inline float2 OpGridCoordinates(constant OpGridPushConstants &pc, float2 at) { return float2(at.x / float(pc.Width), 1.0f - at.y / float(pc.Height)); }
// A texel's point is at its center and holds its texture coordinates in every set, and its neighbors are the next texels' along each axis.
kernel void OpGridDriver(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant OpGridPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    OpPrograms programs [[buffer(BufferIndex_Programs)]]
) {
    const uint columns = pc.Width + 2u * pc.Apron;
    if (i >= columns * (pc.Height + 2u * pc.Apron)) return;
    const float2 at = float2(i % columns, i / columns) + 0.5f - float(pc.Apron);
    OpPoint p;
    for (uint s = 0u; s < 4u; ++s) p.Texcoords[s] = OpGridCoordinates(pc, at);
    OpStart(p, b, pc.Program, pc.Frame, pc.Time);
    OpPoint neighbors[2];
    if (pc.Program.Needs & OpNeedsNeighbors) {
        for (uint axis = 0u; axis < 2u; ++axis) {
            neighbors[axis] = p;
            for (uint s = 0u; s < 4u; ++s) neighbors[axis].Texcoords[s] = OpGridCoordinates(pc, at + float2(axis == 0u, axis == 1u));
        }
        p.Neighbors = neighbors;
    }
    uint out[OpMaxOutputWords];
    OpRun(programs, b, pc.Program, p, out);
    OpWriteTexel(b, pc.Output, i, pc.Stride, out, pc.Components);
}
