#ifndef MESHLET_EDIT_OVERLAY_MSL
#define MESHLET_EDIT_OVERLAY_MSL

#include "LineQuad.metal"
#include "MeshletEditEdgeEncoding.metal"
#include "MeshletLimit.metal"
#include "MeshletResolve.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"

constant uint MeshletEditSimdGroups = 5u;
constant uint MeshletEditPointSimdGroups = 2u;
using MeshletEditEdgeOutput = metal::mesh<EdgeQuadVaryings, void, MeshletLimit_MaxTriangles * 4u, MeshletLimit_MaxTriangles * 2u, metal::topology::triangle>;
using MeshletEditPointOutput = metal::mesh<PointVaryings, void, MeshletLimit_MaxVertices, MeshletLimit_MaxVertices, metal::topology::point>;

inline MeshletWork ResolveMeshletEdit(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc, uint group_index
) {
    MeshletWork work = ResolveMeshletWork(bindless, pc, group_index);
    work.Valid = work.Valid && !MeshletCoarse(work.Meshlet) &&
        MeshletPrimitiveTopology(work.Meshlet) == MeshPrimitiveTopology_Triangle;
    return work;
}

inline uint MeshletPackedEditEdge(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    const thread MeshletWork &work, uint local_triangle, uint edge_corner
) {
    const uint source_triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
        work.Meshlet.TriangleOffset + local_triangle
    ];
    return BindlessBuffer(uint, bindless.Buffer, pc.MeshletEditEdgeSlot)[
        work.Draw.EditEdgeOffset + source_triangle * 3u + edge_corner
    ];
}

inline uint2 CompactPresent(
    uint present, uint thread_index, uint lane, threadgroup uint *simd_counts, uint simd_group_count
) {
    const uint simd_rank = simd_prefix_exclusive_sum(present);
    const uint simd_count = simd_sum(present);
    const uint simdgroup = thread_index / 32u;
    if (lane == 0u) simd_counts[simdgroup] = simd_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = simd_rank;
    for (uint i = 0u; i < simdgroup; ++i) rank += simd_counts[i];
    uint count = 0u;
    for (uint i = 0u; i < simd_group_count; ++i) count += simd_counts[i];
    return uint2(rank, count);
}

[[kernel]] void MeshletEditClaim(
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const MeshletWork work = ResolveMeshletEdit(bindless, pc, group.x);
    if (!work.Valid) return;

    if (thread_index < work.Meshlet.VertexCount) {
        const uint packed = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index);
        if ((packed & MeshletGeometryEncoding_EditVertexOwnerBit) != 0u) {
            const uint corner = packed & MeshletGeometryEncoding_CornerMask;
            const uint vertex_id = BindlessBuffer(uint, bindless.IndexBuffer, work.Draw.IndexSlotOffset.Slot)[
                work.Draw.IndexSlotOffset.Offset + corner
            ];
            device atomic_uint *owners = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.EditVertexOwnerSlot);
            atomic_fetch_min_explicit(&owners[work.Draw.VertexOffset + vertex_id], work.MeshletIndex, memory_order_relaxed);
        }
    }

    if (thread_index < work.Meshlet.TriangleCount * 3u) {
        const uint local_triangle = thread_index / 3u;
        const uint edge_corner = thread_index % 3u;
        const uint packed_edge = MeshletPackedEditEdge(bindless, pc, work, local_triangle, edge_corner);
        if (packed_edge != INVALID_OFFSET) {
            const uint edge = packed_edge & MeshletEditEdgeEncoding_EdgeMask;
            device atomic_uint *owners = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.EditEdgeOwnerSlot);
            atomic_fetch_min_explicit(
                &owners[work.Instance.EditEdgeSharpnessOffset + edge], work.MeshletIndex, memory_order_relaxed
            );
        }
    }
}

struct MeshletEditEdgeData {
    float4 Clip0, Clip1;
    float4 Color0, Color1;
    float4 OuterColor;
    float HalfWidth;
    bool Sharp;
};

inline MeshletEditEdgeData ResolveMeshletEditEdge(
    const thread Scene &scene, const thread MeshletWork &work,
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    uint local_triangle, uint edge_corner, uint packed_edge
) {
    device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
    const uint triangle_base = MeshletLocalTriangleOffset(work.Meshlet) + local_triangle * 3u;
    const uint local0 = uint(triangles[triangle_base + edge_corner] & MeshletGeometryEncoding_LocalIndexMask);
    const uint local1 = uint(triangles[triangle_base + (edge_corner + 1u) % 3u] & MeshletGeometryEncoding_LocalIndexMask);
    const uint packed0 = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local0);
    const uint packed1 = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local1);
    const uint vertex0 = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed0);
    const uint vertex1 = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed1);
    const Transform world = MeshletWorld(scene, work.Draw);
    MeshletEditEdgeData result;
    result.Clip0 = MeshletPosition(scene, work.Draw, world, vertex0);
    result.Clip1 = MeshletPosition(scene, work.Draw, world, vertex1);
    result.Clip0.z -= NdcOffsetFactor(scene);
    result.Clip1.z -= NdcOffsetFactor(scene);

    const uint edge = packed_edge & MeshletEditEdgeEncoding_EdgeMask;
    const bool reversed = (packed_edge & MeshletEditEdgeEncoding_ReversedBit) != 0u;
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const bool edit_edge = scene.View.EditElement == Element_Edge;
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const uint state_endpoint = reversed ? 1u - endpoint : endpoint;
        const uint state = uint(scene.ElementStates(work.Instance.EditEdgeStateSlotOffset.Slot)[
            work.Instance.EditEdgeStateSlotOffset.Offset + edge * 2u + state_endpoint
        ]);
        float4 color = WireBaseColor(scene);
        if ((state & STATE_SELECTED) != 0u) {
            color = edit_edge ? float4(float3(colors.EdgeSelected), 1.0f) :
                                float4(float3(colors.EdgeSelectedIncidental), 1.0f);
        }
        if ((state & STATE_ACTIVE) != 0u) color = float4(float4(colors.ElementActive).rgb, 1.0f);
        if (endpoint == 0u) result.Color0 = color;
        else result.Color1 = color;
    }

    result.Sharp = pc.EdgeSharpnessSlot != INVALID_SLOT &&
        uint(scene.ElementStates(pc.EdgeSharpnessSlot)[work.Instance.EditEdgeSharpnessOffset + edge]) != 0u;
    result.OuterColor = result.Sharp ? float4(float3(colors.EdgeSharp), 1.0f) : float4(0.0f);
    const float edge_width = scene.Theme.EdgeWidth;
    result.HalfWidth = edge_width + (result.Sharp ? max(edge_width, 1.0f) : 0.0f) + 0.5f;
    return result;
}

inline EdgeQuadVaryings MeshletEditEdgeCorner(
    const thread Scene &scene, const thread MeshletEditEdgeData &edge, uint quad_corner
) {
    EdgeQuadVaryings out;
    out.OuterColor = edge.OuterColor;
    out.Position = line_quad_position(scene, edge.Clip0, edge.Clip1, quad_corner, edge.HalfWidth);
    if (edge.Sharp) out.Position.z -= 5e-7f * abs(out.Position.w);
    out.EdgeCoord = line_quad_side(quad_corner) * edge.HalfWidth;
    out.Color = line_quad_endpoint(quad_corner) == 0u ? edge.Color0 : edge.Color1;
    return out;
}

inline void EmitMeshletEditEdge(
    thread MeshletEditEdgeOutput output, uint thread_index, uint lane, uint group_index,
    threadgroup uint *simd_counts,
    device const BindlessSet &bindless, constant SceneViewUBO &view,
    constant ViewportTheme &theme, constant WorkspaceLights &workspace,
    constant MeshletDrawPushConstants &pc, uint edge_corner
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletEdit(bindless, pc, group_index);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    uint packed_edge = INVALID_OFFSET;
    if (thread_index < work.Meshlet.TriangleCount) {
        packed_edge = MeshletPackedEditEdge(bindless, pc, work, thread_index, edge_corner);
    }
    const uint edge_id = packed_edge & MeshletEditEdgeEncoding_EdgeMask;
    const uint present = packed_edge != INVALID_OFFSET &&
            BindlessBuffer(uint, bindless.Buffer, pc.EditEdgeOwnerSlot)[work.Instance.EditEdgeSharpnessOffset + edge_id] == work.MeshletIndex ?
        1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (present == 0u) return;

    const uint vertex_base = compact.x * 4u;
    const MeshletEditEdgeData edge = ResolveMeshletEditEdge(
        scene, work, bindless, pc, thread_index, edge_corner, packed_edge
    );
    for (uint corner = 0u; corner < 4u; ++corner) {
        output.set_vertex(vertex_base + corner, MeshletEditEdgeCorner(scene, edge, corner));
    }
    for (uint i = 0u; i < 6u; ++i) {
        output.set_index(compact.x * 6u + i, vertex_base + LineQuadCornerLut[i]);
    }
}

#define MESHLET_EDIT_EDGE_ENTRY(name, edge) \
[[mesh]] void name( \
    MeshletEditEdgeOutput output, uint thread_index [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint3 group [[threadgroup_position_in_grid]], \
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]], \
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]], \
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]], \
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]], \
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]] \
) { \
    threadgroup uint simd_counts[MeshletEditSimdGroups]; \
    EmitMeshletEditEdge(output, thread_index, lane, group.x, simd_counts, bindless, view, theme, workspace, pc, edge); \
}

MESHLET_EDIT_EDGE_ENTRY(MeshletEditEdge0Mesh, 0u)
MESHLET_EDIT_EDGE_ENTRY(MeshletEditEdge1Mesh, 1u)
MESHLET_EDIT_EDGE_ENTRY(MeshletEditEdge2Mesh, 2u)

[[mesh]] void MeshletEditPointMesh(
    MeshletEditPointOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[MeshletEditPointSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletEdit(bindless, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }

    const uint packed = thread_index < work.Meshlet.VertexCount ?
        MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index) : 0u;
    const bool local_owner = thread_index < work.Meshlet.VertexCount &&
        (packed & MeshletGeometryEncoding_EditVertexOwnerBit) != 0u;
    const uint vertex_id = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed);
    const uint present = local_owner &&
            BindlessBuffer(uint, bindless.Buffer, pc.EditVertexOwnerSlot)[work.Draw.VertexOffset + vertex_id] == work.MeshletIndex ?
        1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditPointSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (present == 0u) return;

    const uint state = uint(scene.ElementStates(pc.VertexStateSlot)[work.Draw.VertexOffset + vertex_id]);
    PointVaryings out;
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    out.Color = (state & STATE_ACTIVE) != 0u ? float4(float4(colors.ElementActive).rgb, 1.0f) :
        (state & STATE_SELECTED) != 0u ? float4(float3(colors.VertexSelected), 1.0f) :
                                        float4(float3(colors.Vertex), 1.0f);
    out.Position = MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id);
    out.Position.z -= NdcOffsetFactor(scene) * 1.5f;
    out.PointSize = PointSize;
    output.set_vertex(compact.x, out);
    output.set_index(compact.x, compact.x);
}

#endif
