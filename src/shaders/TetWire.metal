#ifndef TETWIRE_MSL
#define TETWIRE_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "TetWirePushConstants.metal"

// Tetrahedral wireframe for the overlay pass, read straight from the canonical tet arenas.
// Each threadgroup emits up to 48 edges as lines, two dedicated vertices apiece.
constant uint TetWireEdges = 48u;
using TetWireOutput = metal::mesh<LineVaryings, void, TetWireEdges * 2u, TetWireEdges, metal::topology::line>;

[[mesh]] void TetWireMesh(
    TetWireOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant TetWirePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_edge = threadgroup_position.x * TetWireEdges;
    const uint edge_count = min(TetWireEdges, pc.EdgeCount - first_edge);
    output.set_primitive_count(edge_count);
    if (thread_index >= edge_count * 2u) return;

    const uint edge = first_edge + thread_index / 2u;
    const uint point = BindlessBuffer(uint, bindless.Buffer, pc.EdgeIndexSlot)[pc.EdgeIndexOffset + edge * 2u + (thread_index & 1u)];
    const float3 local_pos = float3(BindlessBuffer(packed_float3, bindless.Buffer, pc.PositionSlot)[pc.PositionOffset + point]);
    const Transform world = scene.Models(pc.ModelSlot)[pc.InstanceIndex];
    float4 clip = scene.ViewProj() * float4(trs_transform_point(world, local_pos), 1.0f);
    clip.z -= NdcOffsetFactor(scene); // Push the wireframe in front of faces.

    // The wireframe reads as its own overlay rather than part of the selection, so it keeps the wire colour.
    output.set_vertex(thread_index, MakeLineVertex(clip, WireBaseColor(scene), float2(scene.View.ViewportSize)));
    output.set_index(thread_index, thread_index);
}

#endif
