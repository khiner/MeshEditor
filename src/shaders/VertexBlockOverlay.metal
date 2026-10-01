#ifndef VERTEXBLOCKOVERLAY_MSL
#define VERTEXBLOCKOVERLAY_MSL

#include "CompactPresent.metal"
#include "ElementOverlay.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "VertexBlocks.metal"

using VertexBlockPointOutput = metal::mesh<PointVaryings, void, VertexBlockLanes, VertexBlockLanes, metal::topology::point>;
using VertexBlockSelectOutput = metal::mesh<ElementIdVaryings, void, VertexBlockLanes, VertexBlockLanes, metal::topology::point>;

// Edit, object and excite points, colored by vertex state.
[[mesh]] void VertexBlockPointMesh(
    VertexBlockPointOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VertexBlockPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[VertexBlockSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    // A point sprite spans PointSize pixels, pulled toward the camera as EditPointSprite pulls it.
    const VertexBlockLane work = ResolveVertexBlockLane(scene, pc, group.x, thread_index, PointSize * 0.5f + 0.5f, NdcOffsetFactor(scene) * 1.5f);
    const uint state = work.VertexId != InvalidOffset ? EditVertexState(scene, work.Draw, work.VertexId) : 0u;
    const bool present = work.VertexId != InvalidOffset && (pc.SoundPoints == 0u || (state & STATE_SELECTED) != 0u);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, VertexBlockSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (!present) return;

    PointVaryings out = ElementPointSprite(
        scene, work.Draw, MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), work.VertexId), work.VertexId
    );
    if (pc.SoundPoints != 0u) {
        constant ViewportThemeColors &colors = scene.Theme.Colors;
        out.Color = work.VertexId == work.Instance.ExcitedVertex ? float4(colors.ElementExcited) :
            work.VertexId == work.Instance.ActiveVertex ? float4(float4(colors.ElementActive).rgb, 1.0f) :
                                                         float4(float3(colors.VertexSelected), 1.0f);
    }
    output.set_vertex(compact.x, out);
    output.set_index(compact.x, compact.x);
}

// Vertex ids for picks and box selection.
[[mesh]] void VertexBlockSelectMesh(
    VertexBlockSelectOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VertexBlockPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[VertexBlockSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const VertexBlockLane work = ResolveVertexBlockLane(scene, pc, group.x, thread_index);
    const bool present = work.VertexId != InvalidOffset &&
        (pc.SoundPoints == 0u || EditSelectionBit(scene, work.Draw.Selection.VertexBits, work.VertexId));
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, VertexBlockSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (!present) return;

    ElementIdVaryings out{
        .Position = MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), work.VertexId),
        .PointSize = PointSize,
        .ElementId = (pc.SoundPoints != 0u ? 0u : work.Draw.ElementIdOffset) + work.VertexId + 1u,
    };
    output.set_vertex(compact.x, out);
    output.set_index(compact.x, compact.x);
}

#endif
