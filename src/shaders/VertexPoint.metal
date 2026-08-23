#ifndef VERTEXPOINT_MSL
#define VERTEXPOINT_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"
#include "SoundPointPushConstants.metal"

// Object mode has no element selection, so points take their object's selection color.
inline float4 point_color(const thread Scene &scene, DrawData draw, uint element_state) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    if (scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        return scene.ObjectSelectionColor(scene.InstanceState(draw), float4(float3(colors.Vertex), 1.0f));
    }
    if ((element_state & STATE_ACTIVE) != 0u) return float4(float4(colors.ElementActive).rgb, 1.0f);
    if ((element_state & STATE_SELECTED) != 0u) return float4(float3(colors.VertexSelected), 1.0f);
    return float4(float3(colors.Vertex), 1.0f);
}

// Element points for the overlay pass: edit-mode vertices, excitable sound vertices, and vertex meshes.
// Each threadgroup emits a point group of round sprites, one thread per point.
constant uint PointMeshPoints = OverlayDispatch_PointGroupPoints;
using PointMeshOutput = metal::mesh<PointVaryings, void, PointMeshPoints, PointMeshPoints, metal::topology::point>;

// One point sprite for the draw's vertex at `element`, sized to zero where excite mode hides it.
inline PointVaryings VertexPointSprite(const thread Scene &scene, DrawData draw, uint element) {
    const uint vertex_count = max(draw.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(element, vertex_count - 1u);
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT ?
        uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + idx]) :
        0u;

    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    PointVaryings out;
    out.Color = point_color(scene, draw, element_state);
    out.Position = MeshletPosition(scene, draw, world, idx);
    out.Position.z -= NdcOffsetFactor(scene) * 1.5f; // Push points in front of lines and faces.
    // Excite mode shows selected and active vertices only.
    out.PointSize = scene.View.InteractionMode == InteractionMode_Excite && !is_selected && !is_active ? 0.0f : PointSize;
    return out;
}

// Excite mode points: one sprite per excitable vertex, read from the canonical sound-vertex handles.
// The clip position of the excitable vertex at `point_index` of the dispatch's handle range.
inline float4 SoundPointClip(const thread Scene &scene, device const BindlessSet &bindless, constant SoundPointPushConstants &pc, uint sound_vertex) {
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex);
    return MeshletPosition(scene, draw, scene.Models(draw.ModelSlot)[draw.FirstInstance], sound_vertex);
}

[[mesh]] void SoundPointMesh(
    PointMeshOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SoundPointPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_point = threadgroup_position.x * PointMeshPoints;
    const uint point_count = min(PointMeshPoints, pc.VertexCount - first_point);
    output.set_primitive_count(point_count);
    if (thread_index >= point_count) return;

    const uint sound_vertex = BindlessBuffer(uint, bindless.Buffer, pc.VertexSlot)[pc.VertexOffset + first_point + thread_index];

    constant ViewportThemeColors &colors = scene.Theme.Colors;
    PointVaryings out;
    out.Color = sound_vertex == pc.ExcitedVertex ? float4(colors.ElementExcited) :
        sound_vertex == pc.ActiveVertex          ? float4(float4(colors.ElementActive).rgb, 1.0f) :
                                             float4(float3(colors.VertexSelected), 1.0f);
    out.Position = SoundPointClip(scene, bindless, pc, sound_vertex);
    out.Position.z -= NdcOffsetFactor(scene) * 1.5f; // Push points in front of lines and faces.
    out.PointSize = PointSize;

    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

// The same excitable vertices, emitting their mesh handles for the element pick.
using SoundPointIdOutput = metal::mesh<ElementIdVaryings, void, PointMeshPoints, PointMeshPoints, metal::topology::point>;

[[mesh]] void SoundPointIdMesh(
    SoundPointIdOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SoundPointPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_point = threadgroup_position.x * PointMeshPoints;
    const uint point_count = min(PointMeshPoints, pc.VertexCount - first_point);
    output.set_primitive_count(point_count);
    if (thread_index >= point_count) return;

    const uint sound_vertex = BindlessBuffer(uint, bindless.Buffer, pc.VertexSlot)[pc.VertexOffset + first_point + thread_index];

    ElementIdVaryings out;
    out.ElementId = sound_vertex + 1u;
    out.Position = SoundPointClip(scene, bindless, pc, sound_vertex);
    out.PointSize = PointSize;
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

[[mesh]] void VertexPointMesh(
    PointMeshOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_point = threadgroup_position.x * PointMeshPoints;
    const uint point_count = min(PointMeshPoints, pc.ElementCount - first_point);
    output.set_primitive_count(point_count);
    if (thread_index >= point_count) return;

    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    output.set_vertex(thread_index, VertexPointSprite(scene, draw, first_point + thread_index));
    output.set_index(thread_index, thread_index);
}

fragment float4 VertexPointFragment(PointVaryings in [[stage_in]], float2 point_coord [[point_coord]]) {
    if (length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();
    return in.Color;
}

#endif
