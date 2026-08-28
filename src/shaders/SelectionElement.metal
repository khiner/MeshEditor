#ifndef SELECTIONELEMENT_MSL
#define SELECTIONELEMENT_MSL

// The element selection rasters: one mesh entry per element kind, each tagging its fragments with
// the element id the pick and box passes read back.
// Each threadgroup covers a chunk of the draw's element list, with the instance in the grid's second dimension.
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"

constant uint ElementGroupEdges = OverlayDispatch_LineGroupLines;
constant uint ElementGroupPoints = OverlayDispatch_PointGroupPoints;
using ElementIdLineOutput = metal::mesh<ElementIdFragmentVaryings, void, ElementGroupEdges * 2u, ElementGroupEdges, metal::topology::line>;
using ElementIdPointOutput = metal::mesh<ElementIdVaryings, void, ElementGroupPoints, ElementGroupPoints, metal::topology::point>;

inline uint ElementIndex(const thread Scene &scene, DrawData draw, uint index_position) {
    return scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + index_position];
}

// These draws carry base vertices alone, so this reads the vertex arena directly.
inline float4 ElementVertexClip(const thread Scene &scene, DrawData draw, uint idx) {
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];
    return scene.ViewProj() * float4(trs_transform_point(world, float3(vert.Position)), 1.0f);
}

inline float4 ElementClip(const thread Scene &scene, DrawData draw, uint index_position) {
    return ElementVertexClip(scene, draw, ElementIndex(scene, draw, index_position));
}

template<typename OutputT>
inline bool ElementChunkThread(
    thread OutputT &output, const thread Scene &scene, DrawData draw,
    constant OverlayMeshPushConstants &pc, uint elements_per_group, uint indices_per_element,
    uint group_index, uint thread_index, thread uint &index_position
) {
    if (!InstanceInFrustum(scene, draw)) {
        output.set_primitive_count(0u);
        return false;
    }
    const uint first_element = pc.FirstElement + group_index * elements_per_group;
    const uint count = min(elements_per_group, pc.ElementCount - first_element);
    output.set_primitive_count(count);
    if (thread_index >= count * indices_per_element) return false;
    index_position = first_element * indices_per_element + thread_index;
    return true;
}

inline void EmitElementPoint(thread ElementIdPointOutput &output, uint thread_index, uint element_id, float4 clip, float point_size) {
    ElementIdVaryings out;
    out.ElementId = element_id;
    out.Position = clip;
    out.PointSize = point_size;
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

[[mesh]] void SelectionElementVertexMesh(
    ElementIdPointOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    uint index_position = 0u;
    if (!ElementChunkThread(output, scene, draw, pc, ElementGroupPoints, 1u, threadgroup_position.x, thread_index, index_position)) return;

    const uint idx = ElementIndex(scene, draw, index_position);
    EmitElementPoint(output, thread_index, draw.ElementIdOffset + idx + 1u, ElementVertexClip(scene, draw, idx), PointSize);
}

[[mesh]] void SelectionElementEdgeMesh(
    ElementIdLineOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    uint index_position = 0u;
    if (!ElementChunkThread(output, scene, draw, pc, ElementGroupEdges, 2u, threadgroup_position.x, thread_index, index_position)) return;

    ElementIdFragmentVaryings out;
    out.ElementId = draw.ElementIdOffset + index_position / 2u + 1u;
    out.Position = ElementClip(scene, draw, index_position);
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

// A slightly enlarged point reduces sample-center misses on near-zero-length projected edges.
[[mesh]] void SelectionElementEdgePointMesh(
    ElementIdPointOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    uint index_position = 0u;
    if (!ElementChunkThread(output, scene, draw, pc, ElementGroupPoints, 1u, threadgroup_position.x, thread_index, index_position)) return;

    EmitElementPoint(output, thread_index, draw.ElementIdOffset + index_position / 2u + 1u, ElementClip(scene, draw, index_position), 2.0f);
}

#endif
