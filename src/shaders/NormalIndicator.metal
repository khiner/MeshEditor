#ifndef NORMALINDICATOR_MSL
#define NORMALINDICATOR_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"
#include "NormalIndicatorConstant.metal"

// Normal indicators for the overlay pass, generated from the posed positions and normals of the frame.
// Each threadgroup emits a line group of indicators, one thread and two dedicated vertices apiece.
// An indicator's length follows local geometry size, so it stays readable at any mesh density.
constant uint NormalIndicatorCount = OverlayDispatch_NormalIndicatorLines;
constant float NormalIndicatorLengthScale = 0.25f;
// Faces are fan-triangulated from their first corner, so a walk over a face's triangles visits
// its distinct vertices as the first triangle's three, then the last vertex of each triangle after.
constant uint NormalIndicatorMaxFaceCorners = 256u;
using NormalIndicatorOutput = metal::mesh<LineVaryings, void, NormalIndicatorCount * 2u, NormalIndicatorCount, metal::topology::line>;

// Mean length of the edges meeting `vertex_id`, from the CSR vertex-edge adjacency.
inline float MeanIncidentEdgeLength(const thread Scene &scene, DrawData draw, uint vertex_id, float3 position) {
    if (draw.VertexEdgeAdjacencyOffset == INVALID_OFFSET) return 0.0f;
    device const uint *adjacency = scene.Adjacency(scene.View.AdjacencySlot);
    const uint offsets = draw.VertexEdgeAdjacencyOffset;
    const uint items = offsets + draw.VertexCountOrHeadImageSlot + 1u;
    const uint first = adjacency[offsets + vertex_id], last = adjacency[offsets + vertex_id + 1u];
    if (last <= first) return 0.0f;

    device const uint *edges = scene.Indices(draw.IndexSlotOffset.Slot);
    float total = 0.0f;
    for (uint i = first; i < last; ++i) {
        const uint edge = adjacency[items + i];
        const uint from = edges[draw.IndexSlotOffset.Offset + edge * 2u];
        const uint to = edges[draw.IndexSlotOffset.Offset + edge * 2u + 1u];
        total += length(scene.GetLocalPosition(draw, from == vertex_id ? to : from) - position);
    }
    return total / float(last - first);
}

// Local-space start and end of one indicator, its length scaled by the element's own size.
inline void NormalIndicatorSegment(const thread Scene &scene, DrawData draw, uint element, thread float3 &start, thread float3 &end) {
    if (!NormalIndicatorFaces) {
        start = scene.GetLocalPosition(draw, element);
        const float3 normal = scene.GetVertexNormal(draw, element);
        end = start + NormalIndicatorLengthScale * MeanIncidentEdgeLength(scene, draw, element, start) * normal;
        return;
    }

    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    device const uint *triangle_faces = scene.ObjectIds(draw.ObjectIdSlot);
    const uint first_triangle = scene.FaceFirstTriangles(scene.View.FaceFirstTriangleSlot)[draw.FaceFirstTriangleOffset + element];

    float3 sum = float3(0.0f);
    float area = 0.0f;
    uint corners = 0u;
    for (uint i = 0u; i < NormalIndicatorMaxFaceCorners; ++i) {
        const uint triangle = first_triangle + i;
        if (triangle_faces[draw.FaceIdOffset + triangle] != element + 1u) break;
        const uint base = draw.IndexSlotOffset.Offset + triangle * 3u;
        const float3 a = scene.GetLocalPosition(draw, indices[base]);
        const float3 b = scene.GetLocalPosition(draw, indices[base + 1u]);
        const float3 c = scene.GetLocalPosition(draw, indices[base + 2u]);
        area += 0.5f * length(cross(b - a, c - a));
        if (i == 0u) {
            sum = a + b + c;
            corners = 3u;
        } else {
            sum += c;
            corners += 1u;
        }
    }
    start = corners > 0u ? sum / float(corners) : float3(0.0f);
    end = start + NormalIndicatorLengthScale * sqrt(area) * scene.GetFaceNormal(draw, element);
}

[[mesh]] void NormalIndicatorMesh(
    NormalIndicatorOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_element = threadgroup_position.x * NormalIndicatorCount;
    const uint element_count = min(NormalIndicatorCount, pc.ElementCount - first_element);
    output.set_primitive_count(element_count);
    if (thread_index >= element_count) return;

    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];
    float3 start, end;
    NormalIndicatorSegment(scene, draw, first_element + thread_index, start, end);

    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float4 color = float4(float3(NormalIndicatorFaces ? colors.FaceNormal : colors.VertexNormal), 1.0f);
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, endpoint == 0u ? start : end));
        float4 clip = scene.ViewProj() * float4(world_pos, 1.0f);
        clip.z -= NdcOffsetFactor(scene); // Push indicators in front of faces.
        const uint slot = thread_index * 2u + endpoint;
        output.set_vertex(slot, MakeLineVertex(clip, color, float2(scene.View.ViewportSize)));
        output.set_index(slot, slot);
    }
}

#endif
