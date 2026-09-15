#include "MeshletResolve.metal"
#include "gpu/MeshletLimit.h"
#include "MeshletNonTriangle.metal"
#include "gpu/MeshVertexConstant.h"
#include "gpu/VisibilityId.h"
#include "VertexTransform.metal"

constant bool NonTriangleTopology [[function_constant(uint(MeshVertexConstant::NonTriangleTopology))]];

// Emit one attribute-carrying vertex per corner because indexed mesh output produces nondeterministic attributes on this driver.
// Opaque meshlets emit welded positions and fetch attributes during shading.
using MeshletOutput = metal::mesh<MeshletVertexVaryings, void, uint(MeshletLimit::MaxTriangles) * 3u, uint(MeshletLimit::MaxTriangles), metal::topology::triangle>;
using MeshletVisibilityOutput = metal::mesh<MeshletPositionVaryings, MeshletVisibilityPrimitiveVaryings, uint(MeshletLimit::MaxVertices), uint(MeshletLimit::MaxTriangles), metal::topology::triangle>;

inline uint MeshletVisibilityId(uint visible_index, uint triangle) {
    return (visible_index << uint(VisibilityId::TriangleBits)) | triangle;
}

inline uint NonTriangleCorner(uint output_vertex) { return line_quad_corner(output_vertex); }

template <typename Output>
inline uchar EmitTriangleIndices(Output output, device const uchar *triangles, MeshletRecord meshlet, uint t) {
    const uint local_triangle = MeshletLocalTriangleOffset(meshlet) + t * 3u;
    const uchar first_index = triangles[local_triangle];
    output.set_index(t * 3u, uint(first_index & uint(MeshletGeometryEncoding::LocalIndexMask)));
    output.set_index(t * 3u + 1u, uint(triangles[local_triangle + 1u] & uint(MeshletGeometryEncoding::LocalIndexMask)));
    output.set_index(t * 3u + 2u, uint(triangles[local_triangle + 2u] & uint(MeshletGeometryEncoding::LocalIndexMask)));
    return first_index;
}

[[mesh]] void MeshletForwardMesh(
    MeshletOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, threadgroup_position.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint topology = NonTriangleTopology ?
        MeshletPrimitiveTopology(work.Meshlet) : uint(MeshPrimitiveTopology::Triangle);
    if (topology == uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.TriangleCount * 3u) {
        device const uint *triangle_ids = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot);
        device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
        const bool coarse = MeshletCoarse(work.Meshlet);
        const uint local_triangle = thread_index / 3u;
        // Coarse triangles have no source-triangle identity.
        const uint triangle = coarse ? 0u : triangle_ids[work.Meshlet.TriangleOffset + local_triangle];
        const MeshletTriangleCorners corners = ResolveMeshletCorners(
            scene, work.Draw, pc.MeshletVertexSlot, pc.MeshletLocalTriangleSlot, work.Meshlet, work.Primitive, triangle, local_triangle
        );
        const uint vertex_index = corners.CornerIds[thread_index % 3u];
        const uint vertex_id = corners.VertexIds[thread_index % 3u];
        const bool flat_face = (triangles[MeshletLocalTriangleOffset(work.Meshlet) + local_triangle * 3u] & uint(MeshletGeometryEncoding::FlatTriangleBit)) != 0u;
        auto out = ToMeshletVertexVaryings(TransformVertex(scene, work.Draw, vertex_index, vertex_index, vertex_id, false, !flat_face, coarse, corners.CoarseNormal));
        const Transform world = MeshletWorld(scene, work.Draw);
        const auto face = coarse ? MeshletCoarseFace(scene, work.Primitive, work.Instance, world) :
                                   MeshletFace(scene, work.Draw, work.Primitive, work.Instance, world, triangle, flat_face);
        out.FlatWorldNormal = face.FlatWorldNormal;
        out.FaceOverlayFlags = face.FaceOverlayFlags;
        out.MaterialIndex = face.MaterialIndex;
        out.WorldScale = face.WorldScale;
        out.ObjectId = face.ObjectId;
        out.ElementId = face.ElementId;
        out.Topology = uint(uint(MeshPrimitiveTopology::Triangle));
        out.PointCoord = float2(0.0f);
        output.set_vertex(thread_index, out);
        output.set_index(thread_index, thread_index);
    } else if (topology != uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.TriangleCount * 6u) {
        device const uint *element_ids = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot);
        const uint element_index = thread_index / 6u;
        const uint corner = NonTriangleCorner(thread_index);
        const uint element = element_ids[work.Meshlet.TriangleOffset + element_index];
        const uint vertex_id = NonTriangleVertexId(
            bindless, pc.MeshletVertexSlot, work.Meshlet, topology, element_index, corner
        );
        auto out = ToMeshletVertexVaryings(TransformVertex(scene, work.Draw, vertex_id, vertex_id, vertex_id, false, true));
        out.Position = NonTrianglePosition(
            scene, bindless, pc.MeshletVertexSlot, work.Draw, work.Meshlet, topology, element_index, corner
        );
        out.FlatWorldNormal = float3(0.0f);
        out.FaceOverlayFlags = 0u;
        out.MaterialIndex = MeshletPrimitiveMaterialIndex(scene, work.Primitive);
        const float3 scale = float3(MeshletWorld(scene, work.Draw).S);
        out.WorldScale = (scale.x + scale.y + scale.z) / 3.0f;
        out.ObjectId = work.Instance.ObjectId;
        out.ElementId = work.Instance.ElementIdOffset + element + 1u;
        out.Topology = topology;
        out.PointCoord = PointQuadCorners[corner] * 0.5f + 0.5f;
        // Alpha zero marks an unselected instance for fill recoloring during shading.
        out.Color = scene.View.InteractionMode == InteractionMode::Object && scene.View.ShowOverlays != 0u ?
            scene.ObjectSelectionColor(scene.InstanceState(work.Draw), float4(0.0f)) : float4(0.0f);
        output.set_vertex(thread_index, out);
        output.set_index(thread_index, thread_index);
    }
    output.set_primitive_count(work.Meshlet.TriangleCount * (topology == uint(MeshPrimitiveTopology::Triangle) ? 1u : 2u));
}

[[mesh]] void MeshletVisibilityMesh(
    MeshletVisibilityOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, threadgroup_position.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint topology = MeshletPrimitiveTopology(work.Meshlet);
    if (topology == uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.VertexCount) {
        const uint packed_vertex = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index);
        const uint vertex_id = MeshletVertexId(scene, work.Draw, topology, packed_vertex);
        output.set_vertex(thread_index, MeshletPositionVaryings{MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id)});
    } else if (topology != uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.TriangleCount * 4u) {
        const uint element = thread_index / 4u;
        const uint corner = thread_index & 3u;
        output.set_vertex(thread_index, MeshletPositionVaryings{NonTrianglePosition(
            scene, bindless, pc.MeshletVertexSlot, work.Draw, work.Meshlet, topology, element, corner
        )});
    }
    if (topology == uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.TriangleCount) {
        device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
        EmitTriangleIndices(output, triangles, work.Meshlet, thread_index);
        output.set_primitive(thread_index, MeshletVisibilityPrimitiveVaryings{
            MeshletVisibilityId(work.VisibleIndex, thread_index)
        });
    } else if (topology != uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.TriangleCount * 2u) {
        const uint element = thread_index / 2u;
        const uint triangle_corner = (thread_index & 1u) * 3u;
        output.set_index(thread_index * 3u, element * 4u + LineQuadCornerLut[triangle_corner]);
        output.set_index(thread_index * 3u + 1u, element * 4u + LineQuadCornerLut[triangle_corner + 1u]);
        output.set_index(thread_index * 3u + 2u, element * 4u + LineQuadCornerLut[triangle_corner + 2u]);
        output.set_primitive(thread_index, MeshletVisibilityPrimitiveVaryings{
            MeshletVisibilityId(work.VisibleIndex, thread_index)
        });
    }
    output.set_primitive_count(work.Meshlet.TriangleCount * (topology == uint(MeshPrimitiveTopology::Triangle) ? 1u : 2u));
}
