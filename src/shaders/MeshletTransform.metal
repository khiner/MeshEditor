#include "MeshletResolve.metal"
#include "VertexTransform.metal"

// The transparent attribute-carrying entry emits one output vertex per corner with identity indices.
// Sharing output vertices across primitives (indexed mesh output) delivers nondeterministic
// attribute values on this driver, while the position stream stays exact. Opaque meshlets use
// the welded position-only visibility entry below and fetch their attributes during shading.
using MeshletOutput = metal::mesh<MeshletVertexVaryings, void, 144, 48, metal::topology::triangle>;
using MeshletVisibilityOutput = metal::mesh<MeshletPositionVaryings, MeshletVisibilityPrimitiveVaryings, 64, 48, metal::topology::triangle>;

inline uint MeshletVisibilityId(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc, uint group_index, uint triangle
) {
    const MeshletRouteState routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot)[0];
    const uint visible_index = routes.Offsets[pc.Route] + pc.VisibleOffset + group_index;
    return (pc.VisibilityPhase << 31u) | (visible_index << 6u) | triangle;
}


// Emit one triangle's three local vertex indices, returning its packed first byte.
template <typename Output>
inline uchar EmitTriangleIndices(Output output, device const uchar *triangles, MeshletRecord meshlet, uint t) {
    const uint local_triangle = meshlet.LocalTriangleOffset + t * 3u;
    const uchar first_index = triangles[local_triangle];
    output.set_index(t * 3u, uint(first_index & MeshletGeometryEncoding_LocalIndexMask));
    output.set_index(t * 3u + 1u, uint(triangles[local_triangle + 1u] & MeshletGeometryEncoding_LocalIndexMask));
    output.set_index(t * 3u + 2u, uint(triangles[local_triangle + 2u] & MeshletGeometryEncoding_LocalIndexMask));
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
    if (thread_index < work.Meshlet.TriangleCount * 3u) {
        device const uint *triangle_ids = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot);
        device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
        const uint triangle = triangle_ids[work.Meshlet.TriangleOffset + thread_index / 3u];
        const uint vertex_index = (triangle - work.Primitive.FirstTriangle) * 3u + thread_index % 3u;
        const uint vertex_id = scene.Indices(work.Draw.IndexSlotOffset.Slot)[work.Draw.IndexSlotOffset.Offset + vertex_index];
        const bool flat_face = (triangles[work.Meshlet.LocalTriangleOffset + (thread_index / 3u) * 3u] & MeshletGeometryEncoding_FlatTriangleBit) != 0u;
        auto out = ToMeshletVertexVaryings(TransformVertex(scene, work.Draw, vertex_index, vertex_index, vertex_id, false, !flat_face));
        const Transform world = MeshletWorld(scene, work.Draw);
        const auto face = MeshletFace(scene, work.Draw, work.Primitive, work.Instance, world, triangle, flat_face);
        out.FlatWorldNormal = face.FlatWorldNormal;
        out.FaceOverlayFlags = face.FaceOverlayFlags;
        out.MaterialIndex = face.MaterialIndex;
        out.WorldScale = face.WorldScale;
        out.ObjectId = face.ObjectId;
        out.ElementId = face.ElementId;
        output.set_vertex(thread_index, out);
        output.set_index(thread_index, thread_index);
    }
    output.set_primitive_count(work.Meshlet.TriangleCount);
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
    if (thread_index < work.Meshlet.VertexCount) {
        const uint packed_vertex = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index);
        const uint vertex_id = MeshletVertexId(scene, work.Draw, packed_vertex);
        output.set_vertex(thread_index, MeshletPositionVaryings{MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id)});
    }
    if (thread_index < work.Meshlet.TriangleCount) {
        device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
        EmitTriangleIndices(output, triangles, work.Meshlet, thread_index);
        output.set_primitive(thread_index, MeshletVisibilityPrimitiveVaryings{
            MeshletVisibilityId(bindless, pc, threadgroup_position.x, thread_index)
        });
    }
    output.set_primitive_count(work.Meshlet.TriangleCount);
}
