#include "VertexTransform.metal"
#include "InstanceRecord.metal"
#include "MeshletDrawPushConstants.metal"
#include "MeshletRouteState.metal"
#include "MeshletShared.metal"
#include "PrimitiveRecord.metal"
#include "VisibleMeshlet.metal"

// The attribute-carrying entries emit one output vertex per corner with identity indices.
// Sharing output vertices across primitives (indexed mesh output) delivers nondeterministic
// attribute values on this driver, while the position stream stays exact, so the position-only
// depth entry keeps the welded shared form.
using MeshletOutput = metal::mesh<MeshletVertexVaryings, void, 144, 48, metal::topology::triangle>;
using MeshletDepthOutput = metal::mesh<MeshletDepthVaryings, MeshletEmptyPrimitiveVaryings, 64, 48, metal::topology::triangle>;
using MeshletIdOutput = metal::mesh<MeshletIdVaryings, MeshletIdPrimitiveVaryings, 64, 48, metal::topology::triangle>;

inline DrawData MeshletDraw(PrimitiveRecord primitive, InstanceRecord instance, uint instance_slot) {
    DrawData draw = primitive.Draw;
    draw.FirstInstance = instance_slot;
    draw.BoneDeformOffset = instance.BoneDeformOffset;
    draw.ArmatureDeformOffset = instance.ArmatureDeformOffset;
    draw.MorphDeformOffset = instance.MorphDeformOffset;
    draw.MorphWeightsOffset = instance.MorphWeightsOffset;
    draw.MorphTargetCount = instance.MorphTargetCount;
    draw.PosedPositionOffset = instance.PosedPositionOffset;
    draw.PosedVertexNormalOffset = instance.PosedVertexNormalOffset;
    draw.PosedSeamNormalOffset = instance.PosedSeamNormalOffset;
    draw.PosedFaceNormalOffset = instance.PosedFaceNormalOffset;
    draw.ElementStateSlotOffset = instance.ElementStateSlotOffset;
    draw.HasPendingVertexTransform = instance.HasPendingVertexTransform;
    draw.PrimaryEditInstanceIndex = instance.PrimaryEditInstanceIndex;
    return draw;
}

// The threadgroup's fully resolved draw context. Valid is false when the instance's flags
// exclude it from this route, which draws nothing.
struct MeshletWork {
    InstanceRecord Instance;
    MeshletRecord Meshlet;
    PrimitiveRecord Primitive;
    DrawData Draw;
    bool Valid;
};

inline MeshletWork ResolveMeshletWork(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc, uint group_index
) {
    const MeshletRouteState routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot)[0];
    const VisibleMeshlet work = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleMeshletSlot)
        [routes.Offsets[pc.Route] + pc.VisibleOffset + group_index];
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
    MeshletWork result{};
    result.Instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
    if ((result.Instance.Flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) return result;
    result.Meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[work.Meshlet];
    result.Primitive = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot)[result.Meshlet.Primitive];
    result.Draw = MeshletDraw(result.Primitive, result.Instance, instance_slot);
    result.Valid = true;
    return result;
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

// The corner's per-face values: face state and material, and the face normal when every corner
// of the triangle shades flat.
struct MeshletFaceValues {
    float3 FlatWorldNormal;
    uint FaceOverlayFlags, MaterialIndex;
    float WorldScale;
    uint ObjectId, ElementId;
};

inline MeshletFaceValues MeshletFace(
    const thread Scene &scene, DrawData draw, PrimitiveRecord primitive, InstanceRecord instance,
    Transform world, uint triangle, bool flat_face
) {
    const uint face_id = scene.ObjectIds(draw.ObjectIdSlot)[draw.FaceIdOffset + triangle - primitive.FirstTriangle];
    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT && face_id != 0u ?
        uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + face_id - 1u]) : 0u;
    uint material_index = 0u;
    if (draw.ElementPrimitiveOffset != INVALID_OFFSET && draw.PrimitiveMaterialOffset != INVALID_OFFSET && face_id != 0u) {
        const uint primitive_index = scene.ElementPrimitives(scene.View.ElementPrimitiveSlot)[draw.ElementPrimitiveOffset + face_id - 1u];
        material_index = scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[draw.PrimitiveMaterialOffset + primitive_index];
    }
    float3 flat_world_normal = float3(0.0f);
    if (flat_face) {
        const float3 normal = draw.PosedFaceNormalOffset != INVALID_OFFSET ?
            float3(scene.PosedFaceNormals(scene.View.PosedFaceNormalSlot)[draw.PosedFaceNormalOffset + face_id - 1u]) :
            float3(scene.BaseFaceNormals(scene.View.BaseFaceNormalSlot)[draw.BaseFaceNormalOffset + face_id - 1u]);
        flat_world_normal = trs_transform_normal(world, normal);
    }
    const float3 scale = float3(world.S);
    return {
        flat_world_normal,
        ((element_state & STATE_SELECTED) != 0u ? 1u : 0u) |
            ((element_state & STATE_ACTIVE) != 0u ? 2u : 0u) | (flat_face ? 4u : 0u),
        material_index,
        (scale.x + scale.y + scale.z) / 3.0f,
        instance.ObjectId,
        instance.ElementIdOffset + face_id,
    };
}

inline float4 MeshletPosition(const thread Scene &scene, DrawData draw, Transform world, uint vertex_id) {
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, vertex_id)));
    return scene.ViewProj() * float4(world_pos, 1.0f);
}

[[mesh]] void MeshletTransformMesh(
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

[[mesh]] void MeshletDepthMesh(
    MeshletDepthOutput output,
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
        output.set_vertex(thread_index, MeshletDepthVaryings{MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id)});
    }
    if (thread_index < work.Meshlet.TriangleCount) {
        device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
        EmitTriangleIndices(output, triangles, work.Meshlet, thread_index);
        output.set_primitive(thread_index, MeshletEmptyPrimitiveVaryings{});
    }
    output.set_primitive_count(work.Meshlet.TriangleCount);
}

[[mesh]] void MeshletIdMesh(
    MeshletIdOutput output,
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
    // Shared welded emission: the id attributes are per-primitive, which carries the residual
    // driver nondeterminism, but these routes feed selection and silhouette, not the corpus.
    if (thread_index < work.Meshlet.VertexCount) {
        const uint packed_vertex = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index);
        const uint vertex_id = MeshletVertexId(scene, work.Draw, packed_vertex);
        output.set_vertex(thread_index, MeshletIdVaryings{MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id)});
    }
    if (thread_index < work.Meshlet.TriangleCount) {
        device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
        device const uint *triangle_ids = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot);
        EmitTriangleIndices(output, triangles, work.Meshlet, thread_index);
        const uint triangle = triangle_ids[work.Meshlet.TriangleOffset + thread_index];
        const uint face_id = scene.ObjectIds(work.Draw.ObjectIdSlot)[work.Draw.FaceIdOffset + triangle - work.Primitive.FirstTriangle];
        output.set_primitive(thread_index, MeshletIdPrimitiveVaryings{work.Instance.ObjectId, work.Instance.ElementIdOffset + face_id});
    }
    output.set_primitive_count(work.Meshlet.TriangleCount);
}
