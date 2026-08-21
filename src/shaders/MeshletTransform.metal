#include "VertexTransform.metal"
#include "InstanceRecord.metal"
#include "MeshletDrawPushConstants.metal"
#include "MeshletRecord.metal"
#include "MeshletRouteState.metal"
#include "PrimitiveRecord.metal"
#include "VisibleMeshlet.metal"

using MeshletOutput = metal::mesh<MeshletVaryings, void, 144, 48, metal::topology::triangle>;

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
    device const VisibleMeshlet *visible = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleMeshletSlot);
    const MeshletRouteState routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot)[0];
    const VisibleMeshlet work = visible[routes.Offsets[pc.Route] + pc.VisibleOffset + threadgroup_position.x];
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot);
    device const InstanceRecord *instances = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot);
    device const MeshletRecord *meshlets = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot);
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
    const InstanceRecord instance = instances[instance_slot];
    const MeshletRecord meshlet = meshlets[work.Meshlet];
    const PrimitiveRecord primitive = primitives[meshlet.Primitive];

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

    if (thread_index < meshlet.TriangleCount * 3u) {
        device const uint *triangle_ids = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot);
        const uint triangle = triangle_ids[meshlet.TriangleOffset + thread_index / 3u];
        const uint vertex_index = (triangle - primitive.FirstTriangle) * 3u + thread_index % 3u;
        const uint vertex_id = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_index];
        MeshletVaryings out = ToMeshletVaryings(TransformVertex(scene, draw, vertex_index, vertex_index, vertex_id));
        out.ObjectId = instance.ObjectId;
        out.ElementId = instance.ElementIdOffset + scene.ObjectIds(draw.ObjectIdSlot)[draw.FaceIdOffset + triangle - primitive.FirstTriangle];
        output.set_vertex(thread_index, out);
        output.set_index(thread_index, thread_index);
    }
    output.set_primitive_count(meshlet.TriangleCount);
}
