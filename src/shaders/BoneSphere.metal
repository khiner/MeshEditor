#ifndef BONESPHERE_MSL
#define BONESPHERE_MSL

// Billboard sphere for bone joints: the vertex stage orients a disc toward the camera, and the
// fragment stage ray-traces the sphere it stands for.
#include "Bindless.metal"
#include "BoneUtils.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"

// One emitted vertex of a bone's BoneSphereMesh.
inline BoneSphereVaryings BoneSphereMeshVertexAt(const thread Scene &scene, DrawData draw, uint vertex_id) {
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    BoneSphereVaryings out;
    out.ObjectId = draw.ObjectIdSlot != INVALID_SLOT ? scene.ObjectIds(draw.ObjectIdSlot)[draw.FirstInstance] : 0u;

    // Object mode: neutral shadow, no selection tint.
    const bool is_object_mode = scene.View.InteractionMode == InteractionMode_Object;
    const float3 bone_solid = float3(scene.Theme.Colors.BoneSolid);
    const float3 hint_color = is_object_mode ? bone_solid : bone_joint_wire_color(scene, load_bone_instance_state(scene, draw));
    out.BoneColor = float4(bone_solid, 1.0f);
    out.StateColor = float4(hint_color * hint_color * 0.1f, 1.0f);

    // View-space data for the fragment's ray trace.
    const float3 cam_pos = float3(scene.View.CameraPosition);
    const float3x3 VR = scene.View.ViewRotation.Unpack();
    const float4x4 view_matrix = float4x4(
        float4(VR[0], 0), float4(VR[1], 0), float4(VR[2], 0), float4(-(VR * cam_pos), 1)
    );

    const BoneBillboard bb = bone_sphere_billboard(scene, world, float3(vert.Position));
    out.SphereCenter = (view_matrix * float4(bb.center, 1.0f)).xyz;
    out.SphereRadius = bb.radius;
    out.ViewPos = (view_matrix * float4(bb.world_pos, 1.0f)).xyz;
    out.Position = scene.ViewProj() * float4(bb.world_pos, 1.0f);
    return out;
}

// One threadgroup per bone, emitting that bone's BoneSphereMesh from the shared unit primitive.
// The color and selection-id pipelines both draw this emission, differing only in fragment.
using BoneSphereMeshOutput = metal::mesh<BoneSphereVaryings, void, OverlayDispatch_BoneSphereVertices, 32u, metal::topology::triangle>;

[[mesh]] void BoneSphereMesh(
    BoneSphereMeshOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    output.set_primitive_count(32u);
    if (thread_index >= pc.ElementCount) return;

    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.x);
    output.set_vertex(thread_index, BoneSphereMeshVertexAt(scene, draw, thread_index));
    output.set_index(thread_index, thread_index);
}


fragment OverlayTargetsDepth BoneSphereFragment(
    BoneSphereVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    // Ray-sphere intersection in view space.
    const float3 ray_dir = normalize(in.ViewPos);
    const float3 oc = in.ViewPos - in.SphereCenter;

    const float b = dot(oc, ray_dir);
    const float c = dot(oc, oc) - in.SphereRadius * in.SphereRadius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0f) discard_fragment();

    const float t = -sqrt(discriminant) - b;
    const float3 hit_view = in.ViewPos + ray_dir * t;
    const float3 normal = normalize(hit_view - in.SphereCenter);

    // Blender-style angled lighting, matching the bone fill.
    const float3 light = normalize(float3(0.1f, 0.1f, 0.8f));
    const float fac = clamp(dot(normal, light) * 0.8f + 0.2f, 0.0f, 1.0f);
    const float3 color = mix(in.StateColor.rgb, in.BoneColor.rgb, fac * fac);

    OverlayTargetsDepth out;
    out.Color = float4(color, view.BoneXRay != 0u ? 0.4f : 1.0f);
    out.LineData = float4(0); // Not a line.

    // The hit transformed back to world space, then projected, gives the correct depth.
    const float3 world_hit = transpose(view.ViewRotation.Unpack()) * hit_view + float3(view.CameraPosition);
    const float4 clip = scene.ViewProj() * float4(world_hit, 1.0f);
    // X-ray writes near-far-plane depth so fills do not occlude wires, while still passing the
    // less-than test against the cleared 1.0.
    out.Depth = view.BoneXRay != 0u ? 0.999999f : clip.z / clip.w;
    return out;
}

#endif
