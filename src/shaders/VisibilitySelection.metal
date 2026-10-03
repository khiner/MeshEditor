#include "gpu/MeshletInstanceFlag.h"
#include "SelectionObjectQuery.metal"
#include "VisibilityCoverage.metal"
#include "gpu/VisibilitySelectionPushConstants.h"

// The outlined owner's object ID, or zero.
inline uint OutlinedObjectId(uint id, device const BindlessSet &bindless, VisibilityShadingPushConstants pc) {
    if (id == VisibilityBackground) return 0u;
    const uint visible_index = (id >> uint(VisibilityId::TriangleBits)) & VisibilityIndexMask;
    const uint instance = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleMeshletSlot)[visible_index].Instance;
    device const InstanceRecord &record = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[
        BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[instance]
    ];
    return (record.Flags & uint(MeshletInstanceFlag::Silhouette)) != 0u ? record.ObjectId : 0u;
}

// Writes each 2x2 block's nearest unoutlined depth to the occluder pyramid's first level.
kernel void OutlineOccluderSeedKernel(
    uint2 block [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::write> occluders [[texture(2)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 extent{visibility.get_width(), visibility.get_height()};
    if (any(block * 2u >= extent)) return;
    float nearest = 1.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint2 pixel = block * 2u + uint2(i & 1u, i >> 1u);
        if (all(pixel < extent) && OutlinedObjectId(visibility.read(pixel).r, bindless, pc) == 0u) nearest = min(nearest, depth.read(pixel).r);
    }
    occluders.write(float4(nearest), block);
}

struct VisibilitySilhouetteTarget {
    float2 DepthObject [[color(0)]];
    float Depth [[depth(any)]];
};

// Seeds outlined owners' depth and object ID from the visibility image.
fragment VisibilitySilhouetteTarget SilhouetteSeedFragment(
    float4 position [[position]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 pixel = uint2(position.xy);
    const uint object_id = OutlinedObjectId(visibility.read(pixel).r, bindless, pc);
    if (object_id == 0u) discard_fragment();
    const float z = depth.read(pixel).r;
    return {{z, float(object_id)}, z};
}

fragment float2 MeshletSilhouetteFragment(
    float4 position [[position]],
    uint primitive_id [[primitive_id]],
    bool front_facing [[front_facing]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint object_id = CoveredMeshletObject(Scene{bindless, view, theme, workspace}, pc, primitive_id, position.xy, front_facing, false);
    return {position.z, float(object_id)};
}

// Decodes visibility IDs only within the host-provided pick or box rectangle.
kernel void VisibilityObjectSelectionKernel(
    uint2 gid [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilitySelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (any(gid >= pc.Extent)) return;
    const uint2 pixel = pc.Origin + gid;
    const uint sample = visibility.read(pixel).r;
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        sample, bindless, view, theme, workspace, pc.Visibility
    );
    if (decoded.Valid) WriteObjectSelect(bindless, pc.Object, pixel, depth.read(pixel).r, decoded.ObjectId);
}

// Depth testing is disabled: each overlapping object contributes its nearest raster hit.
fragment void MeshletObjectPickFragment(
    float4 position [[position]],
    uint primitive_id [[primitive_id]],
    bool front_facing [[front_facing]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilitySelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint object_id = CoveredMeshletObject(
        Scene{bindless, view, theme, workspace}, pc.Visibility, primitive_id, position.xy, front_facing, false
    );
    WriteObjectSelect(bindless, pc.Object, uint2(position.xy), position.z, object_id);
}
