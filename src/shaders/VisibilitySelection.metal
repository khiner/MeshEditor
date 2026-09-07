#include "MeshletInstanceFlag.metal"
#include "SelectionObjectQuery.metal"
#include "VisibilityCoverage.metal"
#include "VisibilitySelectionPushConstants.metal"

struct VisibilitySilhouetteTarget {
    float2 DepthObject [[color(0)]];
    float Depth [[depth(any)]];
};

fragment VisibilitySilhouetteTarget VisibilitySilhouetteFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 pixel = uint2(quad.Position.xy);
    const uint sample = visibility.read(pixel).r;
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        sample, bindless, view, theme, workspace, pc
    );
    if (!decoded.Valid || (decoded.InstanceFlags & MeshletInstanceFlag_Silhouette) == 0u) discard_fragment();
    const float z = depth.read(pixel).r;
    return {{z, float(decoded.ObjectId)}, z};
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
