#include "MeshletInstanceFlag.metal"
#include "SelectionObjectQuery.metal"
#include "VisibilityDecode.metal"
#include "VisibilitySelectionPushConstants.metal"

struct VisibilitySilhouetteTarget {
    float2 DepthObject [[color(0)]];
    float Depth [[depth(any)]];
};

fragment VisibilitySilhouetteTarget VisibilitySilhouetteFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 sample = visibility.read(uint2(quad.Position.xy)).rg;
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        sample.x, bindless, view, theme, workspace, pc
    );
    if (!decoded.Valid || (decoded.InstanceFlags & MeshletInstanceFlag_Silhouette) == 0u) discard_fragment();
    const float z = as_type<float>(sample.y);
    return {{z, float(decoded.ObjectId)}, z};
}

// Decodes visibility IDs only within the host-provided pick or box rectangle.
kernel void VisibilityObjectSelectionKernel(
    uint2 gid [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilitySelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (any(gid >= pc.Extent)) return;
    const uint2 pixel = pc.Origin + gid;
    const uint2 sample = visibility.read(pixel).rg;
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        sample.x, bindless, view, theme, workspace, pc.Visibility
    );
    if (decoded.Valid) WriteObjectSelect(bindless, pc.Object, pixel, as_type<float>(sample.y), decoded.ObjectId);
}
