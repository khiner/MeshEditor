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
    texture2d<float, access::read> depth [[texture(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        visibility.read(uint2(quad.Position.xy)).r, bindless, view, theme, workspace, pc
    );
    if (!decoded.Valid || (decoded.InstanceFlags & MeshletInstanceFlag_Silhouette) == 0u) discard_fragment();
    const float z = depth.read(uint2(quad.Position.xy)).r;
    return {{z, float(decoded.ObjectId)}, z};
}

// Selection consumes the same visibility ids as shading. The host limits this grid to the pick or
// box rectangle, so a click decodes only the pixels that can contribute.
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
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        visibility.read(pixel).r, bindless, view, theme, workspace, pc.Visibility
    );
    if (decoded.Valid) WriteObjectSelect(bindless, pc.Object, pixel, depth.read(pixel).r, decoded.ObjectId);
}
