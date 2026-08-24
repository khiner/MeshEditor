#include "MeshletInstanceFlag.metal"
#include "SelectionObjectQuery.metal"
#include "SelectionPickKey.metal"
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

fragment void VisibilityObjectSelectionFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilitySelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        visibility.read(uint2(quad.Position.xy)).r, bindless, view, theme, workspace, pc.Visibility
    );
    if (!decoded.Valid || decoded.ObjectId == 0u) return;
    const uint2 pixel = uint2(quad.Position.xy);
    WriteObjectSelect(bindless, pc.Object, pixel, depth.read(pixel).r, decoded.ObjectId);
}

fragment void VisibilityFaceSelectionFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> depth [[texture(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilitySelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        visibility.read(uint2(quad.Position.xy)).r, bindless, view, theme, workspace, pc.Visibility
    );
    if (!decoded.Valid || decoded.ElementId == 0u ||
        (decoded.InstanceFlags & MeshletInstanceFlag_ElementSelection) == 0u) return;
    const uint2 pixel = uint2(quad.Position.xy);
    WriteElementPick(bindless, pc.Element, pixel, depth.read(pixel).r, decoded.ElementId);
}

fragment void VisibilityFaceBitsetBoxFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilitySelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 pixel = uint2(quad.Position.xy);
    if (pixel.x < pc.Element.Box.x || pixel.x > pc.Element.Box.z || pixel.y < pc.Element.Box.y || pixel.y > pc.Element.Box.w) return;
    const VisibilityMetadata decoded = DecodeVisibilityMetadata(
        visibility.read(uint2(quad.Position.xy)).r, bindless, view, theme, workspace, pc.Visibility
    );
    if (!decoded.Valid || decoded.ElementId == 0u ||
        (decoded.InstanceFlags & MeshletInstanceFlag_ElementSelection) == 0u) return;
    const uint bit = decoded.ElementId - 1u;
    device atomic_uint *bits = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.Element.BoxResultSlot);
    atomic_fetch_or_explicit(&bits[bit >> 5u], 1u << (bit & 31u), memory_order_relaxed);
}
