// Fast math that keeps Inf and NaN semantics.
#pragma METAL fp math_mode(relaxed)
// Previous evaluated positions follow EEVEE GeometryVelocity (222853b0d7e).
#include "RayDeformationSurface.h"

kernel void RayDeformation(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(0)]],
    constant VisibilityShadingPushConstants &pc [[buffer(1)]],
    constant SceneViewUBO &view [[buffer(2)]],
    constant ViewportThemeUbo &theme [[buffer(3)]],
    constant WorkspaceLights &workspace [[buffer(4)]],
    device const RayMotionTransform *motion [[buffer(10)]],
    device RayDeformedSurface *output [[buffer(11)]],
    device const packed_float3 *previous_positions [[buffer(12)]]
) {
    if (pixel.x >= visibility.get_width() || pixel.y >= visibility.get_height()) return;
    const uint index = pixel.y * visibility.get_width() + pixel.x;
    const auto resolved = ResolveVisibilityId(visibility.read(pixel).r, bindless, view, theme, workspace, pc);
    if (!resolved.Valid || MeshletPrimitiveTopology(resolved.Meshlet) != uint(MeshPrimitiveTopology::Triangle)) return;
    const auto transform = motion[resolved.Draw.FirstInstance];
    if (transform.Valid != RayMotionDeformed) return;
    output[index] = RayPreviousVisibleSurface(pixel, resolved, bindless, view, theme, workspace, transform, previous_positions);
}
