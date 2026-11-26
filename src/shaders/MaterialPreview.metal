// The preview sphere's triangle ids, and the display image of a preview's radiance or value.
#include "Bindless.metal"
#include "gpu/MaterialPattern.h"
#include "gpu/MaterialPreview.h"
#include "tonemapping.metal"

struct PreviewSphereVaryings {
    float4 Position [[position]];
    uint Triangle [[flat]];
};

vertex PreviewSphereVaryings PreviewSphereVertex(
    uint vertex_id [[vertex_id]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    device const PreviewCorner *corners [[buffer(BufferIndex_PreviewCorners)]]
) {
    return {view.ViewProj.Unpack() * float4(float3(corners[vertex_id].Position), 1.0f), vertex_id / 3u};
}
fragment uint PreviewSphereFragment(PreviewSphereVaryings in [[stage_in]]) { return in.Triangle; }

// Writes the display color of each preview pixel: the lit radiance through the rendered view's display transform, a color or a float's grey through the sRGB transfer, or a vector's components mapped from [-1, 1].
// Pixels off the sphere are transparent.
kernel void MaterialPreviewDisplay(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> triangles [[texture(0)]],
    texture2d<float, access::read> radiance [[texture(1)]],
    texture2d<float, access::write> display [[texture(2)]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant MaterialPreviewPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (pixel.x >= display.get_width() || pixel.y >= display.get_height()) return;
    if (triangles.read(pixel).r == InvalidOffset) return display.write(float4(0.0f), pixel);
    if (pc.Display == PreviewDisplay::Lit) {
        const float4 color = radiance.read(pixel);
        return display.write(float4(color.a > 0.0f ? linearToDisplay(color.rgb / color.a) : float3(0.0f), color.a), pixel);
    }
    device const float *words = BindlessBuffer(float, b.Buffer, view.PatternWordSlot) + PatternPixelWord(view, pixel);
    const uint components = view.PatternStride;
    float3 value = components == 1u ? float3(words[0]) : float3(words[0], words[1], components > 2u ? words[2] : 0.0f);
    value = pc.Display == PreviewDisplay::Vector ? value * 0.5f + 0.5f : linearTosRGB(max(value, 0.0f));
    display.write(float4(saturate(value), 1.0f), pixel);
}
