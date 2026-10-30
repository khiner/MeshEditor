#include <metal_stdlib>
using namespace metal;

// Writes each destination texel as the linear sample of the source level at the texel's center, a 2x2 box for even sizes.
// The source and destination are one array level each, and the grid covers the destination's texels and slices.
kernel void MipDownsampleKernel(
    uint3 id [[thread_position_in_grid]],
    texture2d_array<float, access::sample> source [[texture(0)]],
    texture2d_array<float, access::write> destination [[texture(1)]]
) {
    if (id.x >= destination.get_width() || id.y >= destination.get_height()) return;
    constexpr sampler box(filter::linear, coord::normalized, address::clamp_to_edge);
    const float2 uv = (float2(id.xy) + 0.5f) / float2(destination.get_width(), destination.get_height());
    destination.write(source.sample(box, uv, id.z, level(0.0f)), id.xy, id.z);
}
