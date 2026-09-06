#include <metal_stdlib>
using namespace metal;

// BGRA8Unorm reads preserve all 256 channel values; report the first differing byte in BGRA order.
kernel void CompareValidationImages(
    texture2d<float, access::read> expected [[texture(0)]],
    texture2d<float, access::read> actual [[texture(1)]],
    device atomic_uint *first_difference [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= expected.get_width() || pixel.y >= expected.get_height()) return;
    const bool4 different = expected.read(pixel) != actual.read(pixel);
    const uint channel = different.z ? 0u : different.y ? 1u : different.x ? 2u : different.w ? 3u : 4u;
    if (channel < 4u) atomic_fetch_min_explicit(first_difference, 4u * (pixel.y * expected.get_width() + pixel.x) + channel, memory_order_relaxed);
}
