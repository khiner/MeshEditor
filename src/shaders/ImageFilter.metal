// The passes of a separable blur of an image's texels.
#include "Bindless.metal"
#include "gpu/ImageFilterPushConstants.h"

// The kernel's weight of the texel `k` texels from the center: its overlap with a box `width` texels wide, or a Gaussian whose deviation is a sixth of the width, before normalization.
inline float FilterWeight(constant ImageFilterPushConstants &pc, float k) {
    if (pc.Gaussian) {
        const float deviation = pc.Width / 6.0f;
        return exp(-k * k / (2.0f * deviation * deviation));
    }
    const float half_width = 0.5f * pc.Width;
    return max(0.0f, min(k + 0.5f, half_width) - max(k - 0.5f, -half_width));
}
// The input texel, clamped to the input, with zero past its stride.
inline float4 FilterTexel(device const BindlessSet &b, constant ImageFilterPushConstants &pc, int2 at) {
    const uint2 texel = uint2(clamp(at, int2(0), int2(pc.InWidth, pc.InHeight) - 1));
    device const float *values = BindlessBuffer(float, b.Buffer, pc.Input.Slot) + pc.Input.Offset + (texel.y * pc.InWidth + texel.x) * pc.Stride;
    float4 value = float4(0.0f);
    for (uint c = 0u; c < pc.Stride; ++c) value[c] = values[c];
    return value;
}
// The weighted mean of the input texels along `step` around `center`, written to output texel i.
inline void Filter(device const BindlessSet &b, constant ImageFilterPushConstants &pc, uint i, int2 center, int2 step) {
    // A zero width keeps the center alone, and every other kernel reaches the texels it weighs.
    const int radius = pc.Width <= 0.0f ? 0 : int(ceil(pc.Gaussian ? 0.5f * pc.Width : 0.5f * pc.Width + 0.5f));
    float4 sum = float4(0.0f);
    float total = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const float weight = pc.Width <= 0.0f ? 1.0f : FilterWeight(pc, float(k));
        sum += weight * FilterTexel(b, pc, center + k * step);
        total += weight;
    }
    device float *output = BindlessBufferMutable(float, b.Buffer, pc.Output.Slot) + pc.Output.Offset + i * pc.Stride;
    for (uint c = 0u; c < pc.Stride; ++c) output[c] = sum[c] / total;
}

kernel void ImageBlurRows(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ImageFilterPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.OutWidth * pc.OutHeight) return;
    Filter(b, pc, i, int2(i % pc.OutWidth + pc.Offset, i / pc.OutWidth), int2(1, 0));
}

kernel void ImageBlurColumns(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ImageFilterPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.OutWidth * pc.OutHeight) return;
    Filter(b, pc, i, int2(i % pc.OutWidth, i / pc.OutWidth + pc.Offset), int2(0, 1));
}
