#ifndef CORNERNORMALOFFSET_MSL
#define CORNERNORMALOFFSET_MSL

#include "Bindless.metal"

// A negative polar angle or a missing block uses the derived normal.
inline float2 CustomNormalOffset(device const BindlessSet &b, ElementAttributeRef attribute, uint h) {
    if (attribute.ValuesSlot == InvalidSlot) return float2(-1.f, 0.f);
    const uint block = BindlessBuffer(uint, b.Buffer, attribute.BlocksSlot)[h / MeshElementBlockSize];
    return block ? float2(BindlessBuffer(packed_float2, b.Buffer, attribute.ValuesSlot)[(block - 1u) * MeshElementBlockSize + h % MeshElementBlockSize]) : float2(-1.f, 0.f);
}

struct CornerNormalFrame {
    float3 Normal, Ref, Ortho;
};

// Match CornerNormalOffset.h. Polygon neighbors anchor the frame independently
// of which triangle references this corner. Unroll the two edge candidates.
inline CornerNormalFrame ComputeCornerFrame(float3 normal, float3 p0, float3 next, float3 previous) {
    const float3 n = dot(normal, normal) > 0.f ? normal : float3(0, 0, 1);
    const float3 e1 = next - p0, r1 = e1 - n * dot(e1, n);
    const float l1 = length(r1);
    float3 ref;
    if (l1 > 1e-3f * length(e1)) ref = r1 / l1;
    else {
        const float3 e2 = previous - p0, r2 = e2 - n * dot(e2, n);
        const float l2 = length(r2);
        if (l2 > 1e-3f * length(e2)) ref = r2 / l2;
        else ref = normalize(cross(n, abs(n.x) < 0.5f ? float3(1, 0, 0) : float3(0, 1, 0)));
    }
    return {n, ref, cross(n, ref)};
}

inline float2 EncodeNormalOffset(float3 normal, CornerNormalFrame frame) {
    const float x = dot(normal, frame.Ref), y = dot(normal, frame.Ortho);
    // Metal fast atan2(0, 0) may be non-finite.
    // Azimuth has no meaning at a pole.
    const float azimuth = x == 0.f && y == 0.f ? 0.f : atan2(y, x);
    return float2(acos(clamp(dot(normal, frame.Normal), -1.f, 1.f)), azimuth);
}

inline float3 DecodeNormalOffset(float2 offset, CornerNormalFrame frame) {
    return cos(offset.x) * frame.Normal + sin(offset.x) * (cos(offset.y) * frame.Ref + sin(offset.y) * frame.Ortho);
}

#endif
