#include "gpu/AreaShadowBounds.h"
#include "TransformUtils.metal"

kernel void PrepareRayShadowBounds(device const BindlessSet &bindless [[buffer(0)]],
    device AreaShadowBounds &result [[buffer(1)]], constant AreaShadowBoundsParams &pc [[buffer(2)]],
    device const uint *slots [[buffer(3)]], uint lane [[thread_index_in_simdgroup]]) {
    float3 lo(FLT_MAX), hi(-FLT_MAX);
    bool valid = true;
    for (uint i = lane; i < pc.InstanceCount; i += 32u) {
        const uint instance = slots[i];
        const AABB local = BindlessBuffer(AABB, bindless.Buffer, pc.BoundsSlot)[instance];
        const Transform world = BindlessBuffer(Transform, bindless.ModelBuffer, pc.ModelSlot)[instance];
        const auto box = TransformBounds(local, world);
        const float3 extent = abs(box.Ax) + abs(box.Ay) + abs(box.Az);
        // Include rounding when large local coordinates and translation cancel.
        const float3 local_scale = max(abs(float3(local.Min)), abs(float3(local.Max))) * abs(float3(world.S));
        const float3 error = 32.0f * FLT_EPSILON * (1.0f + abs(float3(world.P)) + dot(local_scale, float3(1)));
        const float3 a = box.Center - extent - error, b = box.Center + extent + error;
        valid = valid && all(float3(local.Min) <= float3(local.Max)) && all(isfinite(a)) && all(isfinite(b));
        lo = min(lo, a);
        hi = max(hi, b);
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    valid = simd_all(valid);
    if (lane == 0u) result = { {packed_float3(lo), packed_float3(hi)}, uint(valid), pc.CullBackfaces };
}
