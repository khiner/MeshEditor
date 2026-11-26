#pragma once
// The ray samples whose first surface's material computes inputs, which the hit driver evaluates and the resuming hit kernel shades.
#include "RayTracing.h"
#include "gpu/SurfacePattern.h"

// The hit's ray footprint over the surface's derivatives, which its textures filter over.
inline float HitFootprint(const thread DecodedSurface &surface, float footprint) { return footprint / max(min(length(surface.PositionDx), length(surface.PositionDy)), 1e-6f); }
// A listed sample's index in the ray queue, and its surface's second barycentric coordinate.
inline uint2 PatternHitEntry(device const uint *hits, uint listed) { return uint2(hits[PatternHitEntries + 2u * listed], hits[PatternHitEntries + 2u * listed + 1u]); }
// The instance, primitive and barycentric coordinates of a listed sample's first surface, whose ray origin holds all but the second coordinate.
inline uint4 PatternHitIdentity(RayLightingSample sample, uint barycentric_y) { return uint4(as_type<uint3>(float3(sample.OriginFootprint.xyz)), barycentric_y); }
inline DecodedSurface PatternHitSurface(const thread Scene &scene, const thread RayGeometry &geometry, RayLightingSample sample, uint barycentric_y) {
    const uint4 hit = PatternHitIdentity(sample, barycentric_y);
    return DecodeRaySurface(scene, geometry, hit.x, hit.y, as_type<float2>(hit.zw));
}
