/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * EEVEE spatial_main gather and ratio accumulation, with native material evaluation.
 */
#pragma once
#include "EeveeDenoise.h"
#include "RayGuide.h"

// Streams the spatial samples through one prepared material evaluation.
struct RaySpatialReuse {
    texture2d<float, access::read> Guide, Samples, Incident, Validity;
    depth2d<float, access::read> Depth;
    int2 Extent, Center, GridOffset;
    int Step;
    uint Count;
    float2x2 Rotation;
    float Bias, Random, Material;
    float3 World;
    float4x4 ClipToWorld;
    float3 Sum, Moment, Radiance;
    float Weight, Closest, SourceInversePdf, Distance;

    bool Sample(uint index, thread float3 &direction) {
        using namespace eevee_denoise;
        float2 xi = hammersley(index, Count);
        xi.y *= 1.0f - Random / float(Count);
        const int2 cell = Center / Step + int2(floor(Rotation * sample_disk(xi) + 0.5f));
        if (any(cell < 0) || any(cell >= (Extent + Step - 1) / Step)) return false;
        const int2 p = min(cell * Step + GridOffset, Extent - 1);
        const float4 g = Guide.read(uint2(p)), data = Samples.read(uint2(p) / uint(Step));
        if (Validity.read(uint2(p)).a == 0.0f || g.w != Material || g.z < 0.0f || data.w <= 0.0f) return false;
        const float4 incoming = Incident.read(uint2(p) / uint(Step));
        const float3 neighbor = RayWorldPoint(float2(p), Depth.read(uint2(p)), float2(Extent), ClipToWorld);
        direction = normalize(data.xyz * incoming.w + (neighbor - World) * Bias);
        Radiance = incoming.rgb;
        Distance = incoming.w;
        SourceInversePdf = data.w;
        return true;
    }
    void Accumulate(float3 outgoing, float pdf) {
        using namespace eevee_denoise;
        const float weight = min(1e30f, pdf * SourceInversePdf);
        const float3 color = to_log(safe_color(outgoing));
        Sum += color * weight;
        Moment += color * color * weight;
        Weight += weight;
        if (weight > 0.0f) Closest = min(Closest, Distance);
    }
};
