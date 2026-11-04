#pragma once
// A symmetric plane quadric about an origin near its planes, where its cost stays precise in float, as the quadric of a vertex's faces about the vertex.
#include "Bindless.metal"

struct SimplifyQuadric {
    float Q[10];
    float3 Origin;
    static SimplifyQuadric Zero(float3 origin) {
        SimplifyQuadric q;
        for (uint i = 0u; i < 10u; ++i) q.Q[i] = 0.f;
        q.Origin = origin;
        return q;
    }
    static SimplifyQuadric Load(device const float *words) {
        SimplifyQuadric q;
        for (uint i = 0u; i < 10u; ++i) q.Q[i] = words[i];
        q.Origin = float3(words[10], words[11], words[12]);
        return q;
    }
    void Store(device float *words) const {
        for (uint i = 0u; i < 10u; ++i) words[i] = Q[i];
        words[10] = Origin.x;
        words[11] = Origin.y;
        words[12] = Origin.z;
    }
    // The plane n.x + d = 0 in world coordinates, weighted.
    void Add(float3 n, float d, float weight) {
        const float p[4] = {n.x, n.y, n.z, d + dot(n, Origin)};
        uint k = 0u;
        for (uint i = 0u; i < 4u; ++i) {
            for (uint j = i; j < 4u; ++j) Q[k++] += weight * p[i] * p[j];
        }
    }
    void Shift(float3 t) {
        const float3 b = float3(Q[3], Q[6], Q[8]);
        const float3 at = float3(Q[0] * t.x + Q[1] * t.y + Q[2] * t.z, Q[1] * t.x + Q[4] * t.y + Q[5] * t.z, Q[2] * t.x + Q[5] * t.y + Q[7] * t.z);
        Q[9] += 2.f * dot(b, t) + dot(t, at);
        Q[3] += at.x;
        Q[6] += at.y;
        Q[8] += at.z;
        Origin += t;
    }
    void Add(SimplifyQuadric other) {
        other.Shift(Origin - other.Origin);
        for (uint i = 0u; i < 10u; ++i) Q[i] += other.Q[i];
    }
    float Cost(float3 p) const {
        const float3 x = p - Origin;
        return max(0.f, Q[0] * x.x * x.x + 2.f * Q[1] * x.x * x.y + 2.f * Q[2] * x.x * x.z + 2.f * Q[3] * x.x + Q[4] * x.y * x.y + 2.f * Q[5] * x.y * x.z + 2.f * Q[6] * x.y + Q[7] * x.z * x.z + 2.f * Q[8] * x.z + Q[9]);
    }
};
