#ifndef ENCLOSINGSPHERE_MSL
#define ENCLOSINGSPHERE_MSL
#include "Bindless.metal"
#include "gpu/MeshletRecord.h"

// Ritter expansion and the AABB-centred sphere both enclose the cluster's vertices.
// The tighter sphere wins, and its radius is measured again against every vertex.
// Cone culling applies only when every triangle's first local index flags a flat face normal.
// Local names the cluster's first local triangle, and position reads a local vertex.
template<typename ReadPosition>
inline void FitClusterBounds(thread MeshletRecord &record, device const uchar *local, ReadPosition position) {
    float3 lo = float3(INFINITY), hi = float3(-INFINITY), lower[3], upper[3];
    for (uint i = 0u; i < record.VertexCount; ++i) {
        const float3 p = position(i);
        for (uint k = 0u; k < 3u; ++k) {
            if (p[k] < lo[k]) { lo[k] = p[k]; lower[k] = p; }
            if (p[k] > hi[k]) { hi[k] = p[k]; upper[k] = p; }
        }
    }
    uint axis = 0u;
    for (uint k = 1u; k < 3u; ++k) if (dot(lower[k] - upper[k], lower[k] - upper[k]) > dot(lower[axis] - upper[axis], lower[axis] - upper[axis])) axis = k;
    float3 center = (lower[axis] + upper[axis]) * 0.5f;
    float radius = distance(lower[axis], upper[axis]) * 0.5f;
    const float3 box_center = (lo + hi) * 0.5f;
    float box_radius = 0.f;
    for (uint i = 0u; i < record.VertexCount; ++i) {
        const float3 p = position(i);
        const float d = distance(center, p);
        if (d > radius) { const float next = (radius + d) * 0.5f; center += (p - center) * ((next - radius) / d); radius = next; }
        box_radius = max(box_radius, distance(box_center, p));
    }
    if (box_radius < radius) { center = box_center; radius = box_radius; }
    for (uint i = 0u; i < record.VertexCount; ++i) radius = max(radius, distance(center, position(i)));
    record.Center = packed_float3(center);
    record.Radius = radius * 1.00001f + 1e-30f;
    record.ConeAxisCutoff = 127u << 24u;
    if (record.Topology != 0u) return;
    float3 normal_sum = float3(0);
    for (uint i = 0u; i < record.TriangleCount; ++i) {
        if ((local[i * 3u] & 0x80u) == 0u) return;
        const float3 a = position(local[i * 3u] & 63u), c = position(local[i * 3u + 1u] & 63u), d = position(local[i * 3u + 2u] & 63u);
        const float3 n = cross(c - a, d - a);
        if (dot(n, n) > 0.f) normal_sum += normalize(n);
    }
    if (dot(normal_sum, normal_sum) == 0.f) return;
    const float3 cone_axis = normalize(normal_sum);
    float minimum = 1.f;
    for (uint i = 0u; i < record.TriangleCount; ++i) {
        const float3 a = position(local[i * 3u] & 63u), c = position(local[i * 3u + 1u] & 63u), d = position(local[i * 3u + 2u] & 63u);
        const float3 n = cross(c - a, d - a);
        if (dot(n, n) > 0.f) minimum = min(minimum, dot(cone_axis, normalize(n)));
    }
    if (minimum <= 0.f) return;
    const int3 quantized = int3(round(cone_axis * 127.f));
    const float error = length(cone_axis - float3(quantized) / 127.f);
    const uint cutoff = uint(min(127.f, ceil((sqrt(max(0.f, 1.f - minimum * minimum)) + error) * 127.f + 1.f)));
    record.ConeAxisCutoff = uint(uchar(quantized.x)) | (uint(uchar(quantized.y)) << 8u) | (uint(uchar(quantized.z)) << 16u) | (cutoff << 24u);
}

struct EnclosingSphereSample { float4 Sphere; float Error; };
struct EnclosingSphereScratch {
    float lows[7][4],highs[7][4],errors[4];
    uint low_ids[7][4],high_ids[7][4];
    float4 sphere,tile[128];
    float merged_error;
};
struct EnclosingSphereResult { float4 Sphere; float Error; };

// Seven-axis extrema, deterministic Ritter expansion and a final containment pass over child spheres, in one 128-thread reduction.
// Readers return a negative radius for an absent sample, and count is nonzero.
template<typename Reader>
inline EnclosingSphereResult EncloseSpheres(thread const Reader &reader,uint count,uint tid,uint lane,uint simd,
    threadgroup EnclosingSphereScratch &scratch) {
    constexpr float3 axes[7]={float3(1,0,0),float3(0,1,0),float3(0,0,1),
        float3(.57735026f,.57735026f,.57735026f),float3(-.57735026f,.57735026f,.57735026f),
        float3(.57735026f,-.57735026f,.57735026f),float3(.57735026f,.57735026f,-.57735026f)};
    float lo[7], hi[7]; uint ilo[7], ihi[7];
    for (uint a=0u; a<7u; ++a) { lo[a]=INFINITY; hi[a]=-INFINITY; ilo[a]=ihi[a]=UINT_MAX; }
    float error=0.f;
    for (ulong i=tid; i<count; i+=128u) {
        const auto sample=reader.Read(uint(i));
        error=max(error,sample.Error);
        if (sample.Sphere.w<0.f) continue;
        for (uint a=0u; a<7u; ++a) {
            const float projection=dot(axes[a],sample.Sphere.xyz);
            const float l=projection-sample.Sphere.w, h=projection+sample.Sphere.w;
            if (l<lo[a]) { lo[a]=l; ilo[a]=uint(i); }
            if (h>hi[a]) { hi[a]=h; ihi[a]=uint(i); }
        }
    }
    for (uint a=0u; a<7u; ++a) {
        const float l=simd_min(lo[a]), h=simd_max(hi[a]);
        const uint il=simd_min(lo[a]==l ? ilo[a] : UINT_MAX), ih=simd_min(hi[a]==h ? ihi[a] : UINT_MAX);
        if (!lane) { scratch.lows[a][simd]=l; scratch.highs[a][simd]=h; scratch.low_ids[a][simd]=il; scratch.high_ids[a][simd]=ih; }
    }
    const float simd_error=simd_max(error);
    if (!lane) scratch.errors[simd]=simd_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) {
        float diameter=-1.f;
        scratch.sphere=float4(0.f);
        for (uint a=0u; a<7u; ++a) {
            float l=INFINITY,h=-INFINITY; uint il=UINT_MAX,ih=UINT_MAX;
            for (uint s=0u; s<4u; ++s) {
                if (scratch.lows[a][s]<l || (scratch.lows[a][s]==l && scratch.low_ids[a][s]<il)) { l=scratch.lows[a][s]; il=scratch.low_ids[a][s]; }
                if (scratch.highs[a][s]>h || (scratch.highs[a][s]==h && scratch.high_ids[a][s]<ih)) { h=scratch.highs[a][s]; ih=scratch.high_ids[a][s]; }
            }
            if (il==UINT_MAX || ih==UINT_MAX) continue;
            const float4 p=reader.Read(il).Sphere, q=reader.Read(ih).Sphere;
            const float d=distance(p.xyz,q.xyz), dr=d+p.w+q.w;
            if (dr>diameter) {
                diameter=dr;
                const float k=d>0.f ? (d+q.w-p.w)/(2.f*d) : 0.f;
                scratch.sphere=float4(p.xyz+(q.xyz-p.xyz)*k,dr/2.f);
            }
        }
        scratch.merged_error=max(max(scratch.errors[0],scratch.errors[1]),max(scratch.errors[2],scratch.errors[3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong base=0u; base<count; base+=128u) {
        if (base+tid<count) scratch.tile[tid]=reader.Read(uint(base+tid)).Sphere;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) for (uint t=0u; t<min(128ul,ulong(count)-base); ++t) {
            const float4 p=scratch.tile[t]; const float d=distance(p.xyz,scratch.sphere.xyz);
            if (p.w<0.f) continue;
            if (d+p.w>scratch.sphere.w) {
                const float k=d>0.f ? (d+p.w-scratch.sphere.w)/(2.f*d) : 0.f;
                scratch.sphere=float4(scratch.sphere.xyz+k*(p.xyz-scratch.sphere.xyz),(scratch.sphere.w+d+p.w)/2.f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float radius=0.f;
    for (ulong i=tid; i<count; i+=128u) {
        const float4 p=reader.Read(uint(i)).Sphere;
        if (p.w<0.f) continue;
        radius=max(radius,distance(p.xyz,scratch.sphere.xyz)+p.w);
    }
    const float simd_radius=simd_max(radius);
    if (!lane) scratch.errors[simd]=simd_radius;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) {
        scratch.sphere.w=max(max(scratch.errors[0],scratch.errors[1]),max(scratch.errors[2],scratch.errors[3]))*(1.f+1e-5f);
        scratch.merged_error*=1.f+1e-5f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return {scratch.sphere,scratch.merged_error};
}
#endif
