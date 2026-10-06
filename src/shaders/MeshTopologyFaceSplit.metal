#ifndef MESHTOPOLOGYFACESPLIT_MSL
#define MESHTOPOLOGYFACESPLIT_MSL

#include "MeshTopologyFaces.metal"

// A face owns at most n-2 polygons and 3n-6 loop nodes. Count builds this plan
// once for nonplanar splitting or convex partitioning; scatter consumes it.
struct FaceSplitPolygons {
    device uint *Source, *Next, *Prev, *Start, *Length;
    device packed_float3 *Positions;
    float3 Point(uint node) const { return float3(Positions[Source[node]]); }
    float3 Normal(uint first, uint last) const {
        const float3 origin = Point(first);
        float3 normal = float3(0.f), previous = Point(last) - origin;
        uint node = first;
        do {
            const float3 p = Point(node) - origin;
            normal += cross(previous, p);
            previous = p;
            if (node == last) break;
            node = Next[node];
        } while (node != first);
        return NormalizeOrZero(normal);
    }
    float Error(uint first, uint last, float3 normal) const {
        float previous = dot(Point(last), normal), error = 0.f;
        uint node = first;
        do {
            const float z = dot(Point(node), normal);
            error += abs(z - previous);
            previous = z;
            if (node == last) break;
            node = Next[node];
        } while (node != first);
        return error;
    }
    float2 Project(uint node, float3 u, float3 v) const { return TopoProject(Point(node), u, v); }

    bool InsideCorner(uint node, uint other, uint count, float3 u, float3 v) const {
        const float2 p = Project(node, u, v), q = Project(other, u, v);
        uint prev = Prev[node], next = Next[node];
        for (uint k = 0u; k + 3u < count && all(Project(prev, u, v) == p); ++k) prev = Prev[prev];
        for (uint k = 0u; k + 3u < count && all(Project(next, u, v) == p); ++k) next = Next[next];
        const float2 a = Project(prev, u, v), b = Project(next, u, v);
        const bool left_next = TopoCross2(p, b, q) >= 0.f, left_prev = TopoCross2(p, q, a) >= 0.f;
        return TopoCross2(a, p, b) >= 0.f ? left_next && left_prev : left_next || left_prev;
    }
    bool Legal(uint first, uint count, uint a, uint b, float3 u, float3 v) const {
        if (!InsideCorner(a, b, count, u, v) || !InsideCorner(b, a, count, u, v)) return false;
        const float2 p = Project(a, u, v), q = Project(b, u, v);
        uint h = first;
        do {
            const uint next = Next[h];
            if (h != a && h != b && next != a && next != b) {
                const float2 r = Project(h, u, v), s = Project(next, u, v);
                const float pr = TopoCross2(p, q, r), ps = TopoCross2(p, q, s);
                const float rp = TopoCross2(r, s, p), rq = TopoCross2(r, s, q);
                if (((pr < 0.f && ps > 0.f) || (pr > 0.f && ps < 0.f)) &&
                    ((rp < 0.f && rq > 0.f) || (rp > 0.f && rq < 0.f))) return false;
            }
            h = next;
        } while (h != first);
        return true;
    }
};

inline FaceSplitPolygons TopoFaceSplitPolygons(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    device uint *base = ctx.Scratch() + job.FaceLoopOffset + (job.Op == MeshTopologyOp::SplitConcaveFaces ? 29u : 14u) * ctx.SrcHalfedgeDomain(job).Index(range.x);
    return {base, base + 3u * n, base + 6u * n, base + 9u * n, base + 10u * n,
        reinterpret_cast<device packed_float3 *>(base + 11u * n)};
}

inline FaceSplitPolygons TopoInitFaceSplit(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const auto plan = TopoFaceSplitPolygons(ctx, job, f);
    const auto corners = ctx.SrcCorners(job);
    const float3 origin = ctx.SrcPosition(job, corners[range.x]);
    for (uint i = 0u; i < n; ++i) {
        plan.Source[i] = i;
        plan.Next[i] = (i + 1u) % n;
        plan.Prev[i] = (i + n - 1u) % n;
        plan.Positions[i] = packed_float3(ctx.SrcPosition(job, corners[range.x + i]) - origin);
    }
    plan.Start[0] = 0u;
    plan.Length[0] = n;
    return plan;
}

// Blender's connect_verts_nonplanar rule: minimize the sum of absolute plane
// height changes on both loops, then compare their normals to the angle limit.
inline uint3 TopoNonplanarCount(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    if (!ctx.SrcSelectedFace(job, f) || n <= 3u) return uint3(0u, 1u, n);
    const auto plan = TopoInitFaceSplit(ctx, job, f);
    uint faces = 1u, nodes = n;
    const float threshold = cos(job.Param0);
    for (uint face = 0u; face < faces;) {
        const uint first = plan.Start[face], count = plan.Length[face];
        if (count <= 3u) { ++face; continue; }
        const float3 normal = plan.Normal(first, plan.Prev[first]);
        if (dot(normal, normal) == 0.f) { ++face; continue; }
        float3 u, v;
        TopoPlaneFrame(normal, u, v);
        float error = INFINITY, angle_cos = 1.f;
        uint2 best = uint2(InvalidOffset);
        uint length = 0u, a = first;
        for (uint i = 0u; i < count; ++i, a = plan.Next[a]) {
            uint b = plan.Next[plan.Next[a]];
            for (uint j = i + 2u; j < count; ++j, b = plan.Next[b]) {
                if (plan.Next[b] == a) continue;
                const float3 na = plan.Normal(a, b), nb = plan.Normal(b, a);
                if (dot(na, na) == 0.f || dot(nb, nb) == 0.f) continue;
                const float candidate = plan.Error(a, b, na) + plan.Error(b, a, nb);
                if (candidate < error && plan.Legal(first, count, a, b, u, v)) {
                    error = candidate;
                    angle_cos = dot(na, nb);
                    best = uint2(a, b);
                    length = j - i + 1u;
                }
            }
        }
        if (best.x == InvalidOffset || angle_cos >= threshold) { ++face; continue; }
        // Keep a..b in this polygon and b..a in the appended one.
        const uint a_copy = nodes++, b_copy = nodes++;
        const uint before_a = plan.Prev[best.x], after_b = plan.Next[best.y];
        plan.Source[a_copy] = plan.Source[best.x];
        plan.Source[b_copy] = plan.Source[best.y];
        plan.Next[before_a] = a_copy; plan.Prev[a_copy] = before_a;
        plan.Next[a_copy] = b_copy; plan.Prev[b_copy] = a_copy;
        plan.Next[b_copy] = after_b; plan.Prev[after_b] = b_copy;
        plan.Next[best.y] = best.x; plan.Prev[best.x] = best.y;
        plan.Start[face] = best.x; plan.Length[face] = length;
        plan.Start[faces] = b_copy; plan.Length[faces++] = count - length + 2u;
    }
    ctx.FlagFaces(job)[f] = faces;
    return uint3(0u, faces, nodes);
}

inline void TopoFaceSplitEmit(TopoContext ctx, MeshTopologyJob job, uint f, uint fd, uint base) {
    const auto plan = TopoFaceSplitPolygons(ctx, job, f);
    const uint2 range = ctx.SrcFaceRange(job, f);
    const auto corners = ctx.SrcCorners(job);
    for (uint face = 0u; face < ctx.FlagFaces(job)[f]; ++face) {
        const uint first = plan.Start[face], count = plan.Length[face];
        uint node = first;
        for (uint k = 0u; k < count; ++k, node = plan.Next[node]) {
            const uint h = range.x + plan.Source[node], prev = range.x + plan.Source[plan.Prev[node]];
            const bool along = ctx.SrcNext(job, prev) == h;
            ctx.WriteCorner(job, base + k, ctx.Counts(job, TopoCountVertices)[corners[h]], h, h, 0.f,
                along ? h : InvalidOffset, along ? ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)) : true);
        }
        TopoEmitFace(ctx, job, fd + face, base, count, f, true);
        base += count;
    }
}
#endif
