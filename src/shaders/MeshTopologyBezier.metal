#pragma once
// Manifold's smooth patches over the source mesh.
// Each halfedge takes a circular tangent at its start from the derived corner normals, creased where a vertex's corner normals differ.
#include "MeshTopologyContext.metal"

constant float BezierMissingNormal = -3.f;
constant float BezierRelativePrecision = 1e-12f;

inline float3 BezierSafeNormalize(float3 v) {
    const float length2 = dot(v, v);
    return length2 > 0.f ? v * rsqrt(length2) : float3(0.f);
}
inline float BezierAngle(float3 a, float3 b) {
    const float d = dot(a, b);
    return d >= 1.f ? 0.f : d <= -1.f ? M_PI_F : acos(d);
}
inline float BezierWrap(float radians) { return radians < -M_PI_F ? radians + 2.f * M_PI_F : radians > M_PI_F ? radians - 2.f * M_PI_F : radians; }
// A unit vector orthogonal to `ref` in the plane of `ref` and `in`, or of `ref` and `alt` when `in` is colinear with it.
inline float3 BezierOrthogonalTo(float3 in, float3 alt, float3 ref) {
    float3 out = in - ref * dot(in, ref);
    if (dot(out, out) < BezierRelativePrecision * dot(in, in)) out = alt - ref * dot(alt, ref);
    return BezierSafeNormalize(out);
}
// A weighted cubic Bezier tangent along the desired direction that forms a circular arc when mirrored at the far end.
inline float4 BezierCircularTangent(float3 tangent, float3 edge) {
    const float3 dir = BezierSafeNormalize(tangent);
    const float weight = max(.5f, dot(dir, BezierSafeNormalize(edge)));
    const float4 bz2 = float4(dir * (.5f * length(edge)), weight);
    const float4 bz3 = mix(float4(0.f, 0.f, 0.f, 1.f), bz2, 2.f / 3.f);
    return float4(bz3.xyz / bz3.w, bz3.w);
}
// Quaternions as (x, y, z, w).
inline float3 BezierRotate(float4 q, float3 v) { return v + 2.f * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
inline float4 BezierMultiply(float4 a, float4 b) { return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz)); }
inline float4 BezierRotation(float3 a, float3 b) {
    const float d = dot(a, b);
    if (d < -1.f + 1e-6f) {
        const float3 axis = BezierSafeNormalize(abs(a.x) < .9f ? cross(a, float3(1.f, 0.f, 0.f)) : cross(a, float3(0.f, 1.f, 0.f)));
        return float4(axis, 0.f);
    }
    return normalize(float4(cross(a, b), 1.f + d));
}
inline float4 BezierAngleAxis(float angle, float3 axis) { return float4(axis * sin(.5f * angle), cos(.5f * angle)); }
inline float4 BezierFromColumns(float3 x, float3 y, float3 z) {
    const float4 q = float4(x.x - y.y - z.z, y.y - x.x - z.z, z.z - x.x - y.y, x.x + y.y + z.z);
    const float4 s[4] = {float4(1.f, x.y + y.x, z.x + x.z, y.z - z.y), float4(x.y + y.x, 1.f, y.z + z.y, z.x - x.z), float4(x.z + z.x, y.z + z.y, 1.f, x.y - y.x), float4(y.z - z.y, z.x - x.z, x.y - y.x, 1.f)};
    uint largest = 0u;
    for (uint i = 1u; i < 4u; ++i) largest = q[i] > q[largest] ? i : largest;
    float4 r;
    for (uint i = 0u; i < 4u; ++i) r[i] = sqrt(max(0.f, 1.f + q[i]));
    r = normalize(r);
    for (uint i = 0u; i < 4u; ++i) r[i] = copysign(r[i], s[largest][i]);
    return r;
}
inline float4 BezierSlerpWay(float4 x, float4 y, float a, bool long_way) {
    float4 z = y;
    float cosine = dot(x, y);
    if ((cosine < 0.f) != long_way) {
        z = -y;
        cosine = -cosine;
    }
    if (abs(cosine) > 1.f - FLT_EPSILON) return x * (1.f - a) + z * a;
    const float angle = acos(cosine);
    return (x * sin((1.f - a) * angle) + z * sin(a * angle)) * (1.f / sin(angle));
}
inline float4 BezierHomogeneous(float4 v) { return float4(v.xyz * v.w, v.w); }
inline float4 BezierHomogeneous(float3 v) { return float4(v, 1.f); }
inline float3 BezierHNormalize(float4 v) { return v.w == 0.f ? v.xyz : v.xyz / v.w; }
inline float4 BezierControl(float3 point, float4 tangent) { return BezierHomogeneous(float4(point + tangent.xyz, tangent.w)); }
struct BezierLinear {
    float4 P0, P1;
};
inline BezierLinear BezierCubic2Linear(float4 p0, float4 p1, float4 p2, float4 p3, float x) {
    const float4 p12 = mix(p1, p2, x);
    return {mix(mix(p0, p1, x), p12, x), mix(p12, mix(p2, p3, x), x)};
}
inline float3 BezierPoint(BezierLinear points, float x) { return BezierHNormalize(mix(points.P0, points.P1, x)); }
inline float3 BezierTangent(BezierLinear points) { return BezierSafeNormalize(BezierHNormalize(points.P1) - BezierHNormalize(points.P0)); }
inline float3 BezierRotateFromTo(float3 v, float4 start, float4 end) { return BezierRotate(end, BezierRotate(quat_conjugate(start), v)); }
// The point and tangent across an edge's Bezier at `x`, with the tangent frame slerped between the ends.
inline BezierLinear Bezier2Bezier(float3 corner0, float3 corner1, float4 tangent_x0, float4 tangent_x1, float4 tangent_y0, float4 tangent_y1, float x, float3 anchor) {
    const BezierLinear bez = BezierCubic2Linear(BezierHomogeneous(corner0), BezierControl(corner0, tangent_x0), BezierControl(corner1, tangent_x1), BezierHomogeneous(corner1), x);
    const float3 end = BezierPoint(bez, x);
    const float3 tangent = BezierTangent(bez);
    const float3 n_tangent_x0 = BezierSafeNormalize(tangent_x0.xyz), n_tangent_x1 = -BezierSafeNormalize(tangent_x1.xyz);
    const float3 bi_tangent0 = BezierOrthogonalTo(tangent_y0.xyz, anchor - corner0, n_tangent_x0), bi_tangent1 = BezierOrthogonalTo(tangent_y1.xyz, anchor - corner1, n_tangent_x1);
    const float4 q0 = BezierFromColumns(n_tangent_x0, bi_tangent0, cross(n_tangent_x0, bi_tangent0));
    const float4 q1 = BezierFromColumns(n_tangent_x1, bi_tangent1, cross(n_tangent_x1, bi_tangent1));
    const float3 edge = corner1 - corner0;
    const bool long_way = dot(n_tangent_x0, edge) + dot(n_tangent_x1, edge) < 0.f;
    const float4 q_tmp = BezierSlerpWay(q0, q1, x, long_way);
    const float4 q = BezierMultiply(BezierRotation(BezierRotate(q_tmp, float3(1.f, 0.f, 0.f)), tangent), q_tmp);
    const float3 delta = mix(BezierRotateFromTo(tangent_y0.xyz, q0, q), BezierRotateFromTo(tangent_y1.xyz, q1, q), x);
    const float delta_w = tangent_y0.w * (1.f - x) + tangent_y1.w * x;
    return {BezierHomogeneous(end), float4(delta, delta_w)};
}
inline float3 Bezier2D(float3 c0, float3 c1, float3 c2, float3 c3, float4 tx0, float4 tx1, float4 tx2, float4 tx3, float4 ty0, float4 ty1, float4 ty2, float4 ty3, float x, float y, float3 centroid) {
    const BezierLinear bez0 = Bezier2Bezier(c0, c1, tx0, tx1, ty0, ty1, x, centroid);
    const BezierLinear bez1 = Bezier2Bezier(c2, c3, tx2, tx3, ty2, ty3, 1.f - x, centroid);
    const BezierLinear bez = BezierCubic2Linear(bez0.P0, BezierControl(bez0.P0.xyz, bez0.P1), BezierControl(bez1.P0.xyz, bez1.P1), bez1.P0, y);
    return BezierPoint(bez, y);
}

// The tangents per source halfedge, at the halfedge's start along it, and whether each is fixed.
struct BezierTangents {
    device packed_float4 *Tangent;
    device uint *Fixed;
    TopoSourceDomain Domain;
    uint Slot(uint h) const { return Domain.Index(h); }
    float4 Get(uint h) const { return float4(Tangent[Slot(h)]); }
    void Set(uint h, float4 t) const { Tangent[Slot(h)] = packed_float4(t); }
    bool IsFixed(uint h) const { return Fixed[Slot(h)] != 0u; }
    void Fix(uint h, bool fixed) const { Fixed[Slot(h)] = fixed ? 1u : 0u; }
};
inline BezierTangents TopoTangents(TopoContext ctx, MeshTopologyJob job) {
    return {reinterpret_cast<device packed_float4 *>(ctx.Scratch() + job.TangentOffset), ctx.Scratch() + job.TangentOffset + 4u * job.SrcHalfedgeCount, ctx.SrcHalfedgeDomain(job)};
}
// Manifold's halfedge conventions over the source: a halfedge starts at its previous corner's vertex and ends at its own.
inline uint BezierEnd(TopoContext ctx, MeshTopologyJob job, uint h) { return ctx.SrcCorners(job)[h]; }
inline uint BezierAround(TopoContext ctx, MeshTopologyJob job, uint h) { return ctx.SrcNext(job, ctx.SrcOpposite(job, h)); }
inline float3 BezierFaceNormal(TopoContext ctx, MeshTopologyJob job, uint h) {
    const auto src = ctx.Src(job);
    return float3(BindlessBuffer(packed_float3, ctx.B.Buffer, ctx.Pc.Source.BaseFaceNormalSlot)[src.FaceAddress(src.HalfedgeFace(h))]);
}
inline float3 BezierNormalAt(TopoContext ctx, MeshTopologyJob job, uint h) { return TopoDerivedCornerNormal(ctx, ctx.Pc.Source, ctx.Src(job), ctx.SrcPrev(job, h)); }
inline float3 BezierEdge(TopoContext ctx, MeshTopologyJob job, uint h) { return ctx.SrcPosition(job, BezierEnd(ctx, job, h)) - ctx.SrcPosition(job, ctx.SrcFrom(job, h)); }
inline float4 BezierTangentFromNormal(TopoContext ctx, MeshTopologyJob job, float3 normal, uint h) {
    const float3 edge = BezierEdge(ctx, job, h);
    const float3 edge_normal = BezierFaceNormal(ctx, job, h) + BezierFaceNormal(ctx, job, ctx.SrcOpposite(job, h));
    const float3 bi_tangent = dot(normal, edge_normal) < 0.f ? cross(edge_normal, edge) : cross(normal, edge);
    return BezierCircularTangent(cross(bi_tangent, normal), edge);
}

// Each vertex's halfedges take circular tangents constrained to their corner normals.
// A tangent is fixed where two normals meet in a crease, and tangents align across a lone flat face.
kernel void TopologyTangents(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v >= job.SrcVertexCount) return;
    const uint2 fan = ctx.SrcFan(job, v);
    if (!fan.y) return;
    const BezierTangents tangents = TopoTangents(ctx, job);
    const uint e = ctx.SrcNext(job, ctx.SrcFanCorner(job, fan.x));
    int face_edges[2] = {-1, -1};
    int start_halfedge = -1;
    float3 last_normal = float3(0.f);
    const auto mark_to_align = [&](uint h) {
        if (face_edges[0] == -1) face_edges[0] = int(h);
        else if (face_edges[1] == -1) face_edges[1] = int(h);
        else face_edges[0] = -2;
    };
    const auto calculate = [&](uint h, float3 previous, float3 next) {
        if (all(next == previous)) {
            tangents.Set(h, BezierTangentFromNormal(ctx, job, previous, h));
            return;
        }
        tangents.Fix(h, true);
        face_edges[0] = -2;
        const float3 edge = BezierEdge(ctx, job, h), dir = cross(previous, next);
        tangents.Set(h, BezierCircularTangent(dir * (dot(dir, edge) < 0.f ? -1.f : 1.f), edge));
    };
    uint current = e;
    float3 here = BezierNormalAt(ctx, job, current);
    for (uint guard = 0u; guard <= fan.y; ++guard) {
        const uint next_halfedge = BezierAround(ctx, job, current);
        const float3 next = BezierNormalAt(ctx, job, next_halfedge);
        tangents.Fix(current, false);
        if (any(next != float3(0.f)) && any(here != float3(0.f))) calculate(current, here, next);
        else {
            if (any(here != float3(0.f))) {
                mark_to_align(current);
                last_normal = here;
            } else if (any(next != float3(0.f))) {
                mark_to_align(current);
                if (start_halfedge < 0) start_halfedge = int(current);
            } else if (start_halfedge < 0) start_halfedge = -2;
            tangents.Set(current, float4(last_normal, BezierMissingNormal));
        }
        here = next;
        current = next_halfedge;
        if (current == e) break;
    }
    if (start_halfedge == -1 && face_edges[0] == -1) tangents.Fix(e, true);
    if (start_halfedge != -1 && all(last_normal == float3(0.f))) {
        const float3 normal = float3(ctx.SrcVertexNormals(job)[v]);
        tangents.Fix(e, true);
        current = e;
        for (uint guard = 0u; guard <= fan.y; ++guard) {
            tangents.Set(current, BezierTangentFromNormal(ctx, job, normal, current));
            current = BezierAround(ctx, job, current);
            if (current == e) break;
        }
        return;
    }
    if (start_halfedge >= 0) {
        current = uint(start_halfedge);
        float3 previous = BezierNormalAt(ctx, job, BezierAround(ctx, job, current));
        for (uint guard = 0u; guard <= fan.y; ++guard) {
            if (tangents.Get(current).w == BezierMissingNormal) {
                float3 next = tangents.Get(current).xyz;
                if (all(next == float3(0.f))) next = last_normal;
                calculate(current, previous, next);
            }
            const float3 normal = BezierNormalAt(ctx, job, current);
            if (any(normal != float3(0.f))) previous = normal;
            current = ctx.SrcOpposite(job, ctx.SrcPrev(job, current));
            if (current == uint(start_halfedge)) break;
        }
    }
    if (face_edges[0] >= 0 && face_edges[1] >= 0) {
        const uint a = uint(face_edges[0]), c = uint(face_edges[1]);
        const float3 edge0 = BezierEdge(ctx, job, a), edge1 = BezierEdge(ctx, job, c);
        const float3 tangent = normalize(edge0) - normalize(edge1);
        tangents.Set(a, BezierCircularTangent(tangent, edge0));
        tangents.Set(c, BezierCircularTangent(-tangent, edge1));
        tangents.Fix(a, true);
        tangents.Fix(c, true);
    }
}

// A tangent's angle from the first tangent around the approximate normal, or a full turn for the first tangent itself.
inline float BezierTurn(float3 tangent, float3 first_tangent, float3 approx_normal, bool first) {
    if (first) return 2.f * M_PI_F;
    const float angle = BezierAngle(tangent, first_tangent);
    return dot(approx_normal, cross(tangent, first_tangent)) < 0.f ? 2.f * M_PI_F - angle : angle;
}

// Each fixed halfedge spreads the tangents after it, up to the next fixed one, so their angles share the ratios of the angles between the edges.
kernel void TopologyTangentsDistribute(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint halfedge = ctx.SrcHalfedgeDomain(job).Handle(hi);
    if (halfedge == InvalidOffset) return;
    const BezierTangents tangents = TopoTangents(ctx, job);
    if (!tangents.IsFixed(halfedge)) return;
    const uint start_vertex = ctx.SrcFrom(job, halfedge);
    const float3 approx_normal = float3(ctx.SrcVertexNormals(job)[start_vertex]), center = ctx.SrcPosition(job, start_vertex);
    const auto direction = [&](uint h) { return BezierSafeNormalize(ctx.SrcPosition(job, BezierEnd(ctx, job, h)) - center); };
    const float3 first_tangent = BezierSafeNormalize(tangents.Get(halfedge).xyz);
    float3 normal = float3(0.f), last_edge = direction(halfedge), last_tangent = first_tangent;
    uint count = 0u, current = halfedge;
    float desired_total = 0.f, current_total = 0.f;
    const uint limit = job.SrcHalfedgeCount;
    do {
        current = BezierAround(ctx, job, current);
        const float3 this_edge = direction(current);
        const float3 this_tangent = BezierSafeNormalize(tangents.Get(current).xyz);
        normal += cross(this_tangent, last_tangent);
        desired_total += BezierAngle(this_edge, last_edge);
        const float current_angle = BezierTurn(this_tangent, first_tangent, approx_normal, current == halfedge);
        current_total = current_angle;
        last_edge = this_edge;
        last_tangent = this_tangent;
        ++count;
    } while (!tangents.IsFixed(current) && count < limit);
    if (count == 1u || dot(normal, normal) == 0.f) return;
    const float scale = current_total / desired_total;
    const bool closed = current == halfedge;
    normal = normalize(normal);
    // A closed orbit around one fixed tangent offsets every tangent by the mean error.
    float offset = 0.f;
    if (closed) {
        float desired = 0.f;
        last_edge = direction(halfedge);
        current = halfedge;
        for (uint i = 0u; i < count; ++i) {
            current = BezierAround(ctx, job, current);
            const float3 this_edge = direction(current);
            const float3 this_tangent = BezierSafeNormalize(tangents.Get(current).xyz);
            desired += BezierAngle(this_edge, last_edge);
            const float current_angle = BezierTurn(this_tangent, first_tangent, approx_normal, current == halfedge);
            offset += BezierWrap(current_angle - scale * desired);
            last_edge = this_edge;
        }
        offset /= float(count);
    }
    current = halfedge;
    last_edge = direction(halfedge);
    float desired_previous = 0.f, desired = 0.f;
    for (uint i = 0u; i < count; ++i) {
        current = BezierAround(ctx, job, current);
        if (current != halfedge && tangents.IsFixed(current)) break;
        const float3 this_edge = direction(current);
        const float3 this_tangent = BezierSafeNormalize(tangents.Get(current).xyz);
        desired += BezierAngle(this_edge, last_edge);
        float wanted = scale * desired;
        if (wanted - desired_previous > M_PI_F) wanted = desired_previous + M_PI_F;
        else if (i + 1u < count) {
            const uint after = BezierAround(ctx, job, current);
            const float3 after_edge = direction(after);
            const float next_wanted = scale * (desired + BezierAngle(after_edge, this_edge));
            if (next_wanted - wanted > M_PI_F) wanted = next_wanted - M_PI_F;
        }
        const float current_angle = BezierTurn(this_tangent, first_tangent, approx_normal, current == halfedge);
        const float angle = current_angle - wanted - offset;
        const float4 tangent = tangents.Get(current);
        tangents.Set(current, float4(BezierRotate(BezierAngleAxis(angle, normal), tangent.xyz), tangent.w));
        desired_previous = wanted;
        last_edge = this_edge;
    }
    if (closed) tangents.Fix(halfedge, false);
}

inline float3 BezierInterpolate(TopoContext ctx, MeshTopologyJob job, uint2 range, float4 uvw) {
    const BezierTangents tangents = TopoTangents(ctx, job);
    const uint n = range.y - range.x;
    // Manifold's halfedge i leaves corner i, which is the halfedge arriving at corner i + 1.
    const auto leaving = [&](uint i) { return range.x + (i + 1u) % n; };
    float3 corners[4];
    for (uint i = 0u; i < 4u; ++i) corners[i] = i < n ? ctx.SrcPosition(job, ctx.SrcCorners(job)[range.x + i]) : float3(0.f);
    for (uint i = 0u; i < n; ++i) {
        if (uvw[i] == 1.f) return corners[i];
    }
    float4 pos_h = float4(0.f);
    if (n == 3u) {
        float4 tangent_r[3], tangent_l[3];
        for (uint i = 0u; i < 3u; ++i) {
            tangent_r[i] = tangents.Get(leaving(i));
            tangent_l[i] = tangents.Get(ctx.SrcOpposite(job, leaving((i + 2u) % 3u)));
        }
        const float3 centroid = (corners[0] + corners[1] + corners[2]) / 3.f;
        for (uint i = 0u; i < 3u; ++i) {
            const uint j = (i + 1u) % 3u, k = (i + 2u) % 3u;
            const float x = uvw[k] / (1.f - uvw[i]);
            const BezierLinear bez = Bezier2Bezier(corners[j], corners[k], tangent_r[j], tangent_l[k], tangent_l[j], tangent_r[k], x, centroid);
            const BezierLinear bez1 = BezierCubic2Linear(bez.P0, BezierControl(bez.P0.xyz, bez.P1), BezierControl(corners[i], mix(tangent_r[i], tangent_l[i], x)), BezierHomogeneous(corners[i]), uvw[i]);
            const float3 p = BezierPoint(bez1, uvw[i]);
            pos_h += BezierHomogeneous(float4(p, uvw[j] * uvw[k]));
        }
    } else {
        const float4 tx0 = tangents.Get(leaving(0u)), tx1 = tangents.Get(ctx.SrcOpposite(job, leaving(0u))), tx2 = tangents.Get(leaving(2u)), tx3 = tangents.Get(ctx.SrcOpposite(job, leaving(2u)));
        const float4 ty0 = tangents.Get(ctx.SrcOpposite(job, leaving(3u))), ty1 = tangents.Get(leaving(1u)), ty2 = tangents.Get(ctx.SrcOpposite(job, leaving(1u))), ty3 = tangents.Get(leaving(3u));
        const float3 centroid = (corners[0] + corners[1] + corners[2] + corners[3]) * .25f;
        const float x = uvw[1] + uvw[2], y = uvw[2] + uvw[3];
        const float3 px = Bezier2D(corners[0], corners[1], corners[2], corners[3], tx0, tx1, tx2, tx3, ty0, ty1, ty2, ty3, x, y, centroid);
        const float3 py = Bezier2D(corners[1], corners[2], corners[3], corners[0], ty1, ty2, ty3, ty0, tx1, tx2, tx3, tx0, y, 1.f - x, centroid);
        pos_h += BezierHomogeneous(float4(px, x * (1.f - x)));
        pos_h += BezierHomogeneous(float4(py, y * (1.f - y)));
    }
    const float3 pos = BezierHNormalize(pos_h);
    return all(isfinite(pos)) ? pos : corners[0];
}

kernel void TopologyBezierPositions(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint d = tile.y * ScanTileSize + lane;
    if (d >= job.DstVertexCount) return;
    device const uint *map = ctx.VertexMap(job) + TopoVertexMapWords * d;
    if (map[0] != TopoProvenanceMap || map[3] == InvalidOffset) return;
    const uint2 range = ctx.SrcFaceRange(job, map[3]);
    const uint n = range.y - range.x;
    device const packed_uint2 *pairs = ctx.Provenance(job) + map[1];
    const auto corners = ctx.SrcCorners(job);
    float4 uvw = float4(0.f);
    for (uint k = 0u; k < map[2]; ++k) {
        const uint2 pair = uint2(pairs[k]);
        for (uint i = 0u; i < n; ++i) {
            if (corners[range.x + i] == pair.x) uvw[i] += as_type<float>(pair.y);
        }
    }
    ctx.DstVertices(job)[d].Position = BezierInterpolate(ctx, job, range, uvw);
}
