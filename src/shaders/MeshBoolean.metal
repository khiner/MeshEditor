// Strict IEEE arithmetic without contraction keeps the boolean topology predicates consistent across passes.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#include "BooleanCollider.metal"
#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/PolygonTriangulation.h"
#include "MeshBooleanShared.metal"

// Face planes whose normals agree within this count as one plane for a vertex's displacement.
constant float BoolPlaneTolerance = 1e-6f;
constant uint BoolNoCollision = 0xffffffffu;

using BooleanContext = JobContext<BooleanJob>;

// Whether a P edge and a Q edge cross in the xy projection, with the crossing's (x, y) and each edge's z there.
struct BoolCross {
    int S;
    float4 XYZZ;
    bool Valid;
};
inline BoolCross Kernel11(BoolGeometry g, uint p1s, uint p1e, uint q1s, uint q1e) {
    const BoolCrossXY c = ShadowCrossing(g, p1s, p1e, q1s, q1e);
    if (c.S == 0) return {0, float4(0.f), false};
    int s11 = c.S;
    const float4 xyzz = c.XYZZ;
    float z = xyzz.z, w = xyzz.w, dz = 0.f, dw = 0.f;
    if (abs(z - w) <= g.Tol) {
        float3 dprl[2], dqrl[2];
        for (uint j = 0u; j < 2u; ++j) {
            float2 yz, dyz;
            if (c.OfP[j]) {
                dprl[j] = g.Move(c.VertRl[j]);
                BoolLineAt(g.Position(q1s), g.Position(q1e), g.Move(q1s), g.Move(q1e), c.Prl[j].x, dprl[j].x, yz, dyz);
                dqrl[j] = float3(dprl[j].x, dyz.x, dyz.y);
            } else {
                dqrl[j] = g.Move(c.VertRl[j]);
                BoolLineAt(g.Position(p1s), g.Position(p1e), g.Move(p1s), g.Move(p1e), c.Qrl[j].x, dqrl[j].x, yz, dyz);
                dprl[j] = float3(dqrl[j].x, dyz.x, dyz.y);
            }
        }
        float2 zw, dzw;
        BoolCrossingZ(c.Prl[0], c.Prl[1], c.Qrl[0], c.Qrl[1], dprl[0], dprl[1], dqrl[0], dqrl[1], zw, dzw);
        z = zw.x;
        w = zw.y;
        dz = dzw.x;
        dw = dzw.y;
    }
    if (!Shadows(z, w, dz, dw, g.Tol)) s11 = 0;
    return {s11, xyzz, true};
}

// A triangle's sides as forward halfedges with their ends, and whether each side is the forward one.
struct BoolFaceEdge {
    uint Edge, Start, End;
    bool Forward;
};
inline void LoadFaceEdges(BoolGeometry g, uint t, thread BoolFaceEdge *edges) {
    for (uint i = 0u; i < 3u; ++i) {
        const uint h = 3u * t + i, start = g.Start(h), end = g.End(h);
        edges[i] = start < end ? BoolFaceEdge{h, start, end, true} : BoolFaceEdge{g.Pair(h), end, start, false};
    }
}

// Whether a vertex lies below a triangle along z in the xy projection, with the triangle's z above it.
struct BoolPierce {
    int S;
    float Z;
    bool Valid;
};
// The vertex is P's when `forward`, and the triangle's otherwise.
inline BoolPierce Kernel02(BoolGeometry g, bool forward, uint a0, thread const BoolFaceEdge *edges) {
    int s02 = 0;
    uint k = 0u;
    float3 yzz[2];
    uint edge_rl[2];
    bool shadows = false;
    for (uint i = 0u; i < 3u; ++i) {
        const BoolShadow shadow = Shadow01(g, forward, a0, edges[i].Start, edges[i].End);
        if (!shadow.Overlaps) continue;
        s02 += shadow.S * (forward == edges[i].Forward ? -1 : 1);
        if (k < 2u && (k == 0u || (shadow.S != 0) != shadows)) {
            shadows = shadow.S != 0;
            edge_rl[k] = i;
            yzz[k++] = float3(shadow.YZ.x, shadow.YZ.y, shadow.YZ.y);
        }
    }
    if (s02 == 0 || k < 2u) return {0, 0.f, false};
    const float3 a = g.Position(a0);
    const float z02 = BoolInterpolate(yzz[0], yzz[1], a.y).y;
    float z = z02, dz = 0.f, daz = 0.f;
    if (abs(a.z - z02) <= g.Tol) {
        const float3 da = g.Move(a0);
        float3 zrl[2], dzrl[2];
        for (uint j = 0u; j < 2u; ++j) {
            const BoolFaceEdge e = edges[edge_rl[j]];
            float2 yz, dyz;
            BoolLineAt(g.Position(e.Start), g.Position(e.End), g.Move(e.Start), g.Move(e.End), a.x, da.x, yz, dyz);
            zrl[j] = float3(yz.x, yz.y, yz.y);
            dzrl[j] = float3(dyz.x, dyz.y, dyz.y);
        }
        float2 yz, dyz;
        BoolLineAt(zrl[0], zrl[1], dzrl[0], dzrl[1], a.y, da.y, yz, dyz);
        z = yz.y;
        dz = dyz.y;
        daz = da.z;
    }
    if (forward ? !Shadows(a.z, z, daz, dz, g.Tol) : !Shadows(z, a.z, dz, daz, g.Tol)) s02 = 0;
    return {s02, z02, true};
}

// Whether an edge crosses a triangle, signed by its direction through it, and the crossing point.
struct BoolHit {
    int X;
    float3 V;
    bool Valid;
};
// The edge is P's when `forward`, and the triangle's otherwise.
inline BoolHit Kernel12(BoolGeometry g, bool forward, uint a1, uint b2) {
    int x12 = 0;
    uint k = 0u;
    float3 l0[2], l1[2];
    bool shadows = false;
    const uint ends[2] = {g.Start(a1), g.End(a1)};
    BoolFaceEdge edges[3];
    LoadFaceEdges(g, b2, edges);
    for (uint i = 0u; i < 2u; ++i) {
        const BoolPierce pierce = Kernel02(g, forward, ends[i], edges);
        if (!pierce.Valid) continue;
        x12 += pierce.S * ((i == 0u) == forward ? 1 : -1);
        if (k < 2u && (k == 0u || (pierce.S != 0) != shadows)) {
            shadows = pierce.S != 0;
            const float3 v = g.Position(ends[i]);
            l0[k] = float3(v.x, v.z, v.y);
            l1[k] = float3(v.x, pierce.Z, v.y);
            ++k;
        }
    }
    for (uint i = 0u; i < 3u; ++i) {
        const BoolCross cross = forward ? Kernel11(g, ends[0], ends[1], edges[i].Start, edges[i].End) : Kernel11(g, edges[i].Start, edges[i].End, ends[0], ends[1]);
        if (!cross.Valid) continue;
        x12 -= cross.S * (edges[i].Forward ? 1 : -1);
        if (k < 2u && (k == 0u || (cross.S != 0) != shadows)) {
            shadows = cross.S != 0;
            l0[k] = float3(cross.XYZZ.x, forward ? cross.XYZZ.z : cross.XYZZ.w, cross.XYZZ.y);
            l1[k] = float3(cross.XYZZ.x, forward ? cross.XYZZ.w : cross.XYZZ.z, cross.XYZZ.y);
            ++k;
        }
    }
    if (x12 == 0 || k < 2u) return {0, float3(0.f), false};
    const float4 xzyy = BoolIntersect(l0[0], l0[1], l1[0], l1[1], g.Tol);
    return {x12, float3(xzyy.x, xzyy.z, xzyy.y), true};
}

// Transforms vertex positions and normals into P's space and expands each record's bounds.
kernel void BooleanVertices(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (v >= BoolVertexCount(job)) return;
    const uint side = v < job.Vertices[0] ? 0u : 1u;
    const uint handle = ElementWorkDomain{b, job.VertexWork[side]}.Handle(side ? v - job.Vertices[0] : v);
    const ConnectivityView view{b, job.Connectivity};
    const uint address = view.VertexAddress(handle);
    const float3x3 m = job.Rotation.Unpack();
    const float3x3 normals = NormalTransform(m);
    float3 p = float3(BindlessBuffer(Vertex, b.VertexBuffer, job.VertexSlot)[address].Position);
    float3 n = float3(BindlessBuffer(packed_float3, b.Buffer, job.VertexNormalSlot)[address]);
    if (side) {
        p = m * p + float3(job.Translation);
        n = normals * n;
    }
    n = dot(n, n) > 0.f ? normalize(n) : float3(0.f);
    // The displacement that moves every face plane at the vertex by one unit: the normal of one plane, the offset of two, or the apex of three.
    // A vertex on more planes, or on no face, moves along its normal.
    float3 planes[3];
    int count = 0;
    for (const uint2 item : view.Fan(handle)) {
        if (item.y == InvalidOffset || count < 0) continue;
        float3 plane = float3(BindlessBuffer(packed_float3, b.Buffer, job.FaceNormalSlot)[view.FaceAddress(item.y)]);
        if (side) plane = normals * plane;
        if (!(dot(plane, plane) > 0.f)) continue;
        plane = normalize(plane);
        bool seen = false;
        for (int k = 0; k < count; ++k) seen = seen || dot(planes[k], plane) > 1.f - BoolPlaneTolerance;
        if (seen) continue;
        if (count == 3) {
            count = -1;
            continue;
        }
        planes[count++] = plane;
    }
    float3 displacement = n;
    if (count == 1) displacement = planes[0];
    else if (count == 2) displacement = (planes[0] + planes[1]) / (1.f + dot(planes[0], planes[1]));
    else if (count == 3) {
        const float det = dot(planes[0], cross(planes[1], planes[2]));
        if (abs(det) > BoolPlaneTolerance) displacement = (cross(planes[1], planes[2]) + cross(planes[2], planes[0]) + cross(planes[0], planes[1])) / det;
    }
    if (dot(displacement, displacement) > 4.f) displacement = 2.f * normalize(displacement);
    ctx.Run<packed_float3>(job.Positions)[v] = packed_float3(p);
    ctx.Run<packed_float3>(job.Displacements)[v] = packed_float3(displacement);
    device atomic_uint *bounds = ctx.Run<atomic_uint>(job.Bounds) + 6u * side;
    for (uint axis = 0u; axis < 3u; ++axis) {
        atomic_fetch_min_explicit(&bounds[axis], OrderedBits(p[axis]), memory_order_relaxed);
        atomic_fetch_max_explicit(&bounds[3u + axis], OrderedBits(p[axis]), memory_order_relaxed);
    }
}

kernel void BooleanFaceCounts(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint faces = BoolFaceCount(job);
    if (f > faces) return;
    device uint *first = ctx.Run<uint>(job.FaceTriangleFirst);
    if (f == faces) {
        first[f] = 0u;
        return;
    }
    const uint side = f < job.Faces[0] ? 0u : 1u;
    const uint handle = ElementWorkDomain{b, job.FaceWork[side]}.Handle(side ? f - job.Faces[0] : f);
    const uint2 range = ConnectivityView{b, job.Connectivity}.FaceHalfedges(handle);
    first[f] = range.y - range.x - 2u;
}

kernel void BooleanTriangles(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (f >= BoolFaceCount(job)) return;
    const uint side = f < job.Faces[0] ? 0u : 1u;
    const uint handle = ElementWorkDomain{b, job.FaceWork[side]}.Handle(side ? f - job.Faces[0] : f);
    const ElementWorkDomain vertices{b, job.VertexWork[side]};
    const uint vertex_base = side ? job.Vertices[0] : 0u;
    const ConnectivityView connectivity{b, job.Connectivity};
    const uint2 range = connectivity.FaceHalfedges(handle);
    const uint first = BindlessBuffer(uint, b.ObjectIdBuffer, job.FaceTriangleStartSlot)[connectivity.FaceAddress(handle)];
    const auto triangles = ReadElements<packed_uint3>(b, job.TriangleSlot, job.Connectivity.Tables.Triangles);
    device const packed_float3 *positions = ctx.Run<packed_float3>(job.Positions);
    const uint base = ctx.Run<uint>(job.FaceTriangleFirst)[f];
    for (uint k = 0u; k + 2u < range.y - range.x; ++k) {
        const uint t = base + k;
        const uint3 corners = uint3(triangles[first + k]);
        uint3 combined;
        for (uint i = 0u; i < 3u; ++i) combined[i] = vertex_base + vertices.Index(connectivity.VertexOf(corners[i]));
        const float3 p0 = float3(positions[combined.x]), p1 = float3(positions[combined.y]), p2 = float3(positions[combined.z]);
        const float3 normal = cross(p1 - p0, p2 - p0);
        const float length2 = dot(normal, normal);
        ctx.Run<packed_uint3>(job.TriangleVertices)[t] = packed_uint3(combined);
        ctx.Run<packed_uint3>(job.TriangleCorners)[t] = packed_uint3(corners);
        ctx.Run<uint>(job.TriangleFaces)[t] = f;
        ctx.Run<packed_float3>(job.TriangleNormals)[t] = packed_float3(length2 > 0.f ? normal * rsqrt(length2) : float3(0.f));
    }
}

inline uint BoolFindSide(BooleanContext ctx, BooleanJob job, uint f, uint from, uint to) {
    const uint first = ctx.Run<uint>(job.FaceTriangleFirst)[f], end = ctx.Run<uint>(job.FaceTriangleFirst)[f + 1u];
    device const packed_uint3 *corners = ctx.Run<packed_uint3>(job.TriangleCorners);
    for (uint t = first; t < end; ++t) {
        const uint3 c = uint3(corners[t]);
        for (uint i = 0u; i < 3u; ++i)
            if (c[i] == from && c[(i + 1u) % 3u] == to) return 3u * t + i;
    }
    return InvalidOffset;
}

// Pairs each triangle side with the side running back along it: across its face edge, or across its face's diagonal.
kernel void BooleanPairs(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (h >= 3u * BoolTriangleCount(job)) return;
    const uint t = h / 3u, i = h % 3u;
    const uint3 corners = uint3(ctx.Run<packed_uint3>(job.TriangleCorners)[t]);
    const uint from = corners[i], to = corners[(i + 1u) % 3u];
    const ConnectivityView connectivity{b, job.Connectivity};
    const uint2 range = connectivity.FaceHalfedges(connectivity.HalfedgeFace(from));
    const bool edge = (from + 1u < range.y ? from + 1u : range.x) == to;
    uint pair = InvalidOffset;
    if (edge) {
        const uint opposite = connectivity.Opposite(to);
        if (opposite != InvalidOffset) {
            const uint side = t < job.Triangles[0] ? 0u : 1u;
            const uint face = ElementWorkDomain{b, job.FaceWork[side]}.Index(connectivity.HalfedgeFace(opposite));
            if (face != InvalidOffset) pair = BoolFindSide(ctx, job, (side ? job.Faces[0] : 0u) + face, connectivity.Previous(opposite), opposite);
        }
    } else pair = BoolFindSide(ctx, job, ctx.Run<uint>(job.TriangleFaces)[t], to, from);
    ctx.Run<uint>(job.EdgeSources)[h] = edge ? to : InvalidOffset;
    ctx.Run<uint>(job.Pairs)[h] = pair;
    if (pair == InvalidOffset) ctx.Fail();
}

inline void BoolTriangleBox(BoolGeometry g, uint t, thread float3 &low, thread float3 &high) {
    const float3 a = g.Position(g.Start(3u * t)), c = g.Position(g.Start(3u * t + 1u)), d = g.Position(g.Start(3u * t + 2u));
    low = min(min(a, c), d);
    high = max(max(a, c), d);
}

kernel void BooleanMorton(
    uint t [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (t >= BoolTriangleCount(job)) return;
    const uint side = t < job.Triangles[0] ? 0u : 1u;
    const uint local = side ? t - job.Triangles[0] : t;
    const BoolGeometry g = Geometry(ctx, job);
    float3 box_low, box_high;
    BoolTriangleBox(g, t, box_low, box_high);
    const float3 center = (box_low + box_high) * .5f;
    device const uint *bounds = ctx.Run<uint>(job.Bounds) + 6u * side;
    uint code = 0u;
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float low = OrderedFloat(bounds[axis]), extent = OrderedFloat(bounds[3u + axis]) - low;
        const float unit = extent > 0.f ? (center[axis] - low) / extent : 0.f;
        code |= SpreadBits3(uint(clamp(1024.f * unit, 0.f, 1023.f))) << (2u - axis);
    }
    ctx.Run<uint>(job.Colliders[side].Keys)[local] = code;
    ctx.Run<uint>(job.Colliders[side].Order)[local] = local;
}

kernel void BooleanRefit(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint side = group.z, leaf = group.x * 256u + lane;
    const ColliderRuns runs = job.Colliders[side];
    if (leaf >= runs.Leaves) return;
    const BoolGeometry g = Geometry(ctx, job);
    const uint t = ctx.Run<uint>(runs.Order)[leaf] + (side ? job.Triangles[0] : 0u);
    float3 box_low, box_high;
    BoolTriangleBox(g, t, box_low, box_high);
    RefitClimb(ctx, runs, leaf, box_low, box_high);
}

template<typename Visit>
inline uint BoolEdgeHits(BooleanContext ctx, BooleanJob job, BoolGeometry g, uint h, Visit visit) {
    if (!g.Forward(h)) return 0u;
    const bool p_edge = h < 3u * job.Triangles[0];
    const float3 a = g.Position(g.Start(h)), c = g.Position(g.End(h));
    const float3 low = min(a, c), high = max(a, c);
    uint count = 0u;
    BoolTraverse(TreeOf(ctx, job.Colliders[p_edge ? 1u : 0u], p_edge ? job.Triangles[0] : 0u), [&](device const float *box) { return BoxOverlaps(box, low, high); }, [&](uint t) {
        const BoolHit hit = Kernel12(g, p_edge, h, t);
        if (!hit.Valid) return;
        visit(count, t, hit);
        ++count;
    });
    return count;
}

kernel void BooleanIntersectCount(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint halfedges = 3u * BoolTriangleCount(job);
    if (h > halfedges) return;
    device uint *first = ctx.Run<uint>(job.IntersectionFirst);
    first[h] = h == halfedges ? 0u : BoolEdgeHits(ctx, job, Geometry(ctx, job), h, [](uint, uint, BoolHit) {});
}

kernel void BooleanIntersectEmit(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (h >= 3u * BoolTriangleCount(job)) return;
    const uint first = ctx.Run<uint>(job.IntersectionFirst)[h];
    device uint *records = ctx.Run<uint>(job.IntersectionRecords);
    BoolEdgeHits(ctx, job, Geometry(ctx, job), h, [&](uint i, uint t, BoolHit hit) {
        device uint *record = records + 6u * (first + i);
        record[0] = h;
        record[1] = t;
        record[2] = as_type<uint>(hit.X);
        record[3] = as_type<uint>(hit.V.x);
        record[4] = as_type<uint>(hit.V.y);
        record[5] = as_type<uint>(hit.V.z);
    });
}

// Whether a halfedge crosses the other record, which disconnects its edge component.
inline bool BoolBroken(BooleanContext ctx, BooleanJob job, uint h) {
    device const uint *first = ctx.Run<uint>(job.IntersectionFirst);
    return first[h + 1u] != first[h];
}

kernel void BooleanLabelInit(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (v < BoolVertexCount(job)) ctx.Run<uint>(job.Labels)[v] = v;
}

kernel void BooleanLink(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint h = group_id * 256u + lane;
    if (h >= 3u * BoolTriangleCount(job)) return;
    const BoolGeometry g = Geometry(ctx, job);
    if (!g.Forward(h) || BoolBroken(ctx, job, h)) return;
    device uint *labels = ctx.Run<uint>(job.Labels);
    device atomic_uint *atomic_labels = ctx.Run<atomic_uint>(job.Labels);
    const uint a = g.Start(h), c = g.End(h), la = labels[a], lc = labels[c], m = min(la, lc);
    const uint targets[4] = {a, c, la, lc};
    bool changed = false;
    for (uint i = 0u; i < 4u; ++i) changed = atomic_fetch_min_explicit(&atomic_labels[targets[i]], m, memory_order_relaxed) > m || changed;
    if (changed) ctx.Run<uint>(job.LabelArguments)[6] = 1u;
}

kernel void BooleanJump(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint v = group_id * 256u + lane;
    if (v >= BoolVertexCount(job)) return;
    device uint *labels = ctx.Run<uint>(job.Labels);
    const uint label = labels[v], root = labels[label];
    if (root == label) return;
    labels[v] = root;
    ctx.Run<uint>(job.LabelArguments)[6] = 1u;
}

// Zeroes the label rounds' indirect dispatch arguments once a round changes nothing.
kernel void BooleanConverge(
    uint lane [[thread_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (lane != 0u) return;
    const BooleanContext ctx{b, pc};
    device uint *arguments = ctx.Run<uint>(ctx.Job().LabelArguments);
    if (arguments[6] == 0u)
        for (uint w = 0u; w < 6u; ++w) arguments[w] = 0u;
    arguments[6] = 0u;
}

// Each component root's winding number in the other record: the signed crossings of the other's triangles above it along z.
kernel void BooleanWinding(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (v >= BoolVertexCount(job) || ctx.Run<uint>(job.Labels)[v] != v) return;
    const bool forward = v < job.Vertices[0];
    const BoolGeometry g = Geometry(ctx, job);
    const float2 point = g.Position(v).xy;
    int winding = 0;
    BoolTraverse(TreeOf(ctx, job.Colliders[forward ? 1u : 0u], forward ? job.Triangles[0] : 0u), [&](device const float *box) { return BoxContainsXY(box, point); }, [&](uint t) {
        BoolFaceEdge edges[3];
        LoadFaceEdges(g, t, edges);
        const BoolPierce pierce = Kernel02(g, forward, v, edges);
        if (pierce.Valid) winding += pierce.S;
    });
    ctx.Run<int>(job.Windings)[v] = forward ? winding : -winding;
}

kernel void BooleanInclusion(
    uint e [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint vertices = BoolVertexCount(job);
    if (e >= vertices + job.Intersections) return;
    const int c1 = job.Operation == BooleanOperation::Intersection ? 0 : 1, c2 = job.Operation == BooleanOperation::Union ? 1 : 0;
    const int c3 = job.Operation == BooleanOperation::Intersection ? 1 : -1;
    int inclusion;
    if (e < vertices) {
        const int winding = ctx.Run<int>(job.Windings)[ctx.Run<uint>(job.Labels)[e]];
        inclusion = (e < job.Vertices[0] ? c1 : c2) + c3 * winding;
    } else inclusion = c3 * BoolRecordCrossing(ctx, job, e - vertices);
    ctx.Run<int>(job.Inclusion)[e] = inclusion;
}

// Visits each member of the new edge group of P triangle p and Q triangle q: an intersection of an edge of one with the other, and its side.
// Side zero is the member's own halfedge's triangle, and side one its pair's.
template<typename Visit>
inline void BoolGroupMembers(BooleanContext ctx, BooleanJob job, BoolGeometry g, uint p, uint q, Visit visit) {
    device const uint *first = ctx.Run<uint>(job.IntersectionFirst);
    for (uint pass = 0u; pass < 2u; ++pass) {
        const uint face = pass ? q : p, other = pass ? p : q;
        for (uint i = 0u; i < 3u; ++i) {
            const uint h = 3u * face + i, forward = g.Forward(h) ? h : g.Pair(h);
            for (uint k = first[forward]; k < first[forward + 1u]; ++k)
                if (BoolRecordTriangle(ctx, job, k) == other) visit(k, forward / 3u == face ? 0u : 1u);
        }
    }
}
inline uint2 BoolGroupOf(BooleanContext ctx, BooleanJob job, BoolGeometry g, uint member) {
    const uint k = member / 2u, h = BoolRecordHalfedge(ctx, job, k);
    const uint face = member % 2u ? g.Pair(h) / 3u : h / 3u, t = BoolRecordTriangle(ctx, job, k);
    return k < job.FirstIntersections ? uint2(face, t) : uint2(t, face);
}

// A group's owner is its lowest intersection side.
kernel void BooleanCounts(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint vertices = BoolVertexCount(job), triangles = BoolTriangleCount(job), sides = 2u * job.Intersections, entries = vertices + job.Intersections;
    const uint total = 4u * triangles + sides + entries;
    if (i > total) return;
    device uint *counts = ctx.Run<uint>(job.Counts);
    const BoolGeometry g = Geometry(ctx, job);
    if (i < triangles) {
        counts[i] = BoolCount(ctx, job, g.Start(3u * i)) + BoolCount(ctx, job, g.Start(3u * i + 1u)) + BoolCount(ctx, job, g.Start(3u * i + 2u));
    } else if (i < 4u * triangles) {
        const uint h = i - triangles;
        uint count = 0u;
        if (g.Forward(h) && BoolBroken(ctx, job, h)) {
            device const uint *first = ctx.Run<uint>(job.IntersectionFirst);
            for (uint k = first[h]; k < first[h + 1u]; ++k) count += BoolCount(ctx, job, vertices + k);
            count += BoolCount(ctx, job, g.Start(h)) + BoolCount(ctx, job, g.End(h));
        }
        counts[i] = count;
    } else if (i < 4u * triangles + sides) {
        const uint member = i - 4u * triangles;
        const uint2 group = BoolGroupOf(ctx, job, g, member);
        uint owner = InvalidOffset, count = 0u;
        BoolGroupMembers(ctx, job, g, group.x, group.y, [&](uint k, uint side) {
            owner = min(owner, 2u * k + side);
            count += BoolCount(ctx, job, vertices + k);
        });
        counts[i] = owner == member ? count : 0u;
    } else if (i < total) counts[i] = BoolCount(ctx, job, i - 4u * triangles - sides);
    else counts[i] = 0u;
}

kernel void BooleanIntersectionSides(
    uint k [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (k >= job.Intersections) return;
    const BoolGeometry g = Geometry(ctx, job);
    const uint count = BoolCount(ctx, job, BoolVertexCount(job) + k), h = BoolRecordHalfedge(ctx, job, k);
    device atomic_uint *counts = ctx.Run<atomic_uint>(job.Counts);
    const uint triangles[3] = {BoolRecordTriangle(ctx, job, k), h / 3u, g.Pair(h) / 3u};
    for (uint i = 0u; i < 3u; ++i) atomic_fetch_add_explicit(&counts[triangles[i]], count, memory_order_relaxed);
}

kernel void BooleanOutputs(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (i < BoolVertexCount(job) + job.Intersections) {
        const uint first = BoolOutputFirst(ctx, job, i), count = BoolCount(ctx, job, i);
        for (uint j = 0u; j < count; ++j) ctx.Run<uint>(job.OutputEntries)[first + j] = i;
    }
    if (i < BoolTriangleCount(job)) ctx.Run<uint>(job.Cursors)[i] = BoolPolygonFirst(ctx, job, i);
}

// Appends a polygon halfedge to triangle t, and fails the chain when t's polygon is full.
inline void BoolAppend(BooleanContext ctx, BooleanJob job, uint t, uint start, uint end, uint source) {
    const uint slot = atomic_fetch_add_explicit(ctx.Run<atomic_uint>(job.Cursors) + t, 1u, memory_order_relaxed);
    if (slot >= BoolPolygonFirst(ctx, job, t + 1u)) {
        ctx.Fail();
        return;
    }
    device uint *polygon = ctx.Run<uint>(job.Polygons) + 3u * slot;
    polygon[0] = start;
    polygon[1] = end;
    polygon[2] = source;
}
inline void BoolAppendEdge(BooleanContext ctx, BooleanJob job, uint h, uint pair, uint from, uint to) {
    device const uint *sources = ctx.Run<uint>(job.EdgeSources);
    BoolAppend(ctx, job, h / 3u, from, to, sources[h]);
    BoolAppend(ctx, job, pair / 3u, to, from, sources[pair]);
}

// Each unbroken forward edge's copies join both its triangles' polygons, reversed where its start's inclusion is negative.
kernel void BooleanWholeEdges(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (h >= 3u * BoolTriangleCount(job)) return;
    const BoolGeometry g = Geometry(ctx, job);
    if (!g.Forward(h) || BoolBroken(ctx, job, h)) return;
    const uint start = g.Start(h), end = g.End(h), pair = g.Pair(h);
    const int inclusion = ctx.Run<int>(job.Inclusion)[start];
    for (uint j = 0u; j < uint(abs(inclusion)); ++j) {
        const uint a = BoolOutputFirst(ctx, job, start) + j, c = BoolOutputFirst(ctx, job, end) + j;
        BoolAppendEdge(ctx, job, h, pair, inclusion < 0 ? c : a, inclusion < 0 ? a : c);
    }
}

// An output vertex at a position along a partial or new edge, its intersection or none, and whether it starts an edge.
struct BoolEdgePos {
    float Position;
    uint Vertex, Collision, Start;
};
inline bool BoolEdgePosBefore(BoolEdgePos a, BoolEdgePos c) {
    if (a.Start != c.Start) return a.Start > c.Start;
    if (a.Position != c.Position) return a.Position < c.Position;
    if (a.Collision != c.Collision) return a.Collision < c.Collision;
    return a.Vertex < c.Vertex;
}
// Sorts the entries starts first, each half by position then intersection, and pairs the i-th start with the i-th end.
template<typename Emit>
inline void BoolPairUp(BooleanContext ctx, device BoolEdgePos *entries, uint count, Emit emit) {
    InsertionSort(entries, count, BoolEdgePosBefore);
    uint starts = 0u;
    while (starts < count && entries[starts].Start) ++starts;
    if (2u * starts != count) {
        ctx.Fail();
        return;
    }
    for (uint i = 0u; i < starts; ++i) emit(entries[i].Vertex, entries[starts + i].Vertex);
}

// Each broken forward edge keeps the stretches between its crossings and kept ends, ordered along it, in both its triangles' polygons.
kernel void BooleanPartialEdges(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (h >= 3u * BoolTriangleCount(job)) return;
    const BoolGeometry g = Geometry(ctx, job);
    if (!g.Forward(h) || !BoolBroken(ctx, job, h)) return;
    device BoolEdgePos *entries = ctx.Run<BoolEdgePos>(job.PartialEdges) + BoolPartialFirst(ctx, job, h);
    const uint start = g.Start(h), end = g.End(h), vertices = BoolVertexCount(job);
    const float3 direction = g.Position(end) - g.Position(start);
    device const uint *first = ctx.Run<uint>(job.IntersectionFirst);
    uint count = 0u;
    for (uint k = first[h]; k < first[h + 1u]; ++k) {
        const int inclusion = ctx.Run<int>(job.Inclusion)[vertices + k];
        const float position = dot(BoolRecordPosition(ctx, job, k), direction);
        for (uint j = 0u; j < uint(abs(inclusion)); ++j) entries[count++] = {position, BoolOutputFirst(ctx, job, vertices + k) + j, k, uint(inclusion < 0)};
    }
    const int start_inclusion = ctx.Run<int>(job.Inclusion)[start], end_inclusion = ctx.Run<int>(job.Inclusion)[end];
    for (uint j = 0u; j < uint(abs(start_inclusion)); ++j) entries[count++] = {dot(g.Position(start), direction), BoolOutputFirst(ctx, job, start) + j, BoolNoCollision, uint(start_inclusion > 0)};
    for (uint j = 0u; j < uint(abs(end_inclusion)); ++j) entries[count++] = {dot(g.Position(end), direction), BoolOutputFirst(ctx, job, end) + j, BoolNoCollision, uint(end_inclusion < 0)};
    const uint pair = g.Pair(h);
    BoolPairUp(ctx, entries, count, [&](uint from, uint to) { BoolAppendEdge(ctx, job, h, pair, from, to); });
}

// Each new edge group's owner pairs its crossings along the group's longest extent, with opposite segment directions in P and Q.
kernel void BooleanNewEdges(
    uint member [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (member >= 2u * job.Intersections) return;
    const BoolGeometry g = Geometry(ctx, job);
    const uint2 group = BoolGroupOf(ctx, job, g, member);
    uint owner = InvalidOffset;
    BoolGroupMembers(ctx, job, g, group.x, group.y, [&](uint k, uint side) { owner = min(owner, 2u * k + side); });
    if (owner != member) return;
    device BoolEdgePos *entries = ctx.Run<BoolEdgePos>(job.NewEdges) + BoolGroupFirst(ctx, job, member);
    const uint vertices = BoolVertexCount(job);
    float3 low = float3(INFINITY), high = float3(-INFINITY);
    uint count = 0u;
    BoolGroupMembers(ctx, job, g, group.x, group.y, [&](uint k, uint side) {
        const int inclusion = ctx.Run<int>(job.Inclusion)[vertices + k];
        // A P crossing starts its segment for nonnegative inclusion, and a Q crossing for negative inclusion.
        const bool direction = inclusion < 0, first_record = k < job.FirstIntersections;
        const bool start = side ? direction != !first_record : direction != first_record;
        const float3 p = BoolRecordPosition(ctx, job, k);
        low = min(low, p);
        high = max(high, p);
        for (uint j = 0u; j < uint(abs(inclusion)); ++j) entries[count++] = {0.f, BoolOutputFirst(ctx, job, vertices + k) + j, k, uint(start)};
    });
    const float3 size = high - low;
    const uint axis = size.x > size.y && size.x > size.z ? 0u : size.y > size.z ? 1u : 2u;
    for (uint i = 0u; i < count; ++i) entries[i].Position = BoolRecordPosition(ctx, job, entries[i].Collision)[axis];
    BoolPairUp(ctx, entries, count, [&](uint from, uint to) {
        BoolAppend(ctx, job, group.x, from, to, InvalidOffset);
        BoolAppend(ctx, job, group.y, to, from, InvalidOffset);
    });
}

// The axis-aligned projection that keeps a plane with this normal counterclockwise, as Manifold projects polygons.
inline float2 BoolProject(float3 normal, float3 p) {
    const float3 n = abs(normal);
    if (n.z > n.x && n.z > n.y) return float2(normal.z < 0.f ? -p.x : p.x, p.y);
    if (n.y > n.x) return float2(normal.y < 0.f ? -p.z : p.z, p.x);
    return float2(normal.x < 0.f ? -p.y : p.y, p.z);
}

// A triangle emits itself, and a quad the counterclockwise pair of triangles across its shorter diagonal.
// Any other polygon ear-clips with its holes joined.
kernel void BooleanFaces(
    uint t [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (t >= BoolTriangleCount(job)) return;
    device uint *triangle_counts = ctx.Run<uint>(job.PolygonTriangleCounts);
    triangle_counts[t] = 0u;
    const uint first = BoolPolygonFirst(ctx, job, t), n = BoolPolygonFirst(ctx, job, t + 1u) - first;
    if (!n) return;
    if (ctx.Run<uint>(job.Cursors)[t] != first + n) {
        ctx.Fail();
        return;
    }
    // Sorting by start, end and source orders the halfedges independently of the appending passes' arrival.
    device packed_uint3 *halfedges = ctx.Run<packed_uint3>(job.Polygons) + first;
    InsertionSort(halfedges, n, [](packed_uint3 a, packed_uint3 c) { return a.x != c.x ? a.x < c.x : a.y != c.y ? a.y < c.y : a.z < c.z; });
    const auto start_of = [&](uint i) { return uint3(halfedges[i]).x; };
    // Each halfedge's successor in its loop, each node's halfedge, and each loop's first node and then its state.
    device uint *successors = ctx.Run<uint>(job.LoopLinks) + 4u * first, *refs = successors + n, *loops = refs + n;
    // The refs mark consumed halfedges until the nodes overwrite them.
    for (uint i = 0u; i < n; ++i) {
        successors[i] = InvalidOffset;
        refs[i] = 0u;
    }
    uint loop_count = 0u;
    for (uint head = 0u; head < n; ++head) {
        if (successors[head] != InvalidOffset) continue;
        uint i = head;
        do {
            const uint end = uint3(halfedges[i]).y;
            uint lo = PartitionPoint(n, [&](uint k) { return start_of(k) < end; });
            while (lo < n && start_of(lo) == end && refs[lo]) ++lo;
            if (lo == n || start_of(lo) != end) {
                ctx.Fail();
                return;
            }
            refs[lo] = 1u;
            successors[i] = lo;
            i = lo;
        } while (i != head);
        loops[loop_count++] = head;
    }
    // Nodes list each loop's halfedges from its head, projected along the polygon's normal, with room for two copies per joined hole.
    // Q's triangles face inward in a difference, so their polygons project against their normals.
    const bool inverted = job.Operation == BooleanOperation::Difference && t >= job.Triangles[0];
    const float3 normal = float3(ctx.Run<packed_float3>(job.TriangleNormals)[t]) * (inverted ? -1.f : 1.f);
    const uint capacity = 2u * n;
    device uint *links = ctx.Run<uint>(job.LoopNodes) + 7u * first;
    device float *values = ctx.Run<float>(job.LoopValues) + 6u * first;
    uint node = 0u;
    for (uint l = 0u; l < loop_count; ++l) {
        const uint head = loops[l], loop_first = node;
        loops[l] = node;
        uint i = head;
        do {
            refs[node] = i;
            const float2 p = BoolProject(normal, BoolOutputPosition(ctx, job, start_of(i)));
            values[2u * node] = p.x;
            values[2u * node + 1u] = p.y;
            links[node] = node + 1u;
            links[capacity + node] = node - 1u;
            ++node;
            i = successors[i];
        } while (i != head);
        links[node - 1u] = loop_first;
        links[capacity + loop_first] = node - 1u;
    }
    // Each triangle corner refers to its node's halfedge, marked when the side to the next corner is that halfedge.
    device uint *triangles = ctx.Run<uint>(job.PolygonTriangles) + 6u * first;
    uint emitted = 0u;
    const auto emit = [&](uvec3 corners, uint index) {
        for (uint k = 0u; k < 3u; ++k) {
            const uint h = refs[corners[k]];
            triangles[3u * index + k] = h | (successors[h] == refs[corners[(k + 1u) % 3u]] ? BoolRealOut : 0u);
        }
        ++emitted;
    };
    const float tol = Geometry(ctx, job).Tol;
    if (loop_count == 1u && n == 3u) emit(uvec3{0u, 1u, 2u}, 0u);
    else if (loop_count == 1u && n == 4u) {
        const auto point = [&](uint i) { return float2(values[2u * i], values[2u * i + 1u]); };
        const auto ccw = [&](uint a, uint c, uint d) { return PolygonCCW(point(a), point(c), point(d), tol) >= 0; };
        const auto position = [&](uint i) { return BoolOutputPosition(ctx, job, start_of(refs[i])); };
        if (!(ccw(0u, 1u, 2u) && ccw(0u, 2u, 3u)) || (ccw(1u, 2u, 3u) && ccw(0u, 1u, 3u) && distance_squared(position(0u), position(2u)) > distance_squared(position(1u), position(3u)))) {
            emit(uvec3{1u, 2u, 3u}, 0u);
            emit(uvec3{0u, 1u, 3u}, 1u);
        } else {
            emit(uvec3{0u, 1u, 2u}, 0u);
            emit(uvec3{0u, 2u, 3u}, 1u);
        }
    } else {
        TriangulateLoops(n, capacity, loop_count, links, values, loops, tol, emit);
    }
    triangle_counts[t] = emitted;
    atomic_fetch_add_explicit(ctx.Run<atomic_uint>(job.TriangleTotal), emitted, memory_order_relaxed);
}
