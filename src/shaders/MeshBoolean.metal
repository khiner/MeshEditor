// The boolean's topological decisions compare input coordinates and values computed the same way at every use, so this file's arithmetic stays IEEE without contraction.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// Manifold's boolean as GPU passes: a radix tree per record, edge and triangle crossings by shadow predicates with symbolic perturbation,
// winding numbers flooded through the components the crossings leave, and each kept triangle's polygon assembled and retriangulated.
#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "RadixSort.metal"
#include "MeshBooleanShared.metal"

constant uint BoolRealOut = 0x80000000u; // A loop node whose edge to its next node is a polygon halfedge
constant uint BoolNoCollision = 0xffffffffu;

struct BooleanContext {
    device const BindlessSet &B;
    constant BooleanPushConstants &Pc;

    device uint *Storage() const { return BindlessBufferMutable(uint, B.Buffer, Pc.Job.Slot); }
    BooleanJob Job() const { return *reinterpret_cast<device const BooleanJob *>(Storage() + Pc.Job.Offset); }
    template<typename T> device T *Run(uint offset) const { return reinterpret_cast<device T *>(Storage() + offset); }
    // Marks the chain's error word, which fails its submit.
    void Fail() const { atomic_store_explicit(reinterpret_cast<device atomic_uint *>(Storage()), 1u, memory_order_relaxed); }
};

// Symbolic perturbation moves each record's vertices along their normals: a union expands both,
// a difference contracts P and expands Q, and an intersection contracts both, so solids that only touch stay apart.
struct BoolPerturb {
    bool P, Q;
};
inline BoolPerturb BoolExpand(BooleanJob job) { return {job.Operation == BooleanOperation::Union, job.Operation != BooleanOperation::Intersection}; }


// Symbolic perturbation: p < q, and the direction decides an exact tie.
inline bool Shadows(float p, float q, float dir) { return p == q ? dir < 0.f : p < q; }
inline float WithSign(bool positive, float v) { return positive ? v : -v; }

// The (y, z) of segment l-r at x, measured from its end nearer x.
inline float2 BoolInterpolate(float3 l, float3 r, float x) {
    const float dxl = x - l.x, dxr = x - r.x;
    const bool use_l = abs(dxl) < abs(dxr);
    const float3 d = r - l;
    const float lambda = (use_l ? dxl : dxr) / d.x;
    if (!isfinite(lambda) || !isfinite(d.y) || !isfinite(d.z)) return l.yz;
    return float2(lambda * d.y + (use_l ? l.y : r.y), lambda * d.z + (use_l ? l.z : r.z));
}

// The crossing of segments a and b over one x interval, whose y gaps bracket zero, as (x, y, a.z, b.z).
inline float4 BoolIntersect(float3 al, float3 ar, float3 bl, float3 br) {
    const float dyl = bl.y - al.y, dyr = br.y - ar.y;
    const bool use_l = abs(dyl) < abs(dyr);
    const float dx = ar.x - al.x;
    float lambda = (use_l ? dyl : dyr) / (dyl - dyr);
    if (!isfinite(lambda)) lambda = 0.f;
    const float ady = ar.y - al.y, bdy = br.y - bl.y;
    const bool use_a = abs(ady) < abs(bdy);
    return float4(
        lambda * dx + (use_l ? al.x : ar.x),
        lambda * (use_a ? ady : bdy) + (use_l ? (use_a ? al.y : bl.y) : (use_a ? ar.y : br.y)),
        lambda * (ar.z - al.z) + (use_l ? al.z : ar.z),
        lambda * (br.z - bl.z) + (use_l ? bl.z : br.z)
    );
}

// Whether a vertex shadows an edge along x then y, with the edge's (y, z) at the vertex's x when their x ranges overlap.
struct BoolShadow {
    int S;
    float2 YZ;
    bool Overlaps;
};
// The vertex is P's when `forward`, and the edge's otherwise.
inline BoolShadow Shadow01(BoolGeometry g, BoolPerturb expand, bool forward, uint a0, uint b1, uint b1s, uint b1e) {
    const float a0x = g.Position(a0).x, b1sx = g.Position(b1s).x, b1ex = g.Position(b1e).x;
    const float a0xp = g.Normal(a0).x, b1sxp = g.Normal(b1s).x, b1exp = g.Normal(b1e).x;
    BoolShadow result{0, float2(0.f), false};
    result.S = forward ? int(Shadows(a0x, b1ex, WithSign(expand.P, a0xp) - WithSign(expand.Q, b1exp))) - int(Shadows(a0x, b1sx, WithSign(expand.P, a0xp) - WithSign(expand.Q, b1sxp))) :
                         int(Shadows(b1sx, a0x, WithSign(expand.P, b1sxp) - WithSign(expand.Q, a0xp))) - int(Shadows(b1ex, a0x, WithSign(expand.P, b1exp) - WithSign(expand.Q, a0xp)));
    if (result.S == 0) return result;
    result.Overlaps = true;
    result.YZ = BoolInterpolate(g.Position(b1s), g.Position(b1e), g.Position(a0).x);
    const float dir = g.FaceNormal(b1 / 3u).y + g.FaceNormal(g.Pair(b1) / 3u).y;
    if (forward) {
        if (!Shadows(g.Position(a0).y, result.YZ.x, -WithSign(expand.Q, dir))) result.S = 0;
    } else if (!Shadows(result.YZ.x, g.Position(a0).y, WithSign(expand.P, dir))) result.S = 0;
    return result;
}

// Whether a P edge and a Q edge cross in the xy projection, with the crossing's (x, y) and each edge's z there.
struct BoolCross {
    int S;
    float4 XYZZ;
    bool Valid;
};
inline BoolCross Kernel11(BoolGeometry g, BoolPerturb expand, uint p1, uint p1s, uint p1e, uint q1, uint q1s, uint q1e) {
    uint k = 0u;
    float3 prl[2], qrl[2];
    bool shadows = false;
    int s11 = 0;
    const uint p0[2] = {p1s, p1e};
    for (uint i = 0u; i < 2u; ++i) {
        const BoolShadow shadow = Shadow01(g, expand, true, p0[i], q1, q1s, q1e);
        if (!shadow.Overlaps) continue;
        s11 += shadow.S * (i == 0u ? -1 : 1);
        if (k < 2u && (k == 0u || (shadow.S != 0) != shadows)) {
            shadows = shadow.S != 0;
            prl[k] = g.Position(p0[i]);
            qrl[k] = float3(prl[k].x, shadow.YZ.x, shadow.YZ.y);
            ++k;
        }
    }
    const uint q0[2] = {q1s, q1e};
    for (uint i = 0u; i < 2u; ++i) {
        const BoolShadow shadow = Shadow01(g, expand, false, q0[i], p1, p1s, p1e);
        if (!shadow.Overlaps) continue;
        s11 += shadow.S * (i == 0u ? -1 : 1);
        if (k < 2u && (k == 0u || (shadow.S != 0) != shadows)) {
            shadows = shadow.S != 0;
            qrl[k] = g.Position(q0[i]);
            prl[k] = float3(qrl[k].x, shadow.YZ.x, shadow.YZ.y);
            ++k;
        }
    }
    if (s11 == 0 || k < 2u) return {0, float4(0.f), false};
    const float4 xyzz = BoolIntersect(prl[0], prl[1], qrl[0], qrl[1]);
    const float dir_p = g.FaceNormal(p1 / 3u).z + g.FaceNormal(g.Pair(p1) / 3u).z;
    const float dir_q = g.FaceNormal(q1 / 3u).z + g.FaceNormal(g.Pair(q1) / 3u).z;
    if (!Shadows(xyzz.z, xyzz.w, WithSign(expand.P, dir_p) - WithSign(expand.Q, dir_q))) s11 = 0;
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
inline BoolPierce Kernel02(BoolGeometry g, BoolPerturb expand, bool forward, uint a0, uint b2, thread const BoolFaceEdge *edges) {
    int s02 = 0;
    uint k = 0u;
    float3 yzz[2];
    bool shadows = false;
    for (uint i = 0u; i < 3u; ++i) {
        const BoolShadow shadow = Shadow01(g, expand, forward, a0, edges[i].Edge, edges[i].Start, edges[i].End);
        if (!shadow.Overlaps) continue;
        s02 += shadow.S * (forward == edges[i].Forward ? -1 : 1);
        if (k < 2u && (k == 0u || (shadow.S != 0) != shadows)) {
            shadows = shadow.S != 0;
            yzz[k++] = float3(shadow.YZ.x, shadow.YZ.y, shadow.YZ.y);
        }
    }
    if (s02 == 0 || k < 2u) return {0, 0.f, false};
    const float3 a = g.Position(a0);
    const float z = BoolInterpolate(yzz[0], yzz[1], a.y).y;
    if (forward) {
        if (!Shadows(a.z, z, -WithSign(expand.Q, g.FaceNormal(b2).z))) s02 = 0;
    } else if (!Shadows(z, a.z, WithSign(expand.P, g.FaceNormal(b2).z))) s02 = 0;
    return {s02, z, true};
}

// Whether an edge crosses a triangle, signed by its direction through it, and the crossing point.
struct BoolHit {
    int X;
    float3 V;
    bool Valid;
};
// The edge is P's when `forward`, and the triangle's otherwise.
inline BoolHit Kernel12(BoolGeometry g, BoolPerturb expand, bool forward, uint a1, uint b2) {
    int x12 = 0;
    uint k = 0u;
    float3 l0[2], l1[2];
    bool shadows = false;
    const uint ends[2] = {g.Start(a1), g.End(a1)};
    BoolFaceEdge edges[3];
    LoadFaceEdges(g, b2, edges);
    for (uint i = 0u; i < 2u; ++i) {
        const BoolPierce pierce = Kernel02(g, expand, forward, ends[i], b2, edges);
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
        const BoolCross cross = forward ? Kernel11(g, expand, a1, ends[0], ends[1], edges[i].Edge, edges[i].Start, edges[i].End) :
                                          Kernel11(g, expand, edges[i].Edge, edges[i].Start, edges[i].End, a1, ends[0], ends[1]);
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
    const float4 xzyy = BoolIntersect(l0[0], l0[1], l1[0], l1[1]);
    return {x12, float3(xzyy.x, xzyy.z, xzyy.y), true};
}

// A record's radix tree over its triangle boxes: leaves at even nodes, internal nodes at odd ones, and the root at node 1.
struct BoolTree {
    device const float *Boxes; // Six floats per node: minimum then maximum
    device const packed_uint2 *Children;
    device const uint *Order; // Leaf -> the record's triangle
    uint Leaves;
    uint First; // The record's first combined triangle
};
inline BoolTree TreeOf(BooleanContext ctx, BooleanJob job, uint side) {
    return {ctx.Run<float>(job.NodeBoxes[side]), ctx.Run<packed_uint2>(job.Children[side]), ctx.Run<uint>(job.Order[side]), job.Triangles[side], side ? job.Triangles[0] : 0u};
}
// Visits the combined triangles whose leaf boxes `overlaps` accepts, depth first.
template<typename Overlaps, typename Visit>
inline void BoolTraverse(BoolTree tree, Overlaps overlaps, Visit visit) {
    if (tree.Leaves < 2u) {
        if (tree.Leaves == 1u && overlaps(tree.Boxes)) visit(tree.First + tree.Order[0]);
        return;
    }
    uint stack[64];
    int top = -1;
    uint node = 1u;
    while (true) {
        const uint2 children = uint2(tree.Children[(node - 1u) / 2u]);
        bool descend[2];
        for (uint c = 0u; c < 2u; ++c) {
            const uint child = children[c];
            const bool hit = overlaps(tree.Boxes + 6u * child);
            if (hit && (child & 1u) == 0u) visit(tree.First + tree.Order[child / 2u]);
            descend[c] = hit && (child & 1u) != 0u;
        }
        if (!descend[0] && !descend[1]) {
            if (top < 0) break;
            node = stack[top--];
        } else {
            node = descend[0] ? children[0] : children[1];
            if (descend[0] && descend[1]) stack[++top] = children[1];
        }
    }
}
inline bool BoxOverlaps(device const float *box, float3 low, float3 high) {
    return box[0] <= high.x && box[1] <= high.y && box[2] <= high.z && low.x <= box[3] && low.y <= box[4] && low.z <= box[5];
}
inline bool BoxContainsXY(device const float *box, float2 p) { return p.x >= box[0] && p.x <= box[3] && p.y >= box[1] && p.y <= box[4]; }

// Float bits whose unsigned order is the floats' order.
inline uint OrderedBits(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float OrderedFloat(uint bits) { return as_type<float>((bits & 0x80000000u) ? bits & 0x7fffffffu : ~bits); }
inline uint SpreadBits3(uint v) {
    v = 0xFF0000FFu & (v * 0x00010001u);
    v = 0x0F00F00Fu & (v * 0x00000101u);
    v = 0xC30C30C3u & (v * 0x00000011u);
    v = 0x49249249u & (v * 0x00000005u);
    return v;
}

// Takes each vertex's position and normal into P's space, and grows its record's bounds.
kernel void BooleanVertices(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (v >= BoolVertexCount(job)) return;
    const uint side = v < job.Vertices[0] ? 0u : 1u;
    const uint handle = ElementWorkDomain{b, job.VertexWork[side]}.Handle(side ? v - job.Vertices[0] : v);
    const uint address = ConnectivityView{b, job.Connectivity, 0u}.VertexAddress(handle);
    float3 p = float3(BindlessBuffer(Vertex, b.VertexBuffer, job.VertexSlot)[address].Position);
    float3 n = float3(BindlessBuffer(packed_float3, b.Buffer, job.VertexNormalSlot)[address]);
    if (side) {
        const float3x3 m = job.Rotation.Unpack();
        p = m * p + float3(job.Translation);
        n = float3x3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])) * n;
    }
    const float length2 = dot(n, n);
    ctx.Run<packed_float3>(job.Positions)[v] = packed_float3(p);
    ctx.Run<packed_float3>(job.Normals)[v] = packed_float3(length2 > 0.f ? n * rsqrt(length2) : float3(0.f));
    device atomic_uint *bounds = ctx.Run<atomic_uint>(job.Bounds) + 6u * side;
    for (uint axis = 0u; axis < 3u; ++axis) {
        atomic_fetch_min_explicit(&bounds[axis], OrderedBits(p[axis]), memory_order_relaxed);
        atomic_fetch_max_explicit(&bounds[3u + axis], OrderedBits(p[axis]), memory_order_relaxed);
    }
}

// Each face's derived triangle count, and a zero terminator for the scan.
kernel void BooleanFaceCounts(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
    const uint2 range = ConnectivityView{b, job.Connectivity, 0u}.FaceHalfedges(handle);
    first[f] = range.y - range.x - 2u;
}

// An exclusive scan of a run of values in place, whose sum lands in the entry past them when the run ends with a zero.
kernel void BooleanScanSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *storage = BindlessBufferMutable(uint, b.Buffer, pc.Job.Slot);
    ScanBlockSum(storage + pc.ScanValues, pc.ScanCount, group_id, storage + pc.ScanBlocks, lane, simd_lane, simd_group, sums);
}
kernel void BooleanScanPrefix(
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *storage = BindlessBufferMutable(uint, b.Buffer, pc.Job.Slot);
    ScanBlockPrefix(storage + pc.ScanBlocks, (pc.ScanCount + ScanBlockElements - 1u) / ScanBlockElements, lane, simd_lane, simd_group, sums);
}
kernel void BooleanScanOffsets(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *storage = BindlessBufferMutable(uint, b.Buffer, pc.Job.Slot);
    ScanBlockOffsets(storage + pc.ScanValues, pc.ScanCount, group_id, storage + pc.ScanBlocks, storage + pc.ScanValues, lane, simd_lane, simd_group, sums);
}

// Each face's derived triangles: their combined vertices, corner handles, face and normal.
kernel void BooleanTriangles(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (f >= BoolFaceCount(job)) return;
    const uint side = f < job.Faces[0] ? 0u : 1u;
    const uint handle = ElementWorkDomain{b, job.FaceWork[side]}.Handle(side ? f - job.Faces[0] : f);
    const ElementWorkDomain vertices{b, job.VertexWork[side]};
    const uint vertex_base = side ? job.Vertices[0] : 0u;
    const ConnectivityView connectivity{b, job.Connectivity, 0u};
    const uint2 range = connectivity.FaceHalfedges(handle);
    const uint first = BindlessBuffer(uint, b.ObjectIdBuffer, job.FaceTriangleStartSlot)[connectivity.FaceAddress(handle)];
    const auto triangles = ReadElements<packed_uint3>(b, job.TriangleSlot, job.Connectivity.Tables.Triangles);
    device const packed_float3 *positions = ctx.Run<packed_float3>(job.Positions);
    const uint base = ctx.Run<uint>(job.FaceTriangleFirst)[f];
    for (uint k = 0u; k + 2u < range.y - range.x; ++k) {
        const uint t = base + k;
        const uint3 corners = uint3(triangles[first + k]);
        uint3 combined;
        for (uint i = 0u; i < 3u; ++i) combined[i] = vertex_base + vertices.Index(connectivity.CornerVertex(job.CornerSlot, corners[i]));
        const float3 p0 = float3(positions[combined.x]), p1 = float3(positions[combined.y]), p2 = float3(positions[combined.z]);
        const float3 normal = cross(p1 - p0, p2 - p0);
        const float length2 = dot(normal, normal);
        ctx.Run<packed_uint3>(job.TriangleVertices)[t] = packed_uint3(combined);
        ctx.Run<packed_uint3>(job.TriangleCorners)[t] = packed_uint3(corners);
        ctx.Run<uint>(job.TriangleFaces)[t] = f;
        ctx.Run<packed_float3>(job.TriangleNormals)[t] = packed_float3(length2 > 0.f ? normal * rsqrt(length2) : float3(0.f));
    }
}

// The triangle of combined face `f` with the side from corner `from` to corner `to`, as its halfedge.
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
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (h >= 3u * BoolTriangleCount(job)) return;
    const uint t = h / 3u, i = h % 3u;
    const uint3 corners = uint3(ctx.Run<packed_uint3>(job.TriangleCorners)[t]);
    const uint from = corners[i], to = corners[(i + 1u) % 3u];
    const ConnectivityView connectivity{b, job.Connectivity, 0u};
    const uint2 range = connectivity.FaceHalfedges(connectivity.HalfedgeFace(from));
    const bool edge = (from + 1u < range.y ? from + 1u : range.x) == to;
    uint pair = InvalidOffset;
    if (edge) {
        // The canonical halfedge arriving at `to` is the side, and its opposite arrives at `from` in the neighboring face.
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

// Each triangle's Morton code within its record's bounds, as Manifold's collider orders its leaves.
kernel void BooleanMorton(
    uint t [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (t >= BoolTriangleCount(job)) return;
    const uint side = t < job.Triangles[0] ? 0u : 1u;
    const uint local = side ? t - job.Triangles[0] : t;
    const BoolGeometry g = Geometry(ctx, job);
    const float3 a = g.Position(g.Start(3u * t)), c = g.Position(g.Start(3u * t + 1u)), d = g.Position(g.Start(3u * t + 2u));
    const float3 center = (min(min(a, c), d) + max(max(a, c), d)) * .5f;
    device const uint *bounds = ctx.Run<uint>(job.Bounds) + 6u * side;
    uint code = 0u;
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float low = OrderedFloat(bounds[axis]), extent = OrderedFloat(bounds[3u + axis]) - low;
        const float unit = extent > 0.f ? (center[axis] - low) / extent : 0.f;
        code |= SpreadBits3(uint(clamp(1024.f * unit, 0.f, 1023.f))) << (2u - axis);
    }
    ctx.Run<uint>(job.Keys[side])[local] = code;
    ctx.Run<uint>(job.Order[side])[local] = local;
}

inline RadixSortView BoolSort(BooleanContext ctx, BooleanJob job, uint side) {
    const uint count = job.Triangles[side], blocks = (count + 255u) / 256u;
    return {ctx.Run<uint>(job.Keys[side]), ctx.Run<uint>(job.Order[side]), ctx.Run<uint>(job.Temporary[side]), ctx.Run<uint>(job.Histogram[side]), ctx.Run<uint>(job.Totals[side]),
            count, blocks, 1u, 0u, ctx.Pc.Parameter, (ctx.Pc.Parameter & 4u) != 0u};
}
kernel void BooleanSortHistogram(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const BooleanContext ctx{b, pc};
    const RadixSortView sort = BoolSort(ctx, ctx.Job(), group.z);
    if (group.x >= sort.Blocks) return;
    threadgroup atomic_uint counts[16];
    RadixHistogram(sort, lane, group.x, counts);
}
kernel void BooleanSortPrefix(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    const BooleanContext ctx{b, pc};
    threadgroup uint sums[9];
    RadixPrefix(BoolSort(ctx, ctx.Job(), group.z), lane, group.x, sl, sg, sums);
}
kernel void BooleanSortScatter(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    const BooleanContext ctx{b, pc};
    const RadixSortView sort = BoolSort(ctx, ctx.Job(), group.z);
    if (group.x >= sort.Blocks) return;
    threadgroup uint groups[128];
    RadixScatter(sort, lane, group.x, sl, sg, groups);
}

// Karras' radix tree over sorted Morton codes, with leaf index order breaking equal codes.
struct BoolRadixTree {
    device const uint *Keys, *Order;
    int Leaves;
    int Prefix(int i, int j) const {
        if (j < 0 || j >= Leaves) return -1;
        const uint a = Keys[Order[i]], c = Keys[Order[j]];
        return a == c ? 32 + int(clz(uint(i) ^ uint(j))) : int(clz(a ^ c));
    }
    int RangeEnd(int i) const {
        int dir = Prefix(i, i + 1) - Prefix(i, i - 1);
        dir = (dir > 0) - (dir < 0);
        const int common = Prefix(i, i - dir);
        int max_length = 128;
        while (Prefix(i, i + dir * max_length) > common) max_length *= 4;
        int length = 0;
        for (int step = max_length / 2; step > 0; step /= 2)
            if (Prefix(i, i + dir * (length + step)) > common) length += step;
        return i + dir * length;
    }
    int FindSplit(int first, int last) const {
        const int common = Prefix(first, last);
        int split = first, step = last - first;
        do {
            step = (step + 1) >> 1;
            const int next = split + step;
            if (next < last && Prefix(first, next) > common) split = next;
        } while (step > 1);
        return split;
    }
};

// Links each internal node to its children and clears its refit counter.
kernel void BooleanTree(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint side = group.z, internal = group.x * 256u + lane, leaves = job.Triangles[side];
    if (leaves < 2u || internal + 1u >= leaves) return;
    const BoolRadixTree tree{ctx.Run<uint>(job.Keys[side]), ctx.Run<uint>(job.Order[side]), int(leaves)};
    const int end = tree.RangeEnd(int(internal)), first = min(int(internal), end), last = max(int(internal), end);
    int split = tree.FindSplit(first, last);
    const uint child0 = split == first ? 2u * uint(split) : 2u * uint(split) + 1u;
    ++split;
    const uint child1 = split == last ? 2u * uint(split) : 2u * uint(split) + 1u;
    ctx.Run<packed_uint2>(job.Children[side])[internal] = packed_uint2(child0, child1);
    device uint *parents = ctx.Run<uint>(job.NodeParents[side]);
    parents[child0] = parents[child1] = 2u * internal + 1u;
    ctx.Run<uint>(job.Counters[side])[internal] = 0u;
}

// Writes each leaf's triangle box and climbs, where the second child to arrive at a node writes its union.
kernel void BooleanRefit(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint side = group.z, leaf = group.x * 256u + lane, leaves = job.Triangles[side];
    if (leaf >= leaves) return;
    const BoolGeometry g = Geometry(ctx, job);
    const uint t = ctx.Run<uint>(job.Order[side])[leaf] + (side ? job.Triangles[0] : 0u);
    const float3 a = g.Position(g.Start(3u * t)), c = g.Position(g.Start(3u * t + 1u)), d = g.Position(g.Start(3u * t + 2u));
    float3 low = min(min(a, c), d), high = max(max(a, c), d);
    device atomic_uint *boxes = ctx.Run<atomic_uint>(job.NodeBoxes[side]);
    const auto store = [&](uint node) {
        for (uint k = 0u; k < 3u; ++k) {
            atomic_store_explicit(&boxes[6u * node + k], as_type<uint>(low[k]), memory_order_relaxed);
            atomic_store_explicit(&boxes[6u * node + 3u + k], as_type<uint>(high[k]), memory_order_relaxed);
        }
    };
    uint node = 2u * leaf;
    store(node);
    if (leaves < 2u) return;
    device const uint *parents = ctx.Run<uint>(job.NodeParents[side]);
    device const packed_uint2 *children = ctx.Run<packed_uint2>(job.Children[side]);
    device atomic_uint *counters = ctx.Run<atomic_uint>(job.Counters[side]);
    do {
        node = parents[node];
        const uint internal = (node - 1u) / 2u;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(&counters[internal], 1u, memory_order_relaxed) == 0u) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint2 pair = uint2(children[internal]);
        for (uint k = 0u; k < 3u; ++k) {
            low[k] = min(as_type<float>(atomic_load_explicit(&boxes[6u * pair.x + k], memory_order_relaxed)), as_type<float>(atomic_load_explicit(&boxes[6u * pair.y + k], memory_order_relaxed)));
            high[k] = max(as_type<float>(atomic_load_explicit(&boxes[6u * pair.x + 3u + k], memory_order_relaxed)), as_type<float>(atomic_load_explicit(&boxes[6u * pair.y + 3u + k], memory_order_relaxed)));
        }
        store(node);
    } while (node != 1u);
}

// Visits each crossing of a forward halfedge with the other record's triangles, in tree order.
template<typename Visit>
inline uint BoolEdgeHits(BooleanContext ctx, BooleanJob job, BoolGeometry g, uint h, Visit visit) {
    if (!g.Forward(h)) return 0u;
    const bool p_edge = h < 3u * job.Triangles[0];
    const float3 a = g.Position(g.Start(h)), c = g.Position(g.End(h));
    const float3 low = min(a, c), high = max(a, c);
    const BoolPerturb expand = BoolExpand(job);
    uint count = 0u;
    BoolTraverse(TreeOf(ctx, job, p_edge ? 1u : 0u), [&](device const float *box) { return BoxOverlaps(box, low, high); }, [&](uint t) {
        const BoolHit hit = Kernel12(g, expand, p_edge, h, t);
        if (!hit.Valid) return;
        visit(count, t, hit);
        ++count;
    });
    return count;
}

// Counts each halfedge's crossings for the scan, with a zero terminator.
kernel void BooleanIntersectCount(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    const uint halfedges = 3u * BoolTriangleCount(job);
    if (h > halfedges) return;
    device uint *first = ctx.Run<uint>(job.IntersectionFirst);
    first[h] = h == halfedges ? 0u : BoolEdgeHits(ctx, job, Geometry(ctx, job), h, [](uint, uint, BoolHit) {});
}

// Writes each halfedge's crossings at its scanned offset: the halfedge, the triangle, the signed crossing and its position.
kernel void BooleanIntersectEmit(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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

// Whether the halfedge crosses a triangle of the other record, which breaks its edge's component link.
inline bool BoolBroken(BooleanContext ctx, BooleanJob job, uint h) {
    device const uint *first = ctx.Run<uint>(job.IntersectionFirst);
    return first[h + 1u] != first[h];
}

kernel void BooleanLabelInit(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (v < BoolVertexCount(job)) ctx.Run<uint>(job.Labels)[v] = v;
}

// Pulls both ends of each unbroken forward edge, and their labels, to the lower of their labels.
kernel void BooleanLink(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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

// Jumps each label to its label's label.
kernel void BooleanJump(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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

// Ends the label rounds once one changes nothing, by zeroing their indirect arguments.
kernel void BooleanConverge(
    uint lane [[thread_index_in_threadgroup]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (v >= BoolVertexCount(job) || ctx.Run<uint>(job.Labels)[v] != v) return;
    const bool forward = v < job.Vertices[0];
    const BoolGeometry g = Geometry(ctx, job);
    const BoolPerturb expand = BoolExpand(job);
    const float2 point = g.Position(v).xy;
    int winding = 0;
    BoolTraverse(TreeOf(ctx, job, forward ? 1u : 0u), [&](device const float *box) { return BoxContainsXY(box, point); }, [&](uint t) {
        BoolFaceEdge edges[3];
        LoadFaceEdges(g, t, edges);
        const BoolPierce pierce = Kernel02(g, expand, forward, v, t, edges);
        if (pierce.Valid) winding += pierce.S;
    });
    ctx.Run<int>(job.Windings)[v] = forward ? winding : -winding;
}

// Each vertex's and intersection's signed output count by the operation: its winding through its component's root, or its crossing.
kernel void BooleanInclusion(
    uint e [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
// The P and Q triangles of an intersection side's group.
inline uint2 BoolGroupOf(BooleanContext ctx, BooleanJob job, BoolGeometry g, uint member) {
    const uint k = member / 2u, h = BoolRecordHalfedge(ctx, job, k);
    const uint face = member % 2u ? g.Pair(h) / 3u : h / 3u, t = BoolRecordTriangle(ctx, job, k);
    return k < job.FirstIntersections ? uint2(face, t) : uint2(t, face);
}

// The count scan's values: each triangle's polygon halfedges from its corners, each broken halfedge's partial edge entries,
// each group owner's new edge entries, then each vertex's and intersection's output vertices, then the terminator.
// A group's owner is its lowest intersection side.
kernel void BooleanCounts(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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

// Adds each intersection's copies to its triangle's and its halfedge's two triangles' polygon halfedge counts.
kernel void BooleanIntersectionSides(
    uint k [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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


// Names each output vertex's entry, and starts each triangle's polygon halfedge cursor at its run.
kernel void BooleanOutputs(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (i < BoolVertexCount(job) + job.Intersections) {
        const uint first = BoolOutputFirst(ctx, job, i), count = BoolCount(ctx, job, i);
        for (uint j = 0u; j < count; ++j) ctx.Run<uint>(job.OutputEntries)[first + j] = i;
    }
    if (i < BoolTriangleCount(job)) ctx.Run<uint>(job.Cursors)[i] = BoolPolygonFirst(ctx, job, i);
}

// Appends a halfedge to a triangle's polygon, failing the chain past the triangle's counted halfedges.
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

// Each unbroken forward edge's copies join both its triangles' polygons, reversed where its start's inclusion is negative.
kernel void BooleanWholeEdges(
    uint h [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const BooleanContext ctx{b, pc};
    const BooleanJob job = ctx.Job();
    if (h >= 3u * BoolTriangleCount(job)) return;
    const BoolGeometry g = Geometry(ctx, job);
    if (!g.Forward(h) || BoolBroken(ctx, job, h)) return;
    const uint start = g.Start(h), end = g.End(h), pair = g.Pair(h);
    const int inclusion = ctx.Run<int>(job.Inclusion)[start];
    device const uint *sources = ctx.Run<uint>(job.EdgeSources);
    for (uint j = 0u; j < uint(abs(inclusion)); ++j) {
        const uint a = BoolOutputFirst(ctx, job, start) + j, c = BoolOutputFirst(ctx, job, end) + j;
        const uint from = inclusion < 0 ? c : a, to = inclusion < 0 ? a : c;
        BoolAppend(ctx, job, h / 3u, from, to, sources[h]);
        BoolAppend(ctx, job, pair / 3u, to, from, sources[pair]);
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
    for (uint i = 1u; i < count; ++i) {
        const BoolEdgePos entry = entries[i];
        uint j = i;
        for (; j > 0u && BoolEdgePosBefore(entry, entries[j - 1u]); --j) entries[j] = entries[j - 1u];
        entries[j] = entry;
    }
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
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
    device const uint *sources = ctx.Run<uint>(job.EdgeSources);
    BoolPairUp(ctx, entries, count, [&](uint from, uint to) {
        BoolAppend(ctx, job, h / 3u, from, to, sources[h]);
        BoolAppend(ctx, job, pair / 3u, to, from, sources[pair]);
    });
}

// Each group owner pairs its group's crossings along their longest extent into the segments of the P and Q triangles' intersection,
// forward in P's polygon and backward in Q's.
kernel void BooleanNewEdges(
    uint member [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
        // As in Manifold, a P edge's crossing starts its own triangle's segment when its inclusion is nonnegative and a Q edge's when it is negative,
        // and the pair's triangle takes the opposite.
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
inline float BoolCross2(float2 a, float2 c, float2 d) { return (c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x); }
// Whether segments a-c and d-e cross at a point interior to both.
inline bool BoolSegmentsCross(float2 a, float2 c, float2 d, float2 e) {
    const float s0 = BoolCross2(d, e, a), s1 = BoolCross2(d, e, c), s2 = BoolCross2(a, c, d), s3 = BoolCross2(a, c, e);
    return ((s0 > 0.f && s1 < 0.f) || (s0 < 0.f && s1 > 0.f)) && ((s2 > 0.f && s3 < 0.f) || (s2 < 0.f && s3 > 0.f));
}

// A triangle's polygon: its sorted halfedges, its loops, and the linked loop nodes and triangles it emits.
struct BoolPolygon {
    BooleanContext Ctx;
    BooleanJob Job;
    device packed_uint3 *Halfedges; // Start and end output vertices, and source
    device uint *Refs, *Next, *Previous, *Links;
    device packed_float2 *Points;
    device uint *Triangles;
    uint Count, Emitted;
    float3 Normal;

    uint StartOf(uint i) const { return uint3(Halfedges[i]).x; }
    uint EndOf(uint i) const { return uint3(Halfedges[i]).y; }
    uint Vertex(uint node) const { return StartOf(Refs[node] & ~BoolRealOut); }
    float2 Point(uint node) const { return float2(Points[node]); }
    float2 Project(uint i) const { return BoolProject(Normal, BoolOutputPosition(Ctx, Job, StartOf(i))); }

    // Emits the triangle of three nodes, each corner reference marking whether the side to the next corner is its polygon halfedge.
    void Emit(uint a, uint c, uint d, bool ac, bool cd, bool da) {
        device uint *corners = Triangles + 3u * Emitted++;
        corners[0] = (Refs[a] & ~BoolRealOut) | (ac ? BoolRealOut : 0u);
        corners[1] = (Refs[c] & ~BoolRealOut) | (cd ? BoolRealOut : 0u);
        corners[2] = (Refs[d] & ~BoolRealOut) | (da ? BoolRealOut : 0u);
    }
    bool Real(uint node) const { return (Refs[node] & BoolRealOut) != 0u; }

    // Writes a loop's halfedges as linked nodes from `node` on, starting at halfedge `head`, and returns the node after them.
    uint WriteLoop(uint head, uint node) {
        const uint first = node;
        uint i = head;
        do {
            Refs[node] = i | BoolRealOut;
            Points[node] = packed_float2(Project(i));
            Next[node] = node + 1u;
            Previous[node] = node - 1u;
            ++node;
            i = Links[i];
        } while (i != head);
        Next[node - 1u] = first;
        Previous[first] = node - 1u;
        return node;
    }

    // Ear-clips the linked polygon through `node` of `count` nodes, skipping duplicate corners a bridge made when testing ears.
    void EarClip(uint node, uint count) {
        uint cursor = node;
        for (uint remaining = count; remaining > 3u; --remaining) {
            for (uint attempt = 0u; attempt < remaining; ++attempt) {
                const uint i = cursor, a = Previous[i], c = Next[i];
                const float2 pa = Point(a), pi = Point(i), pc = Point(c);
                bool ear = BoolCross2(pa, pi, pc) > 0.f;
                for (uint j = Next[c]; ear && j != a; j = Next[j]) {
                    const uint v = Vertex(j);
                    if (v == Vertex(a) || v == Vertex(i) || v == Vertex(c)) continue;
                    const float2 q = Point(j);
                    if (BoolCross2(pa, pi, q) >= 0.f && BoolCross2(pi, pc, q) >= 0.f && BoolCross2(pc, pa, q) >= 0.f) ear = false;
                }
                if (ear) break;
                cursor = c;
            }
            // A full scan without an ear clips at the cursor, for a degenerate polygon.
            const uint i = cursor, a = Previous[i], c = Next[i];
            Emit(a, i, c, Real(a), Real(i), false);
            Next[a] = c;
            Previous[c] = a;
            Refs[a] &= ~BoolRealOut;
            cursor = c;
        }
        const uint a = Previous[cursor], c = Next[cursor];
        Emit(a, cursor, c, Real(a), Real(cursor), Real(c));
    }
};

// Assembles each kept triangle's polygon halfedges into loops, bridges each hole into the outer loop containing it,
// and triangulates: a triangle as itself, a quad by Manifold's diagonal rule, a convex loop by a fan, and the rest by ear clipping.
kernel void BooleanFaces(
    uint t [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BooleanPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
    // Q's triangles face inward in a difference, so their polygons project against their normals.
    const bool inverted = job.Operation == BooleanOperation::Difference && t >= job.Triangles[0];
    BoolPolygon polygon{
        ctx, job, ctx.Run<packed_uint3>(job.Polygons) + first,
        ctx.Run<uint>(job.LoopRefs) + 2u * first, ctx.Run<uint>(job.LoopNext) + 2u * first, ctx.Run<uint>(job.LoopPrevious) + 2u * first, ctx.Run<uint>(job.LoopLinks) + 2u * first,
        ctx.Run<packed_float2>(job.LoopPoints) + 2u * first, ctx.Run<uint>(job.PolygonTriangles) + 6u * first,
        n, 0u, float3(ctx.Run<packed_float3>(job.TriangleNormals)[t]) * (inverted ? -1.f : 1.f),
    };
    // Sorting by start, end and source orders the halfedges independently of the appending passes' arrival.
    device packed_uint3 *halfedges = polygon.Halfedges;
    const auto before = [](uint3 a, uint3 c) { return a.x != c.x ? a.x < c.x : a.y != c.y ? a.y < c.y : a.z < c.z; };
    for (uint i = 1u; i < n; ++i) {
        const uint3 entry = uint3(halfedges[i]);
        uint j = i;
        for (; j > 0u && before(entry, uint3(halfedges[j - 1u])); --j) halfedges[j] = halfedges[j - 1u];
        halfedges[j] = packed_uint3(entry);
    }
    // Each loop follows the first unconsumed halfedge starting where the last ends, and closes back at its head.
    // Loop heads follow the links, and the refs mark consumed halfedges until the nodes overwrite them.
    device uint *links = polygon.Links, *refs = polygon.Refs;
    for (uint i = 0u; i < n; ++i) {
        links[i] = InvalidOffset;
        refs[i] = 0u;
    }
    uint loops = 0u;
    for (uint head = 0u; head < n; ++head) {
        if (links[head] != InvalidOffset) continue;
        uint i = head;
        do {
            const uint end = polygon.EndOf(i);
            uint lo = 0u, hi = n;
            while (lo < hi) {
                const uint middle = (lo + hi) / 2u;
                if (polygon.StartOf(middle) < end) lo = middle + 1u;
                else hi = middle;
            }
            while (lo < n && polygon.StartOf(lo) == end && refs[lo]) ++lo;
            if (lo == n || polygon.StartOf(lo) != end) {
                ctx.Fail();
                return;
            }
            refs[lo] = 1u;
            links[i] = lo;
            i = lo;
        } while (i != head);
        links[n + loops++] = head;
    }
    if (loops == 1u && n <= 4u) {
        polygon.WriteLoop(links[n], 0u);
        if (n == 3u) polygon.Emit(0u, 1u, 2u, true, true, true);
        else if (n == 4u) {
            const auto ccw = [&](uint a, uint c, uint d) { return BoolCross2(polygon.Point(a), polygon.Point(c), polygon.Point(d)) >= 0.f; };
            const float3 q0 = BoolOutputPosition(ctx, job, polygon.Vertex(0u)), q1 = BoolOutputPosition(ctx, job, polygon.Vertex(1u));
            const float3 q2 = BoolOutputPosition(ctx, job, polygon.Vertex(2u)), q3 = BoolOutputPosition(ctx, job, polygon.Vertex(3u));
            const bool second = !(ccw(0u, 1u, 2u) && ccw(0u, 2u, 3u)) || (ccw(1u, 2u, 3u) && ccw(0u, 1u, 3u) && distance_squared(q0, q2) > distance_squared(q1, q3));
            if (second) {
                polygon.Emit(1u, 2u, 3u, true, true, false);
                polygon.Emit(0u, 1u, 3u, true, false, true);
            } else {
                polygon.Emit(0u, 1u, 2u, true, true, false);
                polygon.Emit(0u, 2u, 3u, false, true, true);
            }
        }
    } else if (loops == 1u) {
        polygon.WriteLoop(links[n], 0u);
        bool convex = true;
        for (uint i = 0u; i < n && convex; ++i) convex = BoolCross2(polygon.Point((i + n - 1u) % n), polygon.Point(i), polygon.Point((i + 1u) % n)) >= 0.f;
        if (convex) {
            for (uint i = 1u; i + 1u < n; ++i) polygon.Emit(0u, i, i + 1u, i == 1u, true, i + 2u == n);
        } else polygon.EarClip(0u, n);
    } else {
        // A loop with negative area is a hole, joined to the smallest outer loop containing its first point, or else the largest.
        // Without an outer loop, every loop triangulates as one.
        const auto area = [&](uint head) {
            const float2 origin = polygon.Project(head);
            float sum = 0.f;
            uint i = head;
            do {
                sum += BoolCross2(origin, polygon.Project(i), polygon.Project(links[i]));
                i = links[i];
            } while (i != head);
            return sum;
        };
        const auto contains = [&](uint head, float2 p) {
            bool inside = false;
            uint i = head;
            do {
                const float2 a = polygon.Project(i), c = polygon.Project(links[i]);
                if ((a.y > p.y) != (c.y > p.y) && p.x < a.x + (p.y - a.y) * (c.x - a.x) / (c.y - a.y)) inside = !inside;
                i = links[i];
            } while (i != head);
            return inside;
        };
        bool outer_any = false;
        for (uint l = 0u; l < loops; ++l) outer_any = outer_any || area(links[n + l]) >= 0.f;
        // Each hole's outer loop, after the loop heads.
        device uint *owners = links + n + loops;
        for (uint l = 0u; l < loops; ++l) {
            const uint head = links[n + l];
            const float hole_area = area(head);
            owners[l] = InvalidOffset;
            if (!outer_any || hole_area >= 0.f) continue;
            const float2 p = polygon.Project(head);
            float best = INFINITY, largest = -1.f;
            uint inner = InvalidOffset, fallback = InvalidOffset;
            for (uint o = 0u; o < loops; ++o) {
                const float outer_area = area(links[n + o]);
                if (outer_area < 0.f) continue;
                if (outer_area > largest) {
                    largest = outer_area;
                    fallback = o;
                }
                if (outer_area < best && contains(links[n + o], p)) {
                    best = outer_area;
                    inner = o;
                }
            }
            owners[l] = inner != InvalidOffset ? inner : fallback;
        }
        uint node = 0u;
        for (uint o = 0u; o < loops; ++o) {
            if (owners[o] != InvalidOffset) continue;
            const uint polygon_first = node;
            node = polygon.WriteLoop(links[n + o], node);
            // Holes join from their rightmost point to the nearest polygon node whose bridge crosses no edge, rightmost holes first.
            while (true) {
                uint hole = InvalidOffset, hole_point = InvalidOffset;
                float right = -INFINITY;
                for (uint l = 0u; l < loops; ++l) {
                    if (owners[l] != o) continue;
                    uint i = links[n + l];
                    do {
                        const float x = polygon.Project(i).x;
                        if (x > right) {
                            right = x;
                            hole = l;
                            hole_point = i;
                        }
                        i = links[i];
                    } while (i != links[n + l]);
                }
                if (hole == InvalidOffset) break;
                owners[hole] = InvalidOffset - 1u;
                const float2 m = polygon.Project(hole_point);
                const auto visible = [&](float2 v) {
                    uint i = polygon_first;
                    do {
                        if (BoolSegmentsCross(m, v, polygon.Point(i), polygon.Point(polygon.Next[i]))) return false;
                        i = polygon.Next[i];
                    } while (i != polygon_first);
                    for (uint l = 0u; l < loops; ++l) {
                        if (owners[l] != o && l != hole) continue;
                        uint j = links[n + l];
                        do {
                            if (BoolSegmentsCross(m, v, polygon.Project(j), polygon.Project(links[j]))) return false;
                            j = links[j];
                        } while (j != links[n + l]);
                    }
                    return true;
                };
                uint bridge = InvalidOffset, nearest = polygon_first;
                float bridge_distance = INFINITY, nearest_distance = INFINITY;
                uint i = polygon_first;
                do {
                    const float d = distance_squared(m, polygon.Point(i));
                    if (d < nearest_distance) {
                        nearest_distance = d;
                        nearest = i;
                    }
                    if (d < bridge_distance && visible(polygon.Point(i))) {
                        bridge_distance = d;
                        bridge = i;
                    }
                    i = polygon.Next[i];
                } while (i != polygon_first);
                const uint v = bridge != InvalidOffset ? bridge : nearest, after = polygon.Next[v];
                // v -> hole loop from its rightmost point -> that point again -> v again -> v's old next.
                const uint hole_first = node;
                node = polygon.WriteLoop(hole_point, node);
                const uint m_copy = node++, v_copy = node++;
                polygon.Refs[m_copy] = polygon.Refs[hole_first] & ~BoolRealOut;
                polygon.Points[m_copy] = polygon.Points[hole_first];
                polygon.Refs[v_copy] = polygon.Refs[v];
                polygon.Points[v_copy] = polygon.Points[v];
                polygon.Refs[v] &= ~BoolRealOut;
                const uint hole_last = polygon.Previous[hole_first];
                polygon.Next[v] = hole_first;
                polygon.Previous[hole_first] = v;
                polygon.Next[hole_last] = m_copy;
                polygon.Previous[m_copy] = hole_last;
                polygon.Next[m_copy] = v_copy;
                polygon.Previous[v_copy] = m_copy;
                polygon.Next[v_copy] = after;
                polygon.Previous[after] = v_copy;
            }
            // A loop of two halfedges bounds no area and emits nothing.
            if (node - polygon_first >= 3u) polygon.EarClip(polygon_first, node - polygon_first);
        }
    }
    triangle_counts[t] = polygon.Emitted;
    atomic_fetch_add_explicit(ctx.Run<atomic_uint>(job.TriangleTotal), polygon.Emitted, memory_order_relaxed);
}
