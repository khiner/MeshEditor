#pragma once
// Manifold's shadow predicates over a geometry G with Position(v), Move(v) and Tol: values within the tolerance tie, and the moves decide ties.
#include "gpu/Types.h"

// The tie tolerance: RelativePrecision times `scale`, the smaller operand's largest coordinate magnitude, or zero when `scale` is not finite.
inline float BoolTolerance(float scale) { return isfinite(scale) ? RelativePrecision * scale : 0.f; }

// p < q, with p of P and q of Q.
inline bool Shadows(float p, float q, float move_p, float move_q, float tol) { return abs(p - q) > tol ? p < q : move_p < move_q; }

// The (y, z) of segment l-r at x clamped onto the segment, measured from the end nearer x.
inline float2 BoolInterpolate(float3 l, float3 r, float x) {
    x = clamp(x, min(l.x, r.x), max(l.x, r.x));
    const float dxl = x - l.x, dxr = x - r.x;
    const bool use_l = abs(dxl) < abs(dxr);
    const float3 d = r - l;
    const float lambda = (use_l ? dxl : dxr) / d.x;
    if (!isfinite(lambda) || !isfinite(d.y) || !isfinite(d.z)) return l.yz;
    return float2(lambda * d.y + (use_l ? l.y : r.y), lambda * d.z + (use_l ? l.z : r.z));
}
// The (y, z) of the whole line through l and r at x, measured from l, and its move when the ends move by dl and dr and x by dx.
inline void BoolLineAt(float3 l, float3 r, float3 dl, float3 dr, float x, float dx, thread float2 &yz, thread float2 &dyz) {
    const float width = r.x - l.x;
    const float t = width == 0.f ? .5f : (x - l.x) / width;
    const float dt = width == 0.f ? 0.f : ((dx - dl.x) - t * (dr.x - dl.x)) / width;
    yz = l.yz + t * (r.yz - l.yz);
    dyz = dl.yz + t * (dr.yz - dl.yz) + dt * (r.yz - l.yz);
}

// The crossing of segments a and b over one x interval whose y gaps bracket zero, as (x, y, a.z, b.z).
inline float4 BoolIntersect(float3 al, float3 ar, float3 bl, float3 br, float tol) {
    if (abs(bl.y - al.y) <= tol) bl.y = al.y;
    if (abs(br.y - ar.y) <= tol) br.y = ar.y;
    const float dyl = bl.y - al.y, dyr = br.y - ar.y;
    const bool use_l = abs(dyl) < abs(dyr);
    const float dx = ar.x - al.x;
    float lambda = (use_l ? dyl : dyr) / (dyl - dyr);
    if (!isfinite(lambda)) lambda = 0.f;
    lambda = use_l ? clamp(lambda, 0.f, 1.f) : clamp(lambda, -1.f, 0.f);
    const float ady = ar.y - al.y, bdy = br.y - bl.y;
    const bool use_a = abs(ady) < abs(bdy);
    return float4(
        lambda * dx + (use_l ? al.x : ar.x),
        lambda * (use_a ? ady : bdy) + (use_l ? (use_a ? al.y : bl.y) : (use_a ? ar.y : br.y)),
        lambda * (ar.z - al.z) + (use_l ? al.z : ar.z),
        lambda * (br.z - bl.z) + (use_l ? bl.z : br.z)
    );
}
// The heights (a.z, b.z) where the whole lines through a and b cross, measured from the left, and their moves when the ends move.
inline void BoolCrossingZ(float3 al, float3 ar, float3 bl, float3 br, float3 dal, float3 dar, float3 dbl, float3 dbr, thread float2 &zw, thread float2 &dzw) {
    const float dyl = bl.y - al.y, dyr = br.y - ar.y;
    const float ddyl = dbl.y - dal.y, ddyr = dbr.y - dar.y;
    const float gap = dyl - dyr;
    const float t = gap == 0.f ? .5f : dyl / gap;
    const float dt = gap == 0.f ? 0.f : (ddyl * gap - dyl * (ddyl - ddyr)) / (gap * gap);
    zw = float2(al.z + t * (ar.z - al.z), bl.z + t * (br.z - bl.z));
    dzw = float2(dal.z + t * (dar.z - dal.z) + dt * (ar.z - al.z), dbl.z + t * (dbr.z - dbl.z) + dt * (br.z - bl.z));
}

// Whether a vertex shadows an edge along x then y, with the edge's (y, z) at the vertex's x when their x ranges overlap.
struct BoolShadow {
    int S;
    float2 YZ;
    bool Overlaps;
};
// The vertex is P's when `forward`, and the edge's otherwise.
template<typename G> inline BoolShadow Shadow01(G g, bool forward, uint a0, uint b1s, uint b1e) {
    const float a0x = g.Position(a0).x, b1sx = g.Position(b1s).x, b1ex = g.Position(b1e).x;
    const float3 da0 = g.Move(a0), db1s = g.Move(b1s), db1e = g.Move(b1e);
    BoolShadow result{0, float2(0.f), false};
    result.S = forward ? int(Shadows(a0x, b1ex, da0.x, db1e.x, g.Tol)) - int(Shadows(a0x, b1sx, da0.x, db1s.x, g.Tol)) :
                         int(Shadows(b1sx, a0x, db1s.x, da0.x, g.Tol)) - int(Shadows(b1ex, a0x, db1e.x, da0.x, g.Tol));
    if (result.S == 0) return result;
    result.Overlaps = true;
    result.YZ = BoolInterpolate(g.Position(b1s), g.Position(b1e), a0x);
    const float ay = g.Position(a0).y;
    float y = result.YZ.x, dy = 0.f, day = 0.f;
    if (abs(ay - result.YZ.x) <= g.Tol) {
        float2 yz, dyz;
        BoolLineAt(g.Position(b1s), g.Position(b1e), db1s, db1e, a0x, da0.x, yz, dyz);
        y = yz.x;
        dy = dyz.x;
        day = da0.y;
    }
    if (forward ? !Shadows(ay, y, day, dy, g.Tol) : !Shadows(y, ay, dy, day, g.Tol)) result.S = 0;
    return result;
}

// The crossing of edges p and q in the xy projection: its sign, and (x, y) with each edge's z there.
// Prl and Qrl hold the shadow points on each side that the z stage interpolates between.
// S is zero without a crossing.
struct BoolCrossXY {
    int S;
    float4 XYZZ;
    float3 Prl[2], Qrl[2];
    uint VertRl[2];
    bool OfP[2];
};
template<typename G> inline BoolCrossXY ShadowCrossing(G g, uint p1s, uint p1e, uint q1s, uint q1e) {
    BoolCrossXY c;
    c.S = 0;
    c.XYZZ = float4(0.f);
    uint k = 0u;
    bool shadows = false;
    const uint ends[2][2] = {{p1s, p1e}, {q1s, q1e}};
    for (uint side = 0u; side < 2u; ++side) {
        for (uint i = 0u; i < 2u; ++i) {
            const uint v = ends[side][i];
            const BoolShadow shadow = Shadow01(g, side == 0u, v, ends[1u - side][0], ends[1u - side][1]);
            if (!shadow.Overlaps) continue;
            c.S += shadow.S * (i == 0u ? -1 : 1);
            if (k < 2u && (k == 0u || (shadow.S != 0) != shadows)) {
                shadows = shadow.S != 0;
                const float3 at = g.Position(v), across = float3(at.x, shadow.YZ.x, shadow.YZ.y);
                c.Prl[k] = side ? across : at;
                c.Qrl[k] = side ? at : across;
                c.VertRl[k] = v;
                c.OfP[k] = side == 0u;
                ++k;
            }
        }
    }
    if (c.S == 0 || k < 2u) {
        c.S = 0;
        return c;
    }
    c.XYZZ = BoolIntersect(c.Prl[0], c.Prl[1], c.Qrl[0], c.Qrl[1], g.Tol);
    return c;
}
