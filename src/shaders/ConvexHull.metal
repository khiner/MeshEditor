// Strict IEEE arithmetic without contraction makes every pass compute the same height for a point over a face.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// Quickhull over point sets in rounds.
// The lowest proposing apex of each set always inserts, so every open set grows each round.
#include "ConnectivityRead.metal"
#include "OrderedFloat.metal"
#include "gpu/ConvexHull.h"

using HullContext = JobContext<HullJob>;

// The set holding point p among sets ordered by first point.
template<typename Sets> inline uint SetContaining(Sets sets, uint count, uint p) {
    return PartitionPoint(count, [&](uint k) { return sets[k].First <= p; }) - 1u;
}

struct HullPoints {
    device HullSet *Sets;
    device const packed_float3 *Positions;
    device uint *Faces;
    uint SetCount;

    float3 Position(uint p) const { return float3(Positions[p]); }
    uint SetOf(uint p) const { return SetContaining(Sets, SetCount, p); }
};
inline bool Growing(uint status) { return status == HullSetOpen || status == HullSetPlanar; }
inline HullPoints Points(HullContext ctx, HullJob job) { return {ctx.Run<HullSet>(job.SetRecords), ctx.Run<packed_float3>(job.Positions), ctx.Run<uint>(job.PointFace), job.Sets}; }

inline float Height(HullFace face, float3 p) {
    const float4 plane = float4(face.Plane);
    return dot(plane.xyz, p) - plane.w;
}
inline bool Outside(HullFace face, float3 p) { return Height(face, p) > face.Tolerance; }
// A face that p lies on within tolerance also counts as seen, so nearly coplanar neighbors cannot pinch the horizon.
inline bool Sees(HullFace face, float3 p) { return Height(face, p) > -face.Tolerance; }
inline void ClearRound(thread HullFace &face) {
    face.ApexHeight = 0u;
    face.Apex = InvalidOffset;
    face.Claim = InvalidOffset;
    face.Next = InvalidOffset;
    face.Accepted = 0u;
    face.FirstNew = InvalidOffset;
}
inline HullFace MakeFace(HullPoints points, uint3 v, uint set, float tolerance) {
    const float3 a = points.Position(v.x), normal = cross(points.Position(v.y) - a, points.Position(v.z) - a);
    HullFace face;
    face.V = packed_uint3(v);
    face.Across = packed_uint3(uint3(InvalidOffset));
    face.Plane = packed_float4(float4(normal, dot(normal, a)));
    face.Tolerance = length(normal) * tolerance;
    face.Set = set;
    ClearRound(face);
    return face;
}
inline float3 VertexPosition(device const BindlessSet &b, constant HullPointsPushConstants &pc, ConnectivityRef at, uint handle) {
    return float3(BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot)[ConnectivityView{b, at}.VertexAddress(handle)].Position);
}
inline device packed_float3 *HullPositions(device const BindlessSet &b, constant HullPointsPushConstants &pc) {
    return reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint, b.Buffer, pc.Positions.Slot) + pc.Positions.Offset);
}

kernel void HullGather(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant HullPointsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Count) return;
    const uint handle = BindlessBuffer(uint, b.Buffer, pc.FirstHandles.Slot)[pc.FirstHandles.Offset + i];
    HullPositions(b, pc)[i] = packed_float3(VertexPosition(b, pc, pc.First, handle));
}

// Each set's pairwise sums of its first record's listed vertices with its second record's, in the first's space.
kernel void HullSums(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant HullPointsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (p >= pc.Count) return;
    device const HullSumSet *sets = reinterpret_cast<device const HullSumSet *>(BindlessBuffer(uint, b.Buffer, pc.Sets.Slot) + pc.Sets.Offset);
    const HullSumSet set = sets[SetContaining(sets, pc.SetCount, p)];
    const uint j = p - set.First;
    const uint a = BindlessBuffer(uint, b.Buffer, pc.FirstHandles.Slot)[pc.FirstHandles.Offset + set.FirstA + j / set.CountB];
    const uint c = BindlessBuffer(uint, b.Buffer, pc.SecondHandles.Slot)[pc.SecondHandles.Offset + set.FirstB + j % set.CountB];
    HullPositions(b, pc)[p] = packed_float3(VertexPosition(b, pc, pc.First, a) + pc.SecondSign * VertexPosition(b, pc, pc.Second, c) + float3(pc.SecondOffset));
}

// A point's distance in a seed stage: squared from its set's first point, from the line through the first two seeds, or from the plane through three.
inline float SeedDistance(HullPoints points, HullSet set, uint stage, float3 p) {
    const float3 a = points.Position(set.First);
    if (stage == 0u) return distance_squared(p, a);
    const float3 ab = points.Position(set.Seed[0]) - a;
    if (stage == 1u) return length(cross(ab, p - a));
    return abs(dot(cross(ab, points.Position(set.Seed[1]) - a), p - a));
}

kernel void HullSeedMax(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (p >= job.Points) return;
    const HullPoints points = Points(ctx, job);
    const uint s = points.SetOf(p);
    const HullSet set = points.Sets[s];
    if (set.Status != HullSetOpen) return;
    atomic_fetch_max_explicit(AtomicWord(&points.Sets[s].Best[pc.Parameter]), OrderedBits(SeedDistance(points, set, pc.Parameter, points.Position(p))), memory_order_relaxed);
}

// The lowest point at each set's farthest distance becomes the stage's seed.
kernel void HullSeedElect(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (p >= job.Points) return;
    const HullPoints points = Points(ctx, job);
    const uint s = points.SetOf(p);
    const HullSet set = points.Sets[s];
    if (set.Status != HullSetOpen || OrderedBits(SeedDistance(points, set, pc.Parameter, points.Position(p))) != set.Best[pc.Parameter]) return;
    atomic_fetch_min_explicit(AtomicWord(&points.Sets[s].Seed[pc.Parameter]), p, memory_order_relaxed);
}

// A set whose points lie within the flat tolerance of the seed plane is planar.
// A planar set lifts an apex over its first point by its extent as the fourth seed, or is flat when its seeds lie on a line.
// A set without a hull writes degenerate faces its first round compacts away.
kernel void HullSeedFaces(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (s >= job.Sets) return;
    const HullPoints points = Points(ctx, job);
    const HullSet set = points.Sets[s];
    device HullFace *faces = ctx.Run<HullFace>(job.Next) + 4u * s;
    uint seed[4] = {set.First, set.Seed[0], set.Seed[1], set.Seed[2]};
    const float extent = sqrt(OrderedFloat(set.Best[0]));
    bool open = set.Status == HullSetOpen;
    if (open && OrderedFloat(set.Best[2]) < HullFlat) {
        const float3 a = points.Position(seed[0]), normal = cross(points.Position(seed[1]) - a, points.Position(seed[2]) - a);
        const float length2 = dot(normal, normal);
        open = length2 > 0.f && extent > 0.f;
        if (open) {
            seed[3] = job.Points + s;
            ctx.Run<packed_float3>(job.Positions)[seed[3]] = packed_float3(a + normal * (extent * rsqrt(length2)));
            points.Sets[s].Status = HullSetPlanar;
        }
    }
    if (!open) {
        if (set.Status == HullSetOpen) points.Sets[s].Status = HullSetFlat;
        for (uint k = 0u; k < 4u; ++k) faces[k] = MakeFace(points, uint3(set.First), s, 0.f);
        return;
    }
    const float tolerance = HullTolerance * extent;
    points.Sets[s].Tolerance = tolerance;
    const uint3 triples[4] = {uint3(0u, 1u, 2u), uint3(0u, 1u, 3u), uint3(0u, 2u, 3u), uint3(1u, 2u, 3u)};
    for (uint k = 0u; k < 4u; ++k) {
        const uint3 v = uint3(seed[triples[k].x], seed[triples[k].y], seed[triples[k].z]);
        HullFace face = MakeFace(points, v, s, tolerance);
        if (Height(face, points.Position(seed[3u - k])) > 0.f) face = MakeFace(points, v.xzy, s, tolerance);
        faces[k] = face;
    }
    for (uint a = 0u; a < 4u; ++a) {
        const uint3 va = uint3(faces[a].V);
        uint3 across = uint3(InvalidOffset);
        for (uint k = 0u; k < 3u; ++k) {
            const uint from = va[(k + 1u) % 3u], to = va[k];
            for (uint c = 0u; c < 4u; ++c) {
                const uint3 vc = uint3(faces[c].V);
                for (uint m = 0u; m < 3u; ++m) {
                    if (c != a && vc[m] == from && vc[(m + 1u) % 3u] == to) across[k] = 4u * s + c;
                }
            }
        }
        faces[a].Across = packed_uint3(across);
    }
}

kernel void HullPropose(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (p >= job.Points) return;
    const HullPoints points = Points(ctx, job);
    const uint f = points.Faces[p];
    if (f >= job.Faces) return;
    device HullFace *faces = ctx.Run<HullFace>(job.Current);
    atomic_fetch_max_explicit(AtomicWord(&faces[f].ApexHeight), OrderedBits(Height(faces[f], points.Position(p))), memory_order_relaxed);
}

// The lowest point at its face's farthest height becomes the face's apex.
kernel void HullElect(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (p >= job.Points) return;
    const HullPoints points = Points(ctx, job);
    const uint f = points.Faces[p];
    if (f >= job.Faces) return;
    device HullFace *faces = ctx.Run<HullFace>(job.Current);
    if (OrderedBits(Height(faces[f], points.Position(p))) != faces[f].ApexHeight) return;
    atomic_fetch_min_explicit(AtomicWord(&faces[f].Apex), p, memory_order_relaxed);
}

// Each proposing face floods the faces its apex sees, claiming each for the lowest proposing face to reach it and stopping at faces a lower one holds.
// A face's worklist links through its claimed faces.
// A lower claim that takes one of them sends the walk into the taker's worklist, where every claim fails and the walk ends.
kernel void HullClaim(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (f >= job.Faces) return;
    device HullFace *faces = ctx.Run<HullFace>(job.Current);
    const uint apex = faces[f].Apex;
    if (apex == InvalidOffset) return;
    atomic_fetch_add_explicit(AtomicWord(ctx.Run<uint>(job.Proposals)), 1u, memory_order_relaxed);
    faces[f].Accepted = 1u;
    const float3 p = Points(ctx, job).Position(apex);
    if (atomic_fetch_min_explicit(AtomicWord(&faces[f].Claim), f, memory_order_relaxed) < f) return;
    atomic_store_explicit(AtomicWord(&faces[f].Next), InvalidOffset, memory_order_relaxed);
    uint head = f;
    for (uint iterations = 0u; head != InvalidOffset && iterations <= job.Faces; ++iterations) {
        const uint g = head;
        head = atomic_load_explicit(AtomicWord(&faces[g].Next), memory_order_relaxed);
        const uint3 across = uint3(faces[g].Across);
        for (uint i = 0u; i < 3u; ++i) {
            const uint h = across[i];
            if (atomic_load_explicit(AtomicWord(&faces[h].Claim), memory_order_relaxed) == f || !Sees(faces[h], p)) continue;
            if (atomic_fetch_min_explicit(AtomicWord(&faces[h].Claim), f, memory_order_relaxed) <= f) continue;
            atomic_store_explicit(AtomicWord(&faces[h].Next), head, memory_order_relaxed);
            head = h;
        }
    }
}

// An apex is accepted when it holds its own face, no face it holds neighbors a lower claim, and every face its apex sees next to its claims is its own.
// Accepted regions are then exact and share no edge, so their cones match sequential insertions.
kernel void HullAccept(
    uint g [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (g >= job.Faces) return;
    device HullFace *faces = ctx.Run<HullFace>(job.Current);
    const HullFace face = faces[g];
    if (face.Apex != InvalidOffset && face.Claim != g) faces[g].Accepted = 0u;
    const uint k = face.Claim;
    if (k == InvalidOffset) return;
    const float3 p = Points(ctx, job).Position(faces[k].Apex);
    const uint3 across = uint3(face.Across);
    for (uint i = 0u; i < 3u; ++i) {
        const HullFace neighbor = faces[across[i]];
        if (neighbor.Claim == k) continue;
        if ((neighbor.Claim != InvalidOffset && neighbor.Claim < k) || Sees(neighbor, p)) {
            faces[k].Accepted = 0u;
            return;
        }
    }
}

inline uint SlotOf(uint3 values, uint value) { return values.x == value ? 0u : values.y == value ? 1u : 2u; }

// The number of horizon edges of apex k's region leaving vertex v, rotating around v from edge m of region face c.
inline uint Exits(device const HullFace *faces, uint k, uint v, uint c, uint m, uint cap) {
    const uint start = c;
    uint exits = 0u;
    for (uint n = 0u; n <= cap; ++n) {
        const uint a = uint3(faces[c].Across)[m];
        if (faces[c].Claim == k && faces[a].Claim != k) ++exits;
        m = SlotOf(uint3(faces[a].V), v);
        c = a;
        if (c == start) break;
    }
    return exits;
}

// Each face held by an accepted apex dies and emits a new face per horizon edge, and every other face of an open set survives.
// A horizon that leaves one of its vertices twice is not one simple loop, and its set is abandoned.
kernel void HullHorizons(
    uint g [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (g > job.Faces) return;
    device uint *new_count = ctx.Run<uint>(job.NewCount), *survivor = ctx.Run<uint>(job.Survivor);
    if (g == job.Faces) {
        new_count[g] = 0u;
        survivor[g] = 0u;
        return;
    }
    const HullPoints points = Points(ctx, job);
    device const HullFace *faces = ctx.Run<HullFace>(job.Current);
    const HullFace face = faces[g];
    const bool open = Growing(points.Sets[face.Set].Status);
    const uint k = face.Claim;
    const bool held = k != InvalidOffset && faces[k].Accepted != 0u;
    survivor[g] = open && !held ? 1u : 0u;
    uint count = 0u;
    if (held) {
        const uint3 across = uint3(face.Across), v = uint3(face.V);
        for (uint i = 0u; i < 3u; ++i) {
            if (faces[across[i]].Claim == k) continue;
            ++count;
            if (Exits(faces, k, v[(i + 1u) % 3u], g, (i + 1u) % 3u, job.Faces) > 1u) points.Sets[face.Set].Status = HullSetAbandoned;
        }
    }
    new_count[g] = count;
}

// The next index of the new face over horizon edge m of dying face c: after the survivors, at its face's first new face plus the horizon edges before it.
inline uint NewFaceOf(device const HullFace *faces, device const uint *new_first, uint survivors, uint c, uint m) {
    const uint k = faces[c].Claim;
    const uint3 across = uint3(faces[c].Across);
    uint ordinal = 0u;
    for (uint j = 0u; j < m; ++j) ordinal += faces[across[j]].Claim != k ? 1u : 0u;
    return survivors + new_first[c] + ordinal;
}
// The next index of the new face over the horizon edge of apex k's region leaving vertex v, or arriving at v when `arriving`.
inline uint Fan(device const HullFace *faces, device const uint *new_first, uint survivors, uint k, uint v, uint c, uint m, uint cap, bool arriving) {
    for (uint n = 0u; n <= cap; ++n) {
        const uint a = uint3(faces[c].Across)[m];
        if (faces[a].Claim != k) return NewFaceOf(faces, new_first, survivors, c, m);
        m = (SlotOf(uint3(faces[a].V), v) + (arriving ? 2u : 0u)) % 3u;
        c = a;
    }
    return InvalidOffset;
}

// Each surviving face moves to its compact next index, with a neighbor that dies replaced by the new face over their shared edge, and its round state cleared.
kernel void HullScatter(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (f >= job.Faces) return;
    device const uint *survivor = ctx.Run<uint>(job.Survivor), *new_first = ctx.Run<uint>(job.NewCount);
    if (survivor[f] == survivor[f + 1u]) return;
    device const HullFace *faces = ctx.Run<HullFace>(job.Current);
    HullFace face = faces[f];
    uint3 across = uint3(face.Across);
    for (uint i = 0u; i < 3u; ++i) {
        const uint h = across[i];
        if (survivor[h] != survivor[h + 1u]) {
            across[i] = survivor[h];
            continue;
        }
        across[i] = NewFaceOf(faces, new_first, job.Survivors, h, SlotOf(uint3(faces[h].Across), f));
    }
    face.Across = packed_uint3(across);
    ClearRound(face);
    ctx.Run<HullFace>(job.Next)[survivor[f]] = face;
}

// Each dying face emits a new face from each of its horizon edges to its apex, linked to the survivor across the edge and to the cone faces on either side.
kernel void HullEmit(
    uint g [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (g >= job.Faces) return;
    device const uint *survivor = ctx.Run<uint>(job.Survivor), *new_first = ctx.Run<uint>(job.NewCount);
    if (new_first[g] == new_first[g + 1u]) return;
    const HullPoints points = Points(ctx, job);
    device HullFace *faces = ctx.Run<HullFace>(job.Current);
    device HullFace *next = ctx.Run<HullFace>(job.Next);
    const HullFace face = faces[g];
    const uint k = face.Claim, apex = faces[k].Apex;
    const float tolerance = points.Sets[face.Set].Tolerance;
    const uint3 across = uint3(face.Across), v = uint3(face.V);
    uint n = job.Survivors + new_first[g];
    for (uint i = 0u; i < 3u; ++i) {
        if (faces[across[i]].Claim == k) continue;
        const uint from = v[i], to = v[(i + 1u) % 3u];
        HullFace created = MakeFace(points, uint3(from, to, apex), face.Set, tolerance);
        created.Across = packed_uint3(uint3(survivor[across[i]], Fan(faces, new_first, job.Survivors, k, to, g, (i + 1u) % 3u, job.Faces, false), Fan(faces, new_first, job.Survivors, k, from, g, (i + 2u) % 3u, job.Faces, true)));
        next[n] = created;
        atomic_fetch_min_explicit(AtomicWord(&faces[k].FirstNew), n, memory_order_relaxed);
        ++n;
    }
}

// A point of a flat or abandoned set is inside, and a point on a surviving face moves with it to its compact index.
// A point whose face died joins the first cone face of its apex that it lies outside of, or is inside.
// In the first round every point joins the first seed face it lies outside of.
kernel void HullAssign(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const HullContext ctx{b, pc};
    const HullJob job = ctx.Job();
    if (p >= job.Points) return;
    const HullPoints points = Points(ctx, job);
    const uint f = points.Faces[p];
    if (f == HullPointInside || f == HullPointUsed) return;
    device const HullFace *next = ctx.Run<HullFace>(job.Next);
    const float3 position = points.Position(p);
    if (f == HullPointUnassigned) {
        const uint s = points.SetOf(p);
        uint assigned = HullPointInside;
        if (Growing(points.Sets[s].Status)) {
            for (uint k = 0u; k < 4u; ++k) {
                if (any(uint3(next[4u * s + k].V) == p)) assigned = HullPointUsed;
            }
            for (uint k = 0u; k < 4u && assigned == HullPointInside; ++k) {
                if (Outside(next[4u * s + k], position)) assigned = 4u * s + k;
            }
        }
        points.Faces[p] = assigned;
        return;
    }
    device const HullFace *faces = ctx.Run<HullFace>(job.Current);
    const HullFace face = faces[f];
    if (!Growing(points.Sets[face.Set].Status)) {
        points.Faces[p] = HullPointInside;
        return;
    }
    const uint k = face.Claim;
    if (k == InvalidOffset || faces[k].Accepted == 0u) {
        points.Faces[p] = ctx.Run<uint>(job.Survivor)[f];
        return;
    }
    const HullFace proposing = faces[k];
    if (proposing.Apex == p) {
        points.Faces[p] = HullPointUsed;
        return;
    }
    uint assigned = HullPointInside;
    const uint start = proposing.FirstNew;
    for (uint g = start, n = 0u; g != InvalidOffset && n <= job.NewFaces; ++n) {
        if (Outside(next[g], position)) {
            assigned = g;
            break;
        }
        g = uint3(next[g].Across).y;
        if (g == start) break;
    }
    points.Faces[p] = assigned;
}
