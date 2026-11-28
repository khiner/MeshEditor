// Selection fields and curvature over a record's connectivity, one thread per live element.
#include "ConnectivityRead.metal"
#include "gpu/AttributeField.h"

inline float3 FieldPosition(device const BindlessSet &b, uint vertex_slot, ConnectivityView conn, uint v) {
    return float3(BindlessBuffer(Vertex, b.VertexBuffer, vertex_slot)[conn.VertexAddress(v)].Position);
}
inline bool FieldSharesFace(ConnectivityView conn, uint e, uint other) {
    const uint2 a = conn.EdgeFaces(e), c = conn.EdgeFaces(other);
    return (a.x != InvalidOffset && (a.x == c.x || a.x == c.y)) || (a.y != InvalidOffset && (a.y == c.x || a.y == c.y));
}
template<typename Visit> inline void FieldNeighbors(constant NeighborFieldJob &pc, ConnectivityView conn, uint handle, Visit visit) {
    if (pc.Domain == Element::Vertex) {
        conn.ForEachIncidentEdge(handle, [&](uint e) { visit(conn.OtherEnd(conn.EdgeHalfedge(e), handle)); });
    } else if (pc.Domain == Element::Face && pc.Adjacency == FieldAdjacency::Radial) {
        const uint2 range = conn.FaceHalfedges(handle);
        for (uint h = range.x; h < range.y; ++h) {
            const uint a = conn.From(h), c = conn.VertexOf(h);
            for (const auto item : conn.Fan(a)) {
                if (item.y != handle && conn.FanSide(item, c) != InvalidOffset) visit(item.y);
            }
        }
    } else if (pc.Domain == Element::Face) {
        const uint2 range = conn.FaceHalfedges(handle);
        for (uint h = range.x; h < range.y; ++h) {
            const uint o = conn.Opposite(h);
            if (o == InvalidOffset) continue;
            const uint g = conn.HalfedgeFace(o);
            if (g != InvalidOffset) visit(g);
        }
    } else if (pc.Adjacency == FieldAdjacency::EdgeRing) {
        const uint2 faces = conn.EdgeFaces(handle);
        for (uint side = 0u; side < 2u; ++side) {
            const uint f = side ? faces.y : faces.x;
            if (f == InvalidOffset) continue;
            const uint2 range = conn.FaceHalfedges(f);
            if (range.y - range.x != 4u) continue;
            const uint h = side ? conn.Opposite(conn.EdgeHalfedge(handle)) : conn.EdgeHalfedge(handle);
            visit(conn.Edge(range.x + (h - range.x + 2u) % 4u));
        }
    } else {
        const uint2 ends = conn.EdgeEnds(handle);
        for (uint side = 0u; side < 2u; ++side) {
            const uint v = side ? ends.y : ends.x;
            if (pc.Adjacency == FieldAdjacency::Domain) {
                conn.ForEachIncidentEdge(v, [&](uint e) {
                    if (e != handle) visit(e);
                });
            } else {
                uint count = 0u, across = InvalidOffset;
                conn.ForEachIncidentEdge(v, [&](uint e) {
                    ++count;
                    if (e != handle && across == InvalidOffset && !FieldSharesFace(conn, handle, e)) across = e;
                });
                if (count == 4u && across != InvalidOffset) visit(across);
            }
        }
    }
}
inline void FieldWriteBool(device const BindlessSet &b, NamedAttributeRef output, uint handle, bool value) {
    WriteNamedAttribute(b, output, AttributeType::Bool, handle, uint4(value ? 1u : 0u, 0u, 0u, 0u));
}
inline bool FieldSeed(device const BindlessSet &b, constant NeighborFieldJob &pc, uint handle) { return ReadNamedAttribute(b, pc.Seed, AttributeType::Bool, handle).x != 0u; }
inline device uint *FieldWords(device const BindlessSet &b, SlotOffset run) { return BindlessBufferMutable(uint, b.Buffer, run.Slot) + run.Offset; }
inline bool FieldTakes(device const BindlessSet &b, constant NeighborFieldJob &pc, uint handle) { return pc.Restrict.Slot == InvalidSlot || FieldWords(b, pc.Restrict)[handle] != 0u; }

// Boundary: edges with fewer than two faces, the vertices on them, or the faces touching them.
kernel void AttributeFieldBoundary(JOB_TABLE_ARGS(AttributeFieldJob)) {
    JOB_THREAD(job, i);
    const uint handle = BlockHandle(i, b, job.Blocks);
    if (handle == InvalidOffset) return;
    const ConnectivityView conn{b, job.Connectivity};
    bool boundary = false;
    if (job.Domain == Element::Edge) boundary = conn.EdgeFaceCount(handle) < 2u;
    else if (job.Domain == Element::Vertex) {
        conn.ForEachIncidentEdge(handle, [&](uint e) { boundary = boundary || conn.EdgeFaceCount(e) < 2u; });
    } else {
        const uint2 range = conn.FaceHalfedges(handle);
        for (uint h = range.x; h < range.y; ++h) boundary = boundary || conn.EdgeFaceCount(conn.Edge(h)) < 2u;
    }
    FieldWriteBool(b, job.Output, handle, boundary);
}

// Plane Side: vertices on the side, or edges and faces whose vertices all are.
inline bool FieldVertexOnSide(device const BindlessSet &b, constant AttributeFieldJob &job, ConnectivityView conn, uint v) {
    const float3 p = FieldPosition(b, job.VertexSlot, conn, v);
    return job.Side == PlaneSideOf(dot(float3(job.PlaneNormal), p) - job.PlaneOffset, p, job.Tolerance);
}
kernel void AttributeFieldPlaneSide(JOB_TABLE_ARGS(AttributeFieldJob)) {
    JOB_THREAD(job, i);
    const uint handle = BlockHandle(i, b, job.Blocks);
    if (handle == InvalidOffset) return;
    const ConnectivityView conn{b, job.Connectivity};
    bool selected = true;
    if (job.Domain == Element::Vertex) selected = FieldVertexOnSide(b, job, conn, handle);
    else if (job.Domain == Element::Edge) {
        const uint2 ends = conn.EdgeEnds(handle);
        selected = FieldVertexOnSide(b, job, conn, ends.x) && FieldVertexOnSide(b, job, conn, ends.y);
    } else {
        const uint2 range = conn.FaceHalfedges(handle);
        for (uint h = range.x; h < range.y; ++h) selected = selected && FieldVertexOnSide(b, job, conn, conn.VertexOf(h));
    }
    FieldWriteBool(b, job.Output, handle, selected);
}

// Declares the directly bound field job `pc` and the handle `handle` of the thread's slot of the job's blocks, or InvalidOffset at an empty slot.
#define FIELD_HANDLE(handle) JOB_THREAD(pc, i); const uint handle = BlockHandle(i, b, pc.Blocks)
kernel void AttributeFieldLabelInit(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset) return;
    FieldWords(b, pc.Labels)[handle] = handle;
    FieldWords(b, pc.Flags)[handle] = pc.Parity ? InvalidOffset : 0u;
    if (pc.Restrict.Slot != InvalidSlot) FieldWords(b, pc.Restrict)[handle] = 0u;
}
kernel void AttributeFieldMarkList(JOB_TABLE_ARGS(NeighborFieldJob)) {
    JOB_THREAD(pc, i);
    if (i >= pc.ListCount) return;
    FieldWords(b, pc.Restrict)[FieldWords(b, pc.List)[i]] = 1u;
}
// Each label falls to the least of its own, its neighbors' and its label's label, until a round changes nothing.
kernel void AttributeFieldLink(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset || !FieldTakes(b, pc, handle)) return;
    const ConnectivityView conn{b, pc.Connectivity};
    device uint *labels = FieldWords(b, pc.Labels);
    const uint current = labels[handle];
    uint least = min(current, labels[current]);
    FieldNeighbors(pc, conn, handle, [&](uint n) { if (FieldTakes(b, pc, n)) least = min(least, labels[n]); });
    if (least < current) {
        labels[handle] = least;
        FieldWords(b, pc.Changed)[0] = 1u;
    }
}
// Each distance falls to zero at its component's root and otherwise to one past the least of its neighbors', until a round changes nothing.
// A distance only falls and always bounds a walk to the root, so the rounds end at the shortest path lengths in any thread order.
kernel void AttributeFieldDistance(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset || !FieldTakes(b, pc, handle)) return;
    const ConnectivityView conn{b, pc.Connectivity};
    device uint *distances = FieldWords(b, pc.Flags);
    const uint current = distances[handle];
    uint least = FieldWords(b, pc.Labels)[handle] == handle ? 0u : current;
    FieldNeighbors(pc, conn, handle, [&](uint n) { if (FieldTakes(b, pc, n) && distances[n] != InvalidOffset) least = min(least, distances[n] + 1u); });
    if (least < current) {
        distances[handle] = least;
        FieldWords(b, pc.Changed)[0] = 1u;
    }
}
kernel void AttributeFieldMarkSeeds(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset || !FieldSeed(b, pc, handle)) return;
    FieldWords(b, pc.Flags)[FieldWords(b, pc.Labels)[handle]] = 1u;
}
kernel void AttributeFieldComponents(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset) return;
    FieldWriteBool(b, pc.Output, handle, FieldWords(b, pc.Flags)[FieldWords(b, pc.Labels)[handle]] != 0u);
}

// Each element starts from its seed, or its complement when shrinking.
kernel void AttributeFieldGrowInit(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset) return;
    FieldWords(b, pc.Labels)[handle] = FieldSeed(b, pc, handle) != (pc.Complement != 0u) ? 1u : 0u;
}
// Each element writes the union of its own and its neighbors' flags from the previous round.
kernel void AttributeFieldGrow(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset) return;
    const ConnectivityView conn{b, pc.Connectivity};
    device const uint *from = FieldWords(b, pc.Labels);
    uint flagged = from[handle];
    FieldNeighbors(pc, conn, handle, [&](uint n) { flagged |= from[n]; });
    FieldWords(b, pc.Flags)[handle] = flagged;
}
kernel void AttributeFieldGrowOutput(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(handle);
    if (handle == InvalidOffset) return;
    FieldWriteBool(b, pc.Output, handle, (FieldWords(b, pc.Labels)[handle] != 0u) != (pc.Complement != 0u));
}

struct FieldTriangles {
    ElementValues<const packed_uint3> Triangles;
    uint First, Count;
    uint3 operator[](uint k) const { return uint3(Triangles[First + k]); }
};
inline FieldTriangles FieldFaceTriangles(device const BindlessSet &b, constant NeighborFieldJob &pc, ConnectivityView conn, uint f) {
    const uint2 range = conn.FaceHalfedges(f);
    return {ReadElements<packed_uint3>(b, pc.TriangleSlot, pc.Connectivity.Tables.Triangles), BindlessBuffer(uint, b.ObjectIdBuffer, pc.FaceTriangleStartSlot)[conn.FaceAddress(f)], range.y - range.x - 2u};
}
inline bool FieldTriangleHas(uint3 triangle, uint corner) { return any(triangle == corner); }
inline float3 FieldTriangleNormal(device const BindlessSet &b, constant NeighborFieldJob &pc, ConnectivityView conn, uint3 triangle) {
    const float3 p0 = FieldPosition(b, pc.VertexSlot, conn, conn.VertexOf(triangle.x)), p1 = FieldPosition(b, pc.VertexSlot, conn, conn.VertexOf(triangle.y)), p2 = FieldPosition(b, pc.VertexSlot, conn, conn.VertexOf(triangle.z));
    return normalize(cross(p1 - p0, p2 - p0));
}
// Writes the normal of the triangle across the edge from corner `a` to corner `c` of face `f`, and returns false at a boundary.
inline bool FieldNeighborNormal(device const BindlessSet &b, constant NeighborFieldJob &pc, ConnectivityView conn, uint f, uint3 triangle, uint a, uint c, thread float3 &normal) {
    uint face = f, first = a, second = c;
    if (conn.Next(a, f) == c || conn.Next(c, f) == a) {
        const uint o = conn.Opposite(conn.Next(a, f) == c ? c : a);
        if (o == InvalidOffset) return false;
        face = conn.HalfedgeFace(o);
        if (face == InvalidOffset) return false;
        first = o;
        second = conn.Previous(o);
    }
    const FieldTriangles triangles = FieldFaceTriangles(b, pc, conn, face);
    for (uint k = 0u; k < triangles.Count; ++k) {
        const uint3 other = triangles[k];
        if (face == f && all(other == triangle)) continue;
        if (FieldTriangleHas(other, first) && FieldTriangleHas(other, second)) {
            normal = FieldTriangleNormal(b, pc, conn, other);
            return true;
        }
    }
    return false;
}
// Gaussian curvature is the angle defect at the vertex.
// Mean curvature is the sum of its edges' signed dihedral angles weighted by edge length.
// Both divide by a third of the incident triangle area.
kernel void AttributeFieldCurvature(JOB_TABLE_ARGS(NeighborFieldJob)) {
    FIELD_HANDLE(v);
    if (v == InvalidOffset) return;
    const ConnectivityView conn{b, pc.Connectivity};
    float gaussian = 2.f * M_PI_F, mean = 0.f, area = 0.f, degree = 0.f;
    for (const auto item : conn.Fan(v)) {
        const uint corner = item.x, f = item.y;
        if (f == InvalidOffset) continue;
        const FieldTriangles triangles = FieldFaceTriangles(b, pc, conn, f);
        for (uint k = 0u; k < triangles.Count; ++k) {
            const uint3 triangle = triangles[k];
            if (!FieldTriangleHas(triangle, corner)) continue;
            const uint at = triangle.x == corner ? 0u : triangle.y == corner ? 1u : 2u;
            float3 position[3], edge[3];
            float length[3];
            for (uint c = 0u; c < 3u; ++c) position[c] = FieldPosition(b, pc.VertexSlot, conn, conn.VertexOf(triangle[c]));
            for (uint c = 0u; c < 3u; ++c) {
                edge[c] = position[(c + 1u) % 3u] - position[c];
                length[c] = metal::length(edge[c]);
                if (length[c] > 0.f) edge[c] /= length[c];
            }
            float phi[3];
            phi[0] = acos(clamp(-dot(edge[2], edge[0]), -1.f, 1.f));
            phi[1] = acos(clamp(-dot(edge[0], edge[1]), -1.f, 1.f));
            phi[2] = M_PI_F - phi[0] - phi[1];
            gaussian -= phi[at];
            area += length[0] * length[1] * metal::length(cross(edge[0], edge[1])) / 6.f;
            degree += 1.f;
            const float3 normal = normalize(cross(position[1] - position[0], position[2] - position[0]));
            for (uint side = 0u; side < 2u; ++side) {
                const uint c = side ? (at + 2u) % 3u : at;
                float3 across;
                if (FieldNeighborNormal(b, pc, conn, f, triangle, triangle[c], triangle[(c + 1u) % 3u], across)) mean += .25f * length[c] * asin(clamp(dot(cross(normal, across), edge[c]), -1.f, 1.f));
            }
        }
    }
    const float value = area > 0.f ? (pc.Mean ? mean : gaussian) * degree / (6.f * area) : 0.f;
    WriteNamedAttribute(b, pc.Output, AttributeType::Float, v, uint4(as_type<uint>(value), 0u, 0u, 0u));
}

// Visits each element of domain `from` related to element t of domain `to`.
// A vertex relates to its edges, the faces around it and their corners at it.
// An edge relates to its two vertices, its faces and the corners of its faces at its two ends.
// A face relates to its vertices, edges and corners, and a corner to its vertex, its face and the two edges of its face at it.
// A wire corner relates to its vertex alone.
template<typename Visit> inline void ForEachRelated(ConnectivityView conn, AttributeDomain from, AttributeDomain to, uint t, Visit visit) {
    if (from == to) {
        visit(t);
    } else if (to == AttributeDomain::Vertex) {
        if (from == AttributeDomain::Edge) {
            conn.ForEachIncidentEdge(t, visit);
            return;
        }
        for (const auto item : conn.Fan(t)) {
            if (item.y != InvalidOffset) visit(from == AttributeDomain::Face ? item.y : item.x);
        }
    } else if (to == AttributeDomain::Edge) {
        const uint first = conn.EdgeHalfedge(t);
        if (from == AttributeDomain::Vertex) {
            visit(conn.From(first));
            visit(conn.VertexOf(first));
            return;
        }
        for (uint side = 0u; side < 2u; ++side) {
            const uint h = side ? conn.Opposite(first) : first;
            if (h == InvalidOffset || conn.HalfedgeFace(h) == InvalidOffset) continue;
            if (from == AttributeDomain::Face) {
                visit(conn.HalfedgeFace(h));
            } else {
                visit(conn.Previous(h));
                visit(h);
            }
        }
    } else if (to == AttributeDomain::Face) {
        const uint2 range = conn.FaceHalfedges(t);
        for (uint h = range.x; h < range.y; ++h) visit(from == AttributeDomain::Vertex ? conn.VertexOf(h) : from == AttributeDomain::Edge ? conn.Edge(h) : h);
    } else if (from == AttributeDomain::Vertex) {
        visit(conn.VertexOf(t));
    } else {
        const uint face = conn.HalfedgeFace(t);
        if (face == InvalidOffset) return;
        if (from == AttributeDomain::Face) {
            visit(face);
        } else {
            visit(conn.Edge(t));
            visit(conn.Edge(conn.Next(t, face)));
        }
    }
}

// The reduction of the values a conversion visits, as the words of a value of its type.
struct DomainReducer {
    AttributeType Type;
    DomainReduction Reduction;
    uint Count = 0u;
    bool Flag = false;
    long Sum = 0;
    int Integer = 0;
    float4 Value = float4(0.f);

    void Add(uint4 words) {
        if (Type == AttributeType::Bool) {
            const bool value = words.x != 0u;
            Flag = Count == 0u ? value : Reduction == DomainReduction::All ? Flag && value : Flag || value;
        } else if (Type == AttributeType::Int) {
            const int value = as_type<int>(words.x);
            Sum += long(value);
            Integer = Count == 0u ? value : Reduction == DomainReduction::Min ? min(Integer, value) : max(Integer, value);
        } else {
            const float4 value = as_type<float4>(words);
            Value = Count == 0u ? value : Reduction == DomainReduction::Mean ? Value + value : Reduction == DomainReduction::Min ? min(Value, value) : max(Value, value);
        }
        ++Count;
    }
    uint4 Result() const {
        if (Count == 0u) return uint4(0u);
        if (Type == AttributeType::Bool) return uint4(Flag ? 1u : 0u, 0u, 0u, 0u);
        if (Type == AttributeType::Int) {
            if (Reduction != DomainReduction::Mean) return uint4(as_type<uint>(Integer), 0u, 0u, 0u);
            const long n = long(Count), quotient = Sum / n, remainder = Sum % n;
            const long rounded = 2 * abs(remainder) >= n ? quotient + (Sum < 0 ? -1 : 1) : quotient;
            return uint4(as_type<uint>(int(rounded)), 0u, 0u, 0u);
        }
        if (Reduction != DomainReduction::Mean) return as_type<uint4>(Value);
        return as_type<uint4>(Value * precise::divide(1.f, float(Count)));
    }
};

// Convert Domain: each target element's reduction of the source values of its related elements.
// A threadgroup covers one block of its job's target domain.
kernel void AttributeFieldConvertDomain(JOB_TABLE_ARGS(DomainConversionJob)) {
    JOB_TILE_THREAD(job, slot, MeshElementBlockSize);
    const uint t = BlockHandle(slot, b, job.Blocks);
    if (t == InvalidOffset) return;
    const ConnectivityView conn{b, job.Connectivity};
    DomainReducer reducer{job.Type, job.Reduction};
    ForEachRelated(conn, job.From, job.To, t, [&](uint s) { reducer.Add(ReadNamedAttribute(b, job.Source, job.Type, s)); });
    WriteNamedAttribute(b, job.Output, job.Type, t, reducer.Result());
}

kernel void AttributeSelectionMasks(JOB_TABLE_ARGS(AttributeMaskJob)) {
    JOB_THREAD(pc, i);
    if (i >= pc.Words) return;
    const uint block = BindlessBuffer(uint, b.Buffer, pc.Blocks.Slot)[pc.Blocks.Offset + i / MeshElementBlockWords], word = i % MeshElementBlockWords;
    const MeshElementBlock membership = BindlessBuffer(MeshElementBlock, b.Buffer, pc.MembershipSlot)[block];
    const uint value = pc.Attribute.ValuesSlot != InvalidSlot ? BindlessBuffer(uint, b.Buffer, pc.Attribute.ValuesSlot)[(pc.Attribute.Offset + membership.Local * MeshElementBlockSize) / 32u + word] : 0u;
    BindlessBufferMutable(uint, b.Buffer, pc.Masks.Slot)[pc.Masks.Offset + block * MeshElementBlockWords + word] = value;
}
