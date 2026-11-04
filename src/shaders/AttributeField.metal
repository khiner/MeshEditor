// Selection fields and curvature over a record's connectivity, one thread per live element.
#include "ConnectivityRead.metal"
#include "gpu/AttributeField.h"

inline float3 FieldPosition(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, ConnectivityView conn, uint v) {
    return float3(BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot)[conn.VertexAddress(v)].Position);
}
inline bool FieldSharesFace(ConnectivityView conn, uint e, uint other) {
    const uint2 a = conn.EdgeFaces(e), c = conn.EdgeFaces(other);
    return (a.x != InvalidOffset && (a.x == c.x || a.x == c.y)) || (a.y != InvalidOffset && (a.y == c.x || a.y == c.y));
}
template<typename Visit> inline void FieldNeighbors(constant AttributeFieldPushConstants &pc, ConnectivityView conn, uint handle, Visit visit) {
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
inline void FieldWriteBool(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, uint handle, bool value) {
    WriteNamedAttribute(b, pc.Output, AttributeType::Bool, handle, uint4(value ? 1u : 0u, 0u, 0u, 0u));
}
inline bool FieldSeed(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, uint handle) { return ReadNamedAttribute(b, pc.Seed, AttributeType::Bool, handle).x != 0u; }
inline device uint *FieldWords(device const BindlessSet &b, SlotOffset run) { return BindlessBufferMutable(uint, b.Buffer, run.Slot) + run.Offset; }
inline bool FieldTakes(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, uint handle) { return pc.Restrict.Slot == InvalidSlot || FieldWords(b, pc.Restrict)[handle] != 0u; }

// Boundary: edges with fewer than two faces, the vertices on them, or the faces touching them.
kernel void AttributeFieldBoundary(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    const ConnectivityView conn{b, pc.Connectivity};
    bool boundary = false;
    if (pc.Domain == Element::Edge) boundary = conn.EdgeFaceCount(handle) < 2u;
    else if (pc.Domain == Element::Vertex) {
        conn.ForEachIncidentEdge(handle, [&](uint e) { boundary = boundary || conn.EdgeFaceCount(e) < 2u; });
    } else {
        const uint2 range = conn.FaceHalfedges(handle);
        for (uint h = range.x; h < range.y; ++h) boundary = boundary || conn.EdgeFaceCount(conn.Edge(h)) < 2u;
    }
    FieldWriteBool(b, pc, handle, boundary);
}

// Plane Side: vertices on the side, or edges and faces whose vertices all are.
inline bool FieldVertexOnSide(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, ConnectivityView conn, uint v) {
    const float3 p = FieldPosition(b, pc, conn, v);
    return pc.Side == PlaneSideOf(dot(float3(pc.PlaneNormal), p) - pc.PlaneOffset, p, pc.Tolerance);
}
kernel void AttributeFieldPlaneSide(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    const ConnectivityView conn{b, pc.Connectivity};
    bool selected = true;
    if (pc.Domain == Element::Vertex) selected = FieldVertexOnSide(b, pc, conn, handle);
    else if (pc.Domain == Element::Edge) {
        const uint2 ends = conn.EdgeEnds(handle);
        selected = FieldVertexOnSide(b, pc, conn, ends.x) && FieldVertexOnSide(b, pc, conn, ends.y);
    } else {
        const uint2 range = conn.FaceHalfedges(handle);
        for (uint h = range.x; h < range.y; ++h) selected = selected && FieldVertexOnSide(b, pc, conn, conn.VertexOf(h));
    }
    FieldWriteBool(b, pc, handle, selected);
}

kernel void AttributeFieldLabelInit(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    FieldWords(b, pc.Labels)[handle] = pc.Parity ? handle << 1u : handle;
    FieldWords(b, pc.Flags)[handle] = 0u;
    if (pc.Restrict.Slot != InvalidSlot) FieldWords(b, pc.Restrict)[handle] = 0u;
}
kernel void AttributeFieldMarkList(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ListCount) return;
    FieldWords(b, pc.Restrict)[FieldWords(b, pc.List)[i]] = 1u;
}
// Each label falls to the least of its own, its neighbors' and its label's label, until a round changes nothing.
kernel void AttributeFieldLink(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset || !FieldTakes(b, pc, handle)) return;
    const ConnectivityView conn{b, pc.Connectivity};
    device uint *labels = FieldWords(b, pc.Labels);
    const uint current = labels[handle];
    const uint shift = pc.Parity ? 1u : 0u, parity = current & shift;
    uint least = min(current, labels[current >> shift] ^ parity);
    FieldNeighbors(pc, conn, handle, [&](uint n) { if (FieldTakes(b, pc, n)) least = min(least, labels[n] ^ shift); });
    if (least >> shift < current >> shift) {
        labels[handle] = least;
        FieldWords(b, pc.Changed)[0] = 1u;
    }
}
kernel void AttributeFieldMarkSeeds(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset || !FieldSeed(b, pc, handle)) return;
    FieldWords(b, pc.Flags)[FieldWords(b, pc.Labels)[handle]] = 1u;
}
kernel void AttributeFieldComponents(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    FieldWriteBool(b, pc, handle, FieldWords(b, pc.Flags)[FieldWords(b, pc.Labels)[handle]] != 0u);
}

// Each element starts from its seed, or its complement when shrinking.
kernel void AttributeFieldGrowInit(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    FieldWords(b, pc.Labels)[handle] = FieldSeed(b, pc, handle) != (pc.Complement != 0u) ? 1u : 0u;
}
// Each element writes the union of its own and its neighbors' flags from the previous round.
kernel void AttributeFieldGrow(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    const ConnectivityView conn{b, pc.Connectivity};
    device const uint *from = FieldWords(b, pc.Labels);
    uint flagged = from[handle];
    FieldNeighbors(pc, conn, handle, [&](uint n) { flagged |= from[n]; });
    FieldWords(b, pc.Flags)[handle] = flagged;
}
kernel void AttributeFieldGrowOutput(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint handle = BlockHandle(i, b, pc.Blocks);
    if (handle == InvalidOffset) return;
    FieldWriteBool(b, pc, handle, (FieldWords(b, pc.Labels)[handle] != 0u) != (pc.Complement != 0u));
}

struct FieldTriangles {
    ElementValues<const packed_uint3> Triangles;
    uint First, Count;
    uint3 operator[](uint k) const { return uint3(Triangles[First + k]); }
};
inline FieldTriangles FieldFaceTriangles(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, ConnectivityView conn, uint f) {
    const uint2 range = conn.FaceHalfedges(f);
    return {ReadElements<packed_uint3>(b, pc.TriangleSlot, pc.Connectivity.Tables.Triangles), BindlessBuffer(uint, b.ObjectIdBuffer, pc.FaceTriangleStartSlot)[conn.FaceAddress(f)], range.y - range.x - 2u};
}
inline bool FieldTriangleHas(uint3 triangle, uint corner) { return any(triangle == corner); }
inline float3 FieldTriangleNormal(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, ConnectivityView conn, uint3 triangle) {
    const float3 p0 = FieldPosition(b, pc, conn, conn.VertexOf(triangle.x)), p1 = FieldPosition(b, pc, conn, conn.VertexOf(triangle.y)), p2 = FieldPosition(b, pc, conn, conn.VertexOf(triangle.z));
    return normalize(cross(p1 - p0, p2 - p0));
}
// Writes the normal of the triangle across the edge from corner `a` to corner `c` of face `f`, and returns false at a boundary.
inline bool FieldNeighborNormal(device const BindlessSet &b, constant AttributeFieldPushConstants &pc, ConnectivityView conn, uint f, uint3 triangle, uint a, uint c, thread float3 &normal) {
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
kernel void AttributeFieldCurvature(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeFieldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint v = BlockHandle(i, b, pc.Blocks);
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
            for (uint c = 0u; c < 3u; ++c) position[c] = FieldPosition(b, pc, conn, conn.VertexOf(triangle[c]));
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
