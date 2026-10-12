#ifndef MESHLET_EDIT_GEOMETRY_MSL
#define MESHLET_EDIT_GEOMETRY_MSL

#include "ConnectivityRead.metal"
#include "MeshletNonTriangle.metal"

struct MeshletEditEdgeGeometry {
    float4 Clip0, Clip1;
    uint Edge, Vertex0, Vertex1;
};

inline uint MeshletEditEdge(device const BindlessSet &bindless, const thread MeshletWork &work, uint local_triangle, uint edge_corner) {
    if (work.Draw.TriangleSlot == InvalidSlot) return InvalidOffset;
    const uint triangle = work.Payloads.TriangleIds[work.Meshlet.TriangleOffset + local_triangle];
    const uint3 corners = uint3(BindlessBuffer(packed_uint3, bindless.Buffer, work.Draw.TriangleSlot)[HandleAddress<MeshElementBlockSize>(bindless, work.Draw.Connectivity.Tables.Triangles, triangle)]);
    const ConnectivityView conn{bindless, work.Draw.Connectivity, work.Draw.FaceCount};
    const uint next = corners[(edge_corner + 1u) % 3u];
    return conn.Next(corners[edge_corner]) == next && conn.EdgeFirst(next) ? conn.Edge(next) : InvalidOffset;
}

inline MeshletEditEdgeGeometry ResolveMeshletLineEdge(const thread Scene &scene, const thread MeshletWork &work, uint element) {
    const uint edge = work.Payloads.TriangleIds[work.Meshlet.TriangleOffset + element];
    const uint vertex0 = NonTriangleVertexId(work.Payloads, work.Meshlet, uint(MeshPrimitiveTopology::Line), element, 0u);
    const uint vertex1 = NonTriangleVertexId(work.Payloads, work.Meshlet, uint(MeshPrimitiveTopology::Line), element, 2u);
    const Transform world = MeshletWorld(scene, work.Draw);
    return {
        MeshletPosition(scene, work.Draw, world, vertex0),
        MeshletPosition(scene, work.Draw, world, vertex1),
        edge, vertex0, vertex1,
    };
}

inline MeshletEditEdgeGeometry ResolveMeshletEditEdge(const thread Scene &scene, const thread MeshletWork &work, uint local_triangle, uint edge_corner, uint edge) {
    const MeshletTriangle triangle = work.Payloads.LocalTriangles[work.Meshlet.LocalTriangleOffset + local_triangle];
    const uint local0 = uint(triangle[edge_corner] & uint(MeshletGeometryEncoding::LocalIndexMask));
    const uint local1 = uint(triangle[(edge_corner + 1u) % 3u] & uint(MeshletGeometryEncoding::LocalIndexMask));
    const uint source0 = MeshletSourceVertex(work.Payloads, work.Meshlet, local0);
    const uint source1 = MeshletSourceVertex(work.Payloads, work.Meshlet, local1);
    const uint vertex0 = MeshletVertexId(scene, work.Draw, uint(MeshPrimitiveTopology::Triangle), source0);
    const uint vertex1 = MeshletVertexId(scene, work.Draw, uint(MeshPrimitiveTopology::Triangle), source1);
    const Transform world = MeshletWorld(scene, work.Draw);
    return {
        MeshletPosition(scene, work.Draw, world, vertex0),
        MeshletPosition(scene, work.Draw, world, vertex1),
        edge,
        vertex0,
        vertex1,
    };
}

inline bool ResolveMeshletEditEdgeCandidate(
    const thread Scene &scene, const thread MeshletWork &work, device const BindlessSet &bindless,
    uint element, uint edge_corner, thread MeshletEditEdgeGeometry &geometry
) {
    if (element >= work.Meshlet.TriangleCount) return false;
    const uint topology = MeshletPrimitiveTopology(work.Meshlet);
    if (topology == uint(MeshPrimitiveTopology::Line) && edge_corner == 0u) {
        geometry = ResolveMeshletLineEdge(scene, work, element);
        return !EditElementHidden(scene,work.Draw,Element::Edge,geometry.Edge);
    }
    if (topology != uint(MeshPrimitiveTopology::Triangle)) return false;
    const uint edge = MeshletEditEdge(bindless, work, element, edge_corner);
    if (edge == InvalidOffset || EditElementHidden(scene,work.Draw,Element::Edge,edge)) return false;
    geometry = ResolveMeshletEditEdge(scene, work, element, edge_corner, edge);
    return true;
}

#endif
