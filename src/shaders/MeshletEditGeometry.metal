#ifndef MESHLET_EDIT_GEOMETRY_MSL
#define MESHLET_EDIT_GEOMETRY_MSL

#include "MeshletEditEdgeEncoding.metal"
#include "MeshletNonTriangle.metal"

struct MeshletEditEdgeGeometry {
    float4 Clip0, Clip1;
    uint Edge, Vertex0, Vertex1;
};

inline uint MeshletPackedEditEdge(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    const thread MeshletWork &work, uint local_triangle, uint edge_corner
) {
    const uint source_triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
        work.Meshlet.TriangleOffset + local_triangle
    ];
    return BindlessBuffer(uint, bindless.Buffer, pc.MeshletEditEdgeSlot)[
        work.Draw.EditEdgeOffset + source_triangle * 3u + edge_corner
    ];
}

inline MeshletEditEdgeGeometry ResolveMeshletLineEdge(
    const thread Scene &scene, const thread MeshletWork &work,
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc, uint element
) {
    device const uint *element_ids = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot);
    const uint edge = element_ids[work.Meshlet.TriangleOffset + element];
    const uint vertex0 = NonTriangleVertexId(
        bindless, pc.MeshletVertexSlot, work.Meshlet, MeshPrimitiveTopology_Line, element, 0u
    );
    const uint vertex1 = NonTriangleVertexId(
        bindless, pc.MeshletVertexSlot, work.Meshlet, MeshPrimitiveTopology_Line, element, 2u
    );
    const Transform world = MeshletWorld(scene, work.Draw);
    return {
        MeshletPosition(scene, work.Draw, world, vertex0),
        MeshletPosition(scene, work.Draw, world, vertex1),
        edge, vertex0, vertex1,
    };
}

inline MeshletEditEdgeGeometry ResolveMeshletEditEdge(
    const thread Scene &scene, const thread MeshletWork &work,
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    uint local_triangle, uint edge_corner, uint packed_edge
) {
    device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
    const uint triangle_base = MeshletLocalTriangleOffset(work.Meshlet) + local_triangle * 3u;
    const uint local0 = uint(triangles[triangle_base + edge_corner] & MeshletGeometryEncoding_LocalIndexMask);
    const uint local1 = uint(triangles[triangle_base + (edge_corner + 1u) % 3u] & MeshletGeometryEncoding_LocalIndexMask);
    const uint packed0 = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local0);
    const uint packed1 = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local1);
    const uint vertex0 = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed0);
    const uint vertex1 = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed1);
    const Transform world = MeshletWorld(scene, work.Draw);
    return {
        MeshletPosition(scene, work.Draw, world, vertex0),
        MeshletPosition(scene, work.Draw, world, vertex1),
        packed_edge & MeshletEditEdgeEncoding_EdgeMask,
        vertex0,
        vertex1,
    };
}

inline bool ResolveMeshletEditEdgeCandidate(
    const thread Scene &scene, const thread MeshletWork &work,
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    uint element, uint edge_corner, thread MeshletEditEdgeGeometry &geometry
) {
    if (element >= work.Meshlet.TriangleCount) return false;
    const uint topology = MeshletPrimitiveTopology(work.Meshlet);
    if (topology == MeshPrimitiveTopology_Line && edge_corner == 0u) {
        geometry = ResolveMeshletLineEdge(scene, work, bindless, pc, element);
        return true;
    }
    if (topology != MeshPrimitiveTopology_Triangle) return false;
    const uint packed_edge = MeshletPackedEditEdge(bindless, pc, work, element, edge_corner);
    if (packed_edge == INVALID_OFFSET) return false;
    geometry = ResolveMeshletEditEdge(scene, work, bindless, pc, element, edge_corner, packed_edge);
    return true;
}

#endif
