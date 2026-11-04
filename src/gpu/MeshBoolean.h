#pragma once
#include "gpu/Collider.h"
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyOp.h"

// Scratch word offsets for a boolean between meshes P and Q in one source record.
// Vertices, faces, triangles and triangle halfedges number P's before Q's, and the halfedge of triangle t's corner i is 3t + i.
// An intersection is a halfedge of one mesh crossing a triangle of the other, and P's halfedges' intersections precede Q's.
struct BooleanJob {
    BooleanOperation Operation DEFAULT();
    GpuArray<ElementWork, 2> VertexWork DEFAULT(), FaceWork DEFAULT();
    // Per mesh: its vertices, faces and derived triangles.
    GpuArray<uint32_t, 2> Vertices DEFAULT(), Faces DEFAULT(), Triangles DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot), TriangleSlot DEFAULT(InvalidSlot);
    uint32_t FaceTriangleStartSlot DEFAULT(InvalidSlot), VertexNormalSlot DEFAULT(InvalidSlot), FaceNormalSlot DEFAULT(InvalidSlot);
    ConnectivityRef Connectivity DEFAULT();
    // The transform taking the second mesh's positions into the first's space.
    mat3 Rotation DEFAULT();
    vec3 Translation DEFAULT();
    uint32_t Intersections DEFAULT(), FirstIntersections DEFAULT();
    // The totals of the count scan's polygon halfedges, partial edge entries and new edge entries.
    uint32_t PolygonHalfedges DEFAULT(), PartialEntries DEFAULT(), GroupEntries DEFAULT();
    // Per vertex: position and the displacement that decides its ties, in P's space.
    uint32_t Positions DEFAULT(), Displacements DEFAULT();
    // Per mesh: the ordered bits of its positions' minimum then maximum on each axis.
    uint32_t Bounds DEFAULT();
    // Per face then a terminator: its first triangle.
    uint32_t FaceTriangleFirst DEFAULT();
    // Per triangle: its vertices, canonical corners, face, normal, and the count of polygon triangles it emits.
    uint32_t TriangleVertices DEFAULT(), TriangleCorners DEFAULT(), TriangleFaces DEFAULT(), TriangleNormals DEFAULT(), PolygonTriangleCounts DEFAULT();
    // Per halfedge: its pair, and the canonical halfedge of its side when the side is a face edge.
    uint32_t Pairs DEFAULT(), EdgeSources DEFAULT();
    GpuArray<ColliderRuns, 2> Colliders DEFAULT(); // Per mesh, over its triangles
    // Per halfedge then a terminator: its first intersection, and per intersection: halfedge, triangle, signed crossing, position.
    uint32_t IntersectionFirst DEFAULT(), IntersectionRecords DEFAULT();
    // Per vertex: its component label and winding number, then two indirect dispatches and the changed word of the label rounds.
    uint32_t Labels DEFAULT(), Windings DEFAULT(), LabelArguments DEFAULT();
    // Per vertex then per intersection: its signed output count.
    uint32_t Inclusion DEFAULT();
    // Scan counts for polygon halfedges, partial edges, intersection edges, and output vertices, followed by a terminator.
    uint32_t Counts DEFAULT();
    // Per output vertex: the vertex or intersection it copies.
    uint32_t OutputEntries DEFAULT();
    // Per triangle: its next polygon halfedge slot.
    // Then per polygon halfedge: start and end output vertices and source canonical halfedge.
    uint32_t Cursors DEFAULT(), Polygons DEFAULT();
    // Edge position entries, four words each, of partial and new edges.
    uint32_t PartialEdges DEFAULT(), NewEdges DEFAULT();
    // Per polygon halfedge: four words of loop links, seven of ear-clipping links and six floats of ear-clipping values.
    // The ear-clipping runs hold two nodes per halfedge for joined holes.
    uint32_t LoopLinks DEFAULT(), LoopNodes DEFAULT(), LoopValues DEFAULT();
    // Two triangles per polygon halfedge, three corner references each, then the total triangle count.
    uint32_t PolygonTriangles DEFAULT(), TriangleTotal DEFAULT();
};
