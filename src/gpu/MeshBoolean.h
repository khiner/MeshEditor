#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyOp.h"

// One boolean of a first mesh P and a second mesh Q, which one source record holds, as word offsets of runs in the chain's scratch.
// Vertices, faces, triangles and triangle halfedges number P's before Q's, and the halfedge of triangle t's corner i is 3t + i.
// An intersection is a halfedge of one mesh crossing a triangle of the other, and P's halfedges' intersections precede Q's.
struct BooleanJob {
    BooleanOperation Operation DEFAULT();
    GpuArray<ElementWork, 2> VertexWork DEFAULT(), FaceWork DEFAULT();
    // Per mesh: its vertices, faces and derived triangles.
    GpuArray<uint32_t, 2> Vertices DEFAULT(), Faces DEFAULT(), Triangles DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot), CornerSlot DEFAULT(InvalidSlot), TriangleSlot DEFAULT(InvalidSlot);
    uint32_t FaceTriangleStartSlot DEFAULT(InvalidSlot), VertexNormalSlot DEFAULT(InvalidSlot);
    ConnectivityRef Connectivity DEFAULT(); // The source's, which holds both meshes' elements
    // The transform taking the second mesh's positions into the first's space.
    mat3 Rotation DEFAULT();
    vec3 Translation DEFAULT();
    uint32_t Intersections DEFAULT(), FirstIntersections DEFAULT();
    // The totals of the count scan's polygon halfedges, partial edge entries and new edge entries.
    uint32_t PolygonHalfedges DEFAULT(), PartialEntries DEFAULT(), GroupEntries DEFAULT();
    // Per vertex: position and perturbation normal, in P's space.
    uint32_t Positions DEFAULT(), Normals DEFAULT();
    // Per mesh: the ordered bits of its positions' minimum then maximum on each axis.
    uint32_t Bounds DEFAULT();
    // Per face then a terminator: its first triangle.
    uint32_t FaceTriangleFirst DEFAULT();
    // Per triangle: its vertices, canonical corners, face, normal, and the count of polygon triangles it emits.
    uint32_t TriangleVertices DEFAULT(), TriangleCorners DEFAULT(), TriangleFaces DEFAULT(), TriangleNormals DEFAULT(), PolygonTriangleCounts DEFAULT();
    // Per halfedge: its pair, and the canonical halfedge of its side when the side is a face edge.
    uint32_t Pairs DEFAULT(), EdgeSources DEFAULT();
    // Per mesh: Morton keys, sort order and its temporaries, then the radix tree's node boxes, parents, children and refit counters.
    GpuArray<uint32_t, 2> Keys DEFAULT(), Order DEFAULT(), Temporary DEFAULT(), Histogram DEFAULT(), Totals DEFAULT();
    GpuArray<uint32_t, 2> NodeBoxes DEFAULT(), NodeParents DEFAULT(), Children DEFAULT(), Counters DEFAULT();
    // Per halfedge then a terminator: its first intersection, and per intersection: halfedge, triangle, signed crossing, position.
    uint32_t IntersectionFirst DEFAULT(), IntersectionRecords DEFAULT();
    // Per vertex: its component label and winding number, then two indirect dispatches and the changed word of the label rounds.
    uint32_t Labels DEFAULT(), Windings DEFAULT(), LabelArguments DEFAULT();
    // Per vertex then per intersection: its signed output count.
    uint32_t Inclusion DEFAULT();
    // The count scan: polygon halfedges per triangle, partial edge entries per halfedge, new edge entries per intersection side, output vertices per vertex and intersection, then a terminator.
    uint32_t Counts DEFAULT();
    // Per output vertex: the vertex or intersection it copies.
    uint32_t OutputEntries DEFAULT();
    // Per triangle: its next polygon halfedge slot.
    // Then per polygon halfedge: start and end output vertices and source canonical halfedge.
    uint32_t Cursors DEFAULT(), Polygons DEFAULT();
    // Edge position entries, four words each, of partial and new edges.
    uint32_t PartialEdges DEFAULT(), NewEdges DEFAULT();
    // Two slots per polygon halfedge for its face's loops and bridged holes: halfedge references, links and projected points.
    uint32_t LoopRefs DEFAULT(), LoopNext DEFAULT(), LoopPrevious DEFAULT(), LoopPoints DEFAULT(), LoopLinks DEFAULT();
    // Two triangles per polygon halfedge, three corner references each, then the total triangle count.
    uint32_t PolygonTriangles DEFAULT(), TriangleTotal DEFAULT();
};

// The boolean's scratch storage and job, a pass parameter, and the values a scan pass reads.
struct BooleanPushConstants {
    SlotOffset Job DEFAULT();
    uint32_t Parameter DEFAULT(); // A radix shift
    uint32_t ScanValues DEFAULT(), ScanCount DEFAULT(), ScanBlocks DEFAULT();
};
