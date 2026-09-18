#pragma once
#include "gpu/Types.h"
// Edit-mode topology operators, each a transform from a source mesh and its selection to a new mesh.
enum class MeshTopologyOp : uint32_t {
    DeleteVertices = 0,
    DeleteEdges = 1,
    DeleteFaces = 2,
    DeleteOnlyEdgesFaces = 3,
    DeleteOnlyFaces = 4,
    DeleteLoose = 5,
    // Merge every selected vertex into the job's target vertex at the target position.
    MergeAtTarget = 6,
    // Move the selected faces onto copied boundary vertices with side quads where they border unselected faces, or duplicate a region that borders nothing.
    ExtrudeRegion = 7,
    // Extrude every selected edge into a quad on copied vertices, leaving the faces in place.
    ExtrudeEdges = 8,
    // Extrude each selected face on its own copied vertices with its own side quads, removing the original.
    ExtrudeFacesIndividual = 9,
    // Duplicate the selected faces on copied vertices, keeping the originals.
    DuplicateFaces = 10,
    // Detach the selected faces from the unselected ones by copying the vertices they share.
    SplitFaces = 11,
    KeepSelectedFaces = 12,
    // Join the faces around each selected vertex and drop the vertex.
    DissolveVertices = 13,
    // Join the face pairs across each selected edge and drop endpoints left with two edges.
    DissolveEdges = 14,
    // Join each connected region of selected faces into one face.
    DissolveFaces = 15,
    // Cut every selected edge Param0 times and split the faces around the cuts by Blender's patterns.
    Subdivide = 16,
    // Split every selected face into triangles, quads along their better diagonal and n-gons by ear clipping.
    Triangulate = 17,
    // Join each pair of selected triangles that are each other's lowest-cost partner into a quad.
    TrisToQuads = 18,
    // Fan every selected face around a new center vertex offset Param0 along its normal.
    Poke = 19,
    FlipNormals = 20,
    // Split vertices along the selected edges so the faces on each side stop sharing them.
    EdgeSplit = 21,
    // Inset the selected region by Param0 and lift it Param1 along the vertex normals.
    InsetRegion = 22,
    InsetIndividual = 23,
    // Append the faces the job's list describes, each a run of vertex indices.
    AddFaces = 24,
    // Merge each selected vertex into the lowest selected vertex within Param0 of it.
    MergeByDistance = 25,
    // Merge each connected run of selected vertices into one at a center the host computes.
    MergeCollapse = 26,
    // Collapse every edge shorter than Param0 into its lower vertex.
    DissolveDegenerate = 27,
    // Dissolve edges between selected faces whose normals differ by under Param0 radians, then vertices left with two nearly collinear edges.
    DissolveLimited = 28,
    // Duplicate the selected faces flipped and pushed Param0 along the vertex normals, with a rim of quads along the region boundary.
    Solidify = 29,
    // Split every face with two or more selected vertices along chords between consecutive selected corners.
    ConnectVertices = 30,
    // Bevel the selected edges by width Param0 with Param1 segments.
    BevelEdges = 31,
    // Bevel the selected vertices by width Param0.
    BevelVertices = 32,
};

// Flags a job's operator reads.
GPU_CONSTANT uint32_t TopologyFlagLoopCutSelect = 1u; // Subdivide: select only the new loop
GPU_CONSTANT uint32_t TopologyFlagPlaneCuts = 2u; // Subdivide: cut where edges cross the job's plane
GPU_CONSTANT uint32_t TopologyFlagListCuts = 4u; // Subdivide: cut the edges the job's list names at its parameters
GPU_CONSTANT uint32_t TopologyFlagEvenOffset = 1u; // Inset: keep the inset width across corners
GPU_CONSTANT uint32_t TopologyFlagTransformCopies = 8u; // Extrude and duplicate: move the copies through the job's transform
GPU_CONSTANT uint32_t TopologyFlagFlipCopies = 16u; // Duplicate: reverse the copies' winding
GPU_CONSTANT uint32_t TopologyFlagPlaneSide = 32u; // Delete faces: delete faces centered on the plane's negative side instead of the selection
GPU_CONSTANT uint32_t TopologyFlagRipSelectCopies = 64u; // Edge split: select only the first copy at each split vertex, for a rip
GPU_CONSTANT uint32_t TopologyFlagKeepVertices = 128u; // Dissolve edges: leave the endpoints in place even with two edges left
GPU_CONSTANT uint32_t TopologyFlagSelectAll = 256u; // Every element counts as selected
GPU_CONSTANT uint32_t TopologyFlagListSelects = 512u; // The job's list names the selected edges of a subdivide, or the selected vertices otherwise
GPU_CONSTANT uint32_t TopologyFlagScreenCuts = 1024u; // Subdivide: cut where edges cross the knife segment in the job's screen space

inline bool TopologyIsMerge(MeshTopologyOp op) {
    return op == MeshTopologyOp::MergeAtTarget || op == MeshTopologyOp::MergeByDistance || op == MeshTopologyOp::MergeCollapse || op == MeshTopologyOp::DissolveDegenerate;
}
inline bool TopologyIsDissolve(MeshTopologyOp op) {
    return op == MeshTopologyOp::DissolveVertices || op == MeshTopologyOp::DissolveEdges || op == MeshTopologyOp::DissolveFaces || op == MeshTopologyOp::DissolveLimited;
}
inline bool TopologyIsBevel(MeshTopologyOp op) { return op == MeshTopologyOp::BevelEdges || op == MeshTopologyOp::BevelVertices; }
// The in-place scans a scan pass selects by its parameter.
enum TopologyScan : uint32_t {
    ScanCounts = 0, // The three count arrays, quantity-major over the count blocks
    ScanCustomNormals = 1, // The custom normal mask popcounts
};

// Operators that repeat a label or target pass until no value changes.
inline bool TopologyIterates(MeshTopologyOp op) {
    return TopologyIsDissolve(op) || op == MeshTopologyOp::TrisToQuads || op == MeshTopologyOp::MergeByDistance || op == MeshTopologyOp::MergeCollapse || op == MeshTopologyOp::DissolveDegenerate;
}
// Whether the gather adds a displacement to output vertices.
inline bool TopologyDisplaces(MeshTopologyOp op, uint32_t flags) {
    return op == MeshTopologyOp::InsetRegion || op == MeshTopologyOp::InsetIndividual || op == MeshTopologyOp::Poke || op == MeshTopologyOp::Solidify || op == MeshTopologyOp::AddFaces || TopologyIsBevel(op) || (flags & TopologyFlagTransformCopies) != 0u;
}
// The operator whose marks an operator reuses: an extrude for an inset, a duplicate for a solidify, and itself otherwise.
inline MeshTopologyOp TopologyBaseOp(MeshTopologyOp op) {
    switch (op) {
        case MeshTopologyOp::InsetRegion: return MeshTopologyOp::ExtrudeRegion;
        case MeshTopologyOp::InsetIndividual: return MeshTopologyOp::ExtrudeFacesIndividual;
        case MeshTopologyOp::Solidify: return MeshTopologyOp::DuplicateFaces;
        default: return op;
    }
}
