#pragma once
#include "gpu/Types.h"
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
    ExtrudeEdges = 8,
    ExtrudeFacesIndividual = 9,
    DuplicateGeometry = 10,
    SplitGeometry = 11,
    KeepSelectedFaces = 12,
    DissolveVertices = 13,
    // Join the face pairs across each selected edge and drop endpoints left with two edges.
    DissolveEdges = 14,
    DissolveFaces = 15,
    // Cut every selected edge Param0 times and split the faces around the cuts by Blender's patterns.
    Subdivide = 16,
    Triangulate = 17,
    TrisToQuads = 18,
    // Fan every selected face around a new center vertex offset Param0 along its normal.
    Poke = 19,
    FlipNormals = 20,
    EdgeSplit = 21,
    // Inset the selected region by Param0 and lift it Param1 along the vertex normals.
    InsetRegion = 22,
    InsetIndividual = 23,
    // Append the edges and faces described by the primitive list.
    AddPrimitives = 24,
    // Merge each selected vertex into the lowest selected vertex within Param0 of it.
    MergeByDistance = 25,
    MergeCollapse = 26,
    // Collapse edges shorter than Param0 whose endpoints are both selected.
    DissolveDegenerate = 27,
    // Dissolve selected edges below Param0's face angle, then simplify selected degree-two chains.
    DissolveLimited = 28,
    // Duplicate the selected faces flipped and pushed Param0 along the vertex normals, with a rim of quads along the region boundary.
    Solidify = 29,
    ConnectVertices = 30,
    // Bevel the selected edges by width Param0 with Steps segments.
    BevelEdges = 31,
    // Bevel the selected vertices by width Param0 with Steps segments.
    BevelVertices = 32,
    // Dissolve selected edges and split each resulting region between the vertices following their ends.
    RotateEdges = 33,
    Wireframe = 34,
    SplitNonplanarFaces = 35,
    SplitConcaveFaces = 36,
    // Replace source faces with planned polygons, optionally retiring listed vertices.
    ReplaceFaces = 37,
    Decimate = 39,
    ExtrudeVertices = 40,
    // Combine the closed surfaces of the source and a second source by the BooleanOperation in Param0 into a new mesh.
    Boolean = 41,
    // March the surface where the source's lattice points, holding a field in Param0Attribute, cross the level in Param1 into a new mesh.
    LevelSet = 42,
};
enum class BooleanOperation : uint32_t {
    Union,
    Difference,
    Intersection,
};

// Primitive-list header: vertex count, boundary length, span and attribute source.
// The boundary handles, the primitive count and the loops of vertex and source corner pairs follow it.
GPU_CONSTANT uint32_t TopologyPrimitiveHeaderWords = 4u;

GPU_CONSTANT uint32_t TopologyFlagNewLoopsOnly = 1u; // Subdivide: select only the new loops
GPU_CONSTANT uint32_t TopologyFlagPlaneCuts = 2u; // Subdivide: cut where edges cross the job's plane
GPU_CONSTANT uint32_t TopologyFlagEvenOffset = 1u; // Inset: keep the inset width across corners
GPU_CONSTANT uint32_t TopologyFlagTransformCopies = 8u; // Extrude and duplicate: move the copies through the job's transform
GPU_CONSTANT uint32_t TopologyFlagSelectCopies = 64u; // Edge split: select only the first copy at each split vertex
GPU_CONSTANT uint32_t TopologyFlagKeepVertices = 128u; // Dissolve edges: leave the endpoints in place even with two edges left
GPU_CONSTANT uint32_t TopologyFlagSelectAll = 256u;
GPU_CONSTANT uint32_t TopologyFlagListSelects = 512u; // The job's list holds the selected vertices of a dissolve
GPU_CONSTANT uint32_t TopologyFlagScreenCuts = 1024u; // Subdivide: cut where edges cross the knife segment in the job's screen space
GPU_CONSTANT uint32_t TopologyFlagAllBoundaries = 32768u;
GPU_CONSTANT uint32_t TopologyFlagDelimitMaterial = 65536u;
GPU_CONSTANT uint32_t TopologyFlagDelimitSharp = 131072u;
GPU_CONSTANT uint32_t TopologyFlagDelimitUV = 262144u;
GPU_CONSTANT uint32_t TopologyFlagListCounts = 524288u; // Subdivide: the job's list holds each cut edge's cut count, then manifold's partition tables for the faces whose counts differ
GPU_CONSTANT uint32_t TopologyFlagFacePoints = 1048576u; // Subdivide: every face splits into a quad per corner around a new face point, and the output follows Catmull-Clark's masks
GPU_CONSTANT uint32_t TopologyFlagBezier = 2097152u; // Subdivide: new vertices move onto manifold's cubic Bezier patches through the derived corner normals
GPU_CONSTANT uint32_t TopologyFlagListJunctions = 8388608u; // Subdivide: the job's list pairs each edge with the vertex on its interior, which enters the faces on the edge in place of a new vertex
GPU_CONSTANT uint32_t TopologyFlagDropFlaps = 16777216u; // Merge: coincident faces winding opposite ways drop in pairs
GPU_CONSTANT uint32_t TopologyFlagFaceSelection = 16384u; // Limited dissolve: protects vertices and edges incident to unselected faces
GPU_CONSTANT uint32_t TopologyFlagWireBoundary = 2048u;
GPU_CONSTANT uint32_t TopologyFlagWireRelative = 4096u;
GPU_CONSTANT uint32_t TopologyFlagWireReplace = 8192u;
GPU_CONSTANT uint32_t TopologyFlagEdgeLists = TopologyFlagListCounts | TopologyFlagListJunctions;
// Subdivide: each cut edge splits once.
GPU_CONSTANT uint32_t TopologyFlagSingleCut = TopologyFlagPlaneCuts | TopologyFlagScreenCuts | TopologyFlagListJunctions;

inline bool TopologyDeletesEdges(MeshTopologyOp op) {
    return op == MeshTopologyOp::DeleteEdges || op == MeshTopologyOp::DeleteOnlyEdgesFaces || op == MeshTopologyOp::DeleteLoose;
}
inline bool TopologyIsDelete(MeshTopologyOp op) {
    return uint32_t(op) <= uint32_t(MeshTopologyOp::DeleteLoose);
}
inline bool TopologyCopiesSelection(MeshTopologyOp op) {
    return op == MeshTopologyOp::DuplicateGeometry || op == MeshTopologyOp::SplitGeometry;
}
inline bool TopologyClassifiesEdges(MeshTopologyOp op) {
    return TopologyIsDelete(op) || TopologyCopiesSelection(op) || op == MeshTopologyOp::ExtrudeRegion;
}
inline bool TopologyExtrudesSides(MeshTopologyOp op) { return op == MeshTopologyOp::ExtrudeEdges || op == MeshTopologyOp::ExtrudeRegion; }
inline bool TopologyIsMerge(MeshTopologyOp op) {
    return op == MeshTopologyOp::MergeAtTarget || op == MeshTopologyOp::MergeByDistance || op == MeshTopologyOp::MergeCollapse || op == MeshTopologyOp::DissolveDegenerate || op == MeshTopologyOp::Decimate;
}
inline bool TopologyIsDissolve(MeshTopologyOp op) {
    return op == MeshTopologyOp::DissolveVertices || op == MeshTopologyOp::DissolveEdges || op == MeshTopologyOp::DissolveFaces || op == MeshTopologyOp::DissolveLimited || op == MeshTopologyOp::RotateEdges;
}
inline bool TopologyMapsWireEdges(MeshTopologyOp op) {
    return TopologyClassifiesEdges(op) || TopologyIsMerge(op) || TopologyIsDissolve(op) || op == MeshTopologyOp::ExtrudeEdges || op == MeshTopologyOp::Subdivide;
}
inline bool TopologyIsBevel(MeshTopologyOp op) { return op == MeshTopologyOp::BevelEdges || op == MeshTopologyOp::BevelVertices; }
// Operators whose output lines can join at the same endpoints, of which a line core keeps one.
inline bool TopologyJoinsLines(MeshTopologyOp op) {
    return TopologyIsMerge(op) || op == MeshTopologyOp::DissolveVertices || op == MeshTopologyOp::DissolveEdges || op == MeshTopologyOp::DissolveLimited;
}
inline bool TopologyIterates(MeshTopologyOp op) {
    return TopologyIsDissolve(op) || op == MeshTopologyOp::TrisToQuads || op == MeshTopologyOp::MergeByDistance || op == MeshTopologyOp::MergeCollapse || op == MeshTopologyOp::DissolveDegenerate;
}
inline bool TopologyDisplaces(MeshTopologyOp op, uint32_t flags) {
    return op == MeshTopologyOp::Boolean || op == MeshTopologyOp::InsetRegion || op == MeshTopologyOp::InsetIndividual || op == MeshTopologyOp::Poke || op == MeshTopologyOp::Solidify || op == MeshTopologyOp::Wireframe || op == MeshTopologyOp::AddPrimitives || op == MeshTopologyOp::Decimate || TopologyIsBevel(op) || (flags & TopologyFlagTransformCopies) != 0u;
}
inline bool TopologyCreatesRecord(MeshTopologyOp op) { return op == MeshTopologyOp::KeepSelectedFaces || op == MeshTopologyOp::Boolean || op == MeshTopologyOp::LevelSet; }
// The operator whose marks an operator reuses: an extrude for an inset, a duplicate for a solidify, and itself otherwise.
inline MeshTopologyOp TopologyBaseOp(MeshTopologyOp op) {
    switch (op) {
        case MeshTopologyOp::InsetRegion: return MeshTopologyOp::ExtrudeRegion;
        case MeshTopologyOp::InsetIndividual: return MeshTopologyOp::ExtrudeFacesIndividual;
        case MeshTopologyOp::Solidify: return MeshTopologyOp::DuplicateGeometry;
        default: return op;
    }
}
