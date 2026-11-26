#pragma once
#include "gpu/Types.h"

#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

struct Mesh;
struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
}

// Connectivity around listed elements gathered on the GPU, each element's items at First[i] to First[i + 1].
template<typename T> struct Gathered {
    std::vector<uint32_t> First;
    std::vector<T> Items;
    std::span<const T> Of(size_t i) const { return std::span{Items}.subspan(First[i], First[i + 1u] - First[i]); }
};
// Other, OtherFace and OtherVertex hold another face's halfedge along the edge in either direction, with its face and end vertex.
// Absent fields hold InvalidOffset.
struct FaceCorner {
    uint32_t Halfedge, Vertex, Edge, Opposite, OppositeFace, Sides, Other, OtherFace, OtherVertex;
};
// From is the corner's origin vertex, and Face and Opposite hold InvalidOffset when absent.
struct CornerFacts {
    uint32_t Vertex, From, Edge, Face, Next, Previous, Opposite;
};
// Other is the edge's far vertex, and Face holds InvalidOffset for a wire.
struct VertexEdge {
    uint32_t Edge, Other, Sides, Halfedge, Face;
};
// Face holds InvalidOffset for a wire.
struct VertexCorner {
    uint32_t Corner, Face, Valence;
};
Gathered<FaceCorner> FaceCorners(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> faces);
Gathered<VertexEdge> VertexEdges(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
// Records the gathers in the record's lane of the chain, which gather with the other pending lanes' gathers, and call `then` with the items after the submits that complete them.
void RecordFaceCorners(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const uint32_t> faces, std::function<void(Gathered<FaceCorner>)> then);
void RecordVertexEdges(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const uint32_t> vertices, std::function<void(Gathered<VertexEdge>)> then);
// The listed vertices' positions, faces' or vertices' base normals, read on the host once the work that writes them completes.
std::vector<vec3> VertexPositions(const MeshStore &, uint32_t id, std::span<const uint32_t> vertices);
std::vector<vec3> FaceNormals(const MeshStore &, uint32_t id, std::span<const uint32_t> faces);
std::vector<vec3> VertexNormals(const MeshStore &, uint32_t id, std::span<const uint32_t> vertices);
std::vector<uint32_t> FacePrimitives(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> faces);
// Listed elements of one record.
struct RecordElements {
    uint32_t Id;
    std::span<const uint32_t> Elements;
};
// An edge's ends, the origin and vertex of its first halfedge.
struct EdgeEnds {
    uint32_t From, To;
};
// Each listed record's results over its listed elements, gathered in one submit per scratch budget.
std::vector<std::vector<uint32_t>> FacePrimitives(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> faces);
std::vector<std::vector<vec3>> VertexPositions(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> vertices);
std::vector<std::vector<vec3>> VertexNormals(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> vertices);
std::vector<std::vector<vec3>> FaceNormals(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> faces);
std::vector<std::vector<uint32_t>> FaceValences(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> faces);
std::vector<std::vector<EdgeEnds>> EdgeEndpoints(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> edges);
std::vector<std::vector<CornerFacts>> Corners(const MeshStore &, const MeshPipelines &, std::span<const RecordElements> corners);
// Each record's largest vertex coordinate magnitude, in one submit.
std::vector<float> VertexExtents(const MeshStore &, const MeshPipelines &, std::span<const uint32_t> ids);
// Query points on one record.
struct RecordPoints {
    uint32_t Id;
    std::span<const vec3> Points;
};
// Each point's nearest live vertex of the record closer than the radius, the lowest handle on ties, or InvalidOffset.
// Records the query in the record's lane of the chain, which gathers with the other pending lanes' queries, and calls `then` with the nearest vertices after the submit that completes them.
void RecordNearestVertices(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const vec3> points, float radius, std::function<void(std::vector<uint32_t>)> then);
// Each query point's nearest live vertex of its record as a vertex ordinal, the lowest handle on ties, in one submit.
std::vector<std::vector<uint32_t>> NearestVertexOrdinals(const MeshStore &, const MeshPipelines &, std::span<const RecordPoints>);
// Each listed record's derived triangle corners as vertex handles, gathered in one submit per scratch budget.
std::vector<std::vector<uint32_t>> TriangleVertices(const MeshStore &, const MeshPipelines &, std::span<const uint32_t> ids);
// Each listed record's derived triangle corners as vertex ordinals, gathered in one submit per scratch budget.
std::vector<std::vector<uint32_t>> OrdinalTriangles(const MeshStore &, const MeshPipelines &, std::span<const uint32_t> ids);
// Each listed record's vertex positions in ordinal order, gathered in one submit per scratch budget.
std::vector<std::vector<vec3>> OrdinalPositions(const MeshStore &, const MeshPipelines &, std::span<const uint32_t> ids);
struct FaceLoops {
    std::vector<uint32_t> Faces;
    Gathered<FaceCorner> Corners;
    std::vector<vec3> Positions;
};
// Gathers the face loops and positions of the listed records in two submits per scratch budget, and visits each record's loops in list order.
void VisitFaceLoops(const MeshStore &, const MeshPipelines &, std::span<const uint32_t> ids, const std::function<void(uint32_t index, FaceLoops &&)> &visit);
FaceLoops ReadFaceLoops(const MeshStore &, const MeshPipelines &, const Mesh &);

// The record's connectivity around a set of vertices, gathered from the GPU once, whose queries take record handles and answer only within it.
class LocalMesh {
public:
    // The local mesh around the sorted vertices of `set`, from their edges and fans, the corners of the faces of their fans, and the facts of the first halfedges of their edges without a face corner.
    LocalMesh(const MeshStore &, uint32_t id, std::span<const uint32_t> set, Gathered<VertexEdge>, Gathered<VertexCorner>, std::span<const uint32_t> faces, Gathered<FaceCorner>, std::span<const uint32_t> firsts, std::span<const CornerFacts> facts);

    std::span<const VertexEdge> Edges(uint32_t vertex) const { return VertexEdgesOf.Of(VertexIndex.at(vertex)); }
    std::span<const VertexCorner> Fan(uint32_t vertex) const { return VertexCornersOf.Of(VertexIndex.at(vertex)); }
    std::span<const FaceCorner> Corners(uint32_t face) const { return FaceCornersOf.Of(FaceIndex.at(face)); }
    uint32_t Valence(uint32_t face) const { return uint32_t(Corners(face).size()); }
    const CornerFacts &At(uint32_t corner) const { return CornerOf.at(corner); }
    vec3 Position(uint32_t vertex) const { return PositionOf.at(vertex); }
    // The face sides between two vertices, one of them in the set, or zero without an edge.
    uint32_t Sides(uint32_t a, uint32_t b) const;

private:
    std::unordered_map<uint32_t, uint32_t> VertexIndex, FaceIndex;
    Gathered<VertexEdge> VertexEdgesOf;
    Gathered<VertexCorner> VertexCornersOf;
    Gathered<FaceCorner> FaceCornersOf;
    std::unordered_map<uint32_t, CornerFacts> CornerOf;
    std::unordered_map<uint32_t, vec3> PositionOf;
    std::unordered_map<uint64_t, uint32_t> SidesOf;
};
// Records the reads of the local mesh around the vertices in the record's lanes of the chain, each once the chain completed the reads it follows, and calls `then` with it once they completed.
void RecordLocalMesh(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const uint32_t> vertices, std::function<void(LocalMesh)> then);
