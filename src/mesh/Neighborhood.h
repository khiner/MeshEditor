#pragma once
#include "gpu/Types.h"

#include <span>
#include <unordered_map>
#include <vector>

struct Mesh;
struct MeshStore;
struct MeshPipelines;

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
Gathered<VertexCorner> VertexCorners(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
std::vector<CornerFacts> Corners(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> corners);
std::vector<vec3> VertexPositions(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
std::vector<vec3> FaceNormals(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> faces);
std::vector<vec3> VertexNormals(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
std::vector<uint32_t> FacePrimitives(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> faces);
// The largest coordinate magnitude over the listed vertices.
float VertexExtent(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);
// Each point's nearest live vertex closer than the radius, the lowest handle on ties, or InvalidOffset.
std::vector<uint32_t> NearestVertices(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const vec3> points, float radius);
std::vector<uint32_t> NearestVertexOrdinals(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const vec3> points);
std::vector<uint32_t> TriangleVertices(const MeshStore &, const MeshPipelines &, uint32_t id);
std::vector<uint32_t> OrdinalTriangles(const MeshStore &, const MeshPipelines &, uint32_t id);
std::vector<vec3> OrdinalPositions(const MeshStore &, const MeshPipelines &, uint32_t id);
struct FaceLoops {
    std::vector<uint32_t> Faces;
    Gathered<FaceCorner> Corners;
    std::vector<vec3> Positions;
};
FaceLoops ReadFaceLoops(const MeshStore &, const MeshPipelines &, const Mesh &);

// The record's connectivity around a set of vertices, gathered from the GPU once, whose queries take record handles and answer only within it.
class LocalMesh {
public:
    LocalMesh(const MeshStore &, const MeshPipelines &, uint32_t id, std::span<const uint32_t> vertices);

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
