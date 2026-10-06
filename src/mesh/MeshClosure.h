#pragma once

#include "gpu/Element.h"
#include "gpu/ElementWork.h"

#include <span>
#include <vector>

namespace mtl {
struct ComputeChain;
}
namespace state {
struct Scene;
}
template<typename T> struct BufferArena;

// Canonical vertices, faces or edges a closure level reads, as work of at most Count elements.
// Incidence bounds the fan corners of vertices, the loop corners of faces, or the fan corners of edge endpoints.
// LoopFans bounds the fan corners of face seeds' loop vertices.
// A seed read on the host lists, ascending and unique, the vertices whose fans those bounds count, which are its vertices, its edges' endpoints or its faces' loop vertices.
// All marks a seed of every live element, which lists none.
// A closure level's seeds list no vertices.
struct ClosureSeed {
    ElementWork Work{};
    uint32_t Count{}, Incidence{}, LoopFans{};
    std::vector<uint32_t> Vertices{};
    bool All{};
};

// One closure level's sparse canonical sets in a chain's scratch.
// A primitive level holds face loops and explicit wire pairs with their vertices and edges, and any retained vertices.
// A vertex level holds its vertices with their fan corners, each fan corner's face and edge, and each fan corner's next corner, or a line corner's pair, and its edge.
// A face or edge level's vertex seed is bounded on the host, so the vertex level it seeds records before the chain submits.
// Counts, face and edge incidences and their seeds are valid once the chain has submitted.
struct MeshClosure {
    std::array<ElementWork, 4> Elements{}; // Vertices, halfedges, faces, edges.
    std::array<uint32_t, 4> Counts{};
    std::array<uint32_t, 4> Bounds{}; // Host bounds on the counts, which size dispatches before the counts are read.
    uint32_t VertexFans{}; // A face or edge level's host bound on the fan corners of its vertices.

    // Records the loop corners of the level's faces or the fan corners of its edges' endpoints, which size the next level.
    void EncodeIncidence(state::Scene &, mtl::ComputeChain &, uint32_t id, Element);
    void Finish(const mtl::ComputeChain &);
    ClosureSeed Seed(Element) const;

private:
    std::array<uint32_t, 2> IncidenceWords{InvalidOffset, InvalidOffset}; // Faces, edges
    std::array<uint32_t, 2> Incidences{};
};

// Edge seeds contribute only loose wires; face-owned edges arrive through the face seeds.
MeshClosure EncodePrimitiveClosure(state::Scene &, mtl::ComputeChain &, uint32_t id, const ClosureSeed &faces, const ClosureSeed &wires = {}, const ClosureSeed &retained = {}, bool retain_isolated_only = false);
MeshClosure EncodeVertexClosure(state::Scene &, mtl::ComputeChain &, uint32_t id, const ClosureSeed &vertices);
// The endpoints of seed edges, as a vertex seed.
ClosureSeed EncodeEdgeVertices(state::Scene &, mtl::ComputeChain &, uint32_t id, const ClosureSeed &edges);
// The selected elements of a domain, or all its live elements, with host bounds read from the canonical masks and connectivity.
ClosureSeed EncodeSelectionSeed(state::Scene &, mtl::ComputeChain &, uint32_t id, Element, bool select_all);
// Listed live handles of a domain, finished on the host.
ClosureSeed ListSeed(state::Scene &, mtl::ComputeChain &, uint32_t id, Element, std::span<const uint32_t> handles);
// Finished face work, with its incidences read on the host.
ClosureSeed FaceSeed(const state::Scene &, uint32_t id, const BufferArena<uint32_t> &, ElementWork faces);
// The faces or edges around the host vertices of a seed read on the host, as `work`, that domain's work of a vertex level over that seed.
// Its bounds and vertices are read on the host, so the face or edge level over it records before the chain submits.
ClosureSeed AroundVertices(const state::Scene &, uint32_t id, Element, ElementWork work, const ClosureSeed &vertices);

// The derived triangles of finished faces, checked against the mesh's face and triangle ownership.
// Count is valid once the chain has submitted.
struct FaceTriangles {
    ElementWork Triangles{};
    uint32_t Count{};

    void Finish(const mtl::ComputeChain &);

private:
    friend FaceTriangles EncodeFaceTriangles(state::Scene &, mtl::ComputeChain &, uint32_t, const ClosureSeed &);
    uint32_t TotalWord{InvalidOffset};
};
FaceTriangles EncodeFaceTriangles(state::Scene &, mtl::ComputeChain &, uint32_t id, const ClosureSeed &faces);
