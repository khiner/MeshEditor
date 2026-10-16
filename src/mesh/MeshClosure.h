#pragma once

#include "gpu/Element.h"
#include "gpu/ElementWork.h"

#include <span>
#include <vector>

namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Scene;
} // namespace state
template<typename T> struct BufferArena;

// Bound closure elements by Count and their corner incidence by Incidence.
// LoopFans bounds fan corners of face-seed vertices.
// Host seeds list unique ascending vertices; All and closure-level seeds omit the list.
struct ClosureSeed {
    ElementWork Work{};
    uint32_t Count{}, Incidence{}, LoopFans{};
    std::vector<uint32_t> Vertices{};
    bool All{};
};

// Store sparse closure sets in chain scratch.
// Primitive levels include face loops, wire pairs, and retained vertices.
// Vertex levels include incident corners, faces, edges, and next corners or wire partners.
// Host bounds allow dependent vertex levels to record before submission.
// Read counts and derived seeds after submission.
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
// All live elements of a domain, bounded by the record's counts.
ClosureSeed EncodeAllSeed(state::Scene &, mtl::ComputeChain &, uint32_t id, Element);
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
