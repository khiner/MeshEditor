#pragma once

#include "gpu/ElementWork.h"

struct MeshTopologyTask;
namespace mtl { struct ComputeChain; }
namespace state { struct Scene; }

// Exact GPU face membership for geometric topology predicates.
// Traversal prunes the live meshlet spans.
// Canonical positions are read only for faces in intersecting finest meshlets.
// No geometry values are copied to the host.
// The face work lives in the chain's scratch, and the query submits the chain twice.
struct SpatialFaceWork {
    SpatialFaceWork(state::Scene &, mtl::ComputeChain &, const MeshTopologyTask &);

    ElementWork Faces;
    uint32_t Count{};
    uint32_t CandidateTriangles{}, CandidateMeshlets{};
};
