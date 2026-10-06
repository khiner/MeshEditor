#pragma once

#include "Range.h"
#include "gpu/ElementWork.h"
#include "gpu/SpatialFaceQueryPushConstants.h"

struct MeshTopologyTask;
namespace mtl {
struct ComputeChain;
}
namespace state {
struct Scene;
}

// Exact GPU face membership for geometric topology predicates.
// Traversal prunes the live meshlet spans.
// Canonical positions are read only for faces in intersecting finest meshlets.
// No geometry values are copied to the host.
// The face work lives in the chain's scratch, and the queries of one chain share its submits.
// Each query records its candidate count, the chain submits, each records its gather, and the next submit reads its faces.
struct SpatialFaceWork {
    // Records the candidate count of the task's predicate.
    SpatialFaceWork(state::Scene &, mtl::ComputeChain &, const MeshTopologyTask &);
    // Reads the candidate count once the chain has submitted it, and records the gather, expansion and sort of the candidate faces.
    void RecordFaces(state::Scene &, mtl::ComputeChain &);

    ElementWork Faces;
    uint32_t Count{};
    uint32_t CandidateTriangles{}, CandidateMeshlets{};

private:
    SpatialFaceQueryPushConstants Pc{};
    Range State{}; // The candidate triangle count, error flag, gathered triangles and meshlets.
    uint32_t Groups{}, FaceBlocks{};
};
