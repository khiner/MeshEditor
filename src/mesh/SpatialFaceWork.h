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

// Exact GPU face predicates over canonical candidate membership in UMA.
// Geometry and attributes stay in the canonical arenas, independent of render caches.
struct SpatialFaceWork {
    // Records candidate face membership.
    SpatialFaceWork(state::Scene &, mtl::ComputeChain &, const MeshTopologyTask &);
    // Records the exact predicate and sorts its affected face work.
    void RecordFaces(state::Scene &, mtl::ComputeChain &);

    ElementWork Faces;
    uint32_t Count{};
    uint32_t CandidateCount{}, CandidateBlocks{}, VisitedNodes{};

private:
    SpatialFaceQueryPushConstants Pc{};
    Range State{}; // Reserved word and topology error flag.
};
