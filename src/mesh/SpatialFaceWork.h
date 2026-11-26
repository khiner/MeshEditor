#pragma once

#include "gpu/SpatialFaceQueryJob.h"

struct MeshTopologyTask;
namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

// Exact GPU face predicates over canonical candidate membership in UMA.
// Geometry and attributes stay in the canonical arenas, independent of render caches.
struct SpatialFaceWork {
    // Records candidate face membership.
    SpatialFaceWork(state::Registry &, mtl::ComputeChain &, const MeshTopologyTask &);
    // Records the exact predicate and sorts its affected face work, with the works the other pending lanes record at the same step.
    void RecordFaces(state::Registry &, mtl::ComputeChain &);

    ElementWork Faces;
    uint32_t Count{};
    uint32_t CandidateCount{}, CandidateBlocks{}, VisitedNodes{};

private:
    SpatialFaceQueryJob Job{};
};
